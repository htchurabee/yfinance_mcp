"""Answer financial questions with Azure OpenAI and the local Yahoo Finance MCP server."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool

from models import get_azure_chat_model

DEFAULT_SERVER_PATH = PROJECT_ROOT / "yahoo-finance-mcp" / "server.py"
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are a financial-data assistant using Yahoo Finance MCP tools.

When a user provides a company name, an ambiguous company reference, or an unknown
ticker, call search_ticker first. Inspect each candidate's symbol, exchange, and
quoteType before selecting a ticker-specific tool. If the results do not uniquely
identify the requested instrument, ask the user to clarify the company, exchange,
or instrument type. Do not invent ticker symbols.

Use returned tool data as the source for factual market claims. State that prices
are market data and may be delayed when that matters to the user's request.
"""


async def load_history(
    checkpointer: AsyncSqliteSaver, checkpoint_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Load the current process's conversation history from a SQLite checkpoint."""
    checkpoint = await checkpointer.aget(checkpoint_config)
    if checkpoint is None:
        return []
    history = checkpoint["channel_values"].get("history", [])
    if not isinstance(history, list):
        raise ValueError("Conversation checkpoint history must be a list.")
    return history


async def save_history(
    checkpointer: AsyncSqliteSaver,
    checkpoint_config: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Save one session's compact conversation history as a SQLite checkpoint."""
    checkpoint = {
        "v": 1,
        "id": str(uuid4()),
        "ts": datetime.now(UTC).isoformat(),
        "channel_values": {"history": history},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"source": "loop", "step": len(history), "parents": {}}
    return await checkpointer.aput(checkpoint_config, checkpoint, metadata, new_versions={})


def history_messages(history: list[dict[str, Any]]) -> list[Any]:
    """Rebuild checkpoint records as chat history for the next model turn."""
    messages: list[Any] = []
    for record in history:
        question = record.get("question")
        answer = record.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        messages.extend([HumanMessage(question), AIMessage(answer)])
        selected_tools = record.get("selected_tools", [])
        if selected_tools:
            messages.append(
                SystemMessage(
                    "Tools used for the immediately preceding answer: "
                    + json.dumps(selected_tools, ensure_ascii=False)
                )
            )
    return messages


def mcp_tool_to_aoai_tool(tool: Tool) -> dict[str, Any]:
    """Convert an MCP tool definition to the Azure OpenAI function-tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


def call_result_to_text(result: CallToolResult) -> str:
    """Extract textual MCP tool content for the next model turn."""
    parts: list[str] = []
    for content in result.content:
        text = getattr(content, "text", None)
        if text is not None:
            parts.append(text)
        elif hasattr(content, "model_dump"):
            parts.append(json.dumps(content.model_dump(mode="json")))
        else:
            parts.append(str(content))
    return "\n".join(parts)


async def answer_question(
    question: str,
    checkpointer: AsyncSqliteSaver,
    checkpoint_config: dict[str, Any],
    server_path: Path = DEFAULT_SERVER_PATH,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> tuple[str, dict[str, Any]]:
    """Use AOAI to select and execute Yahoo Finance MCP tools for one question."""
    if not server_path.is_file():
        raise FileNotFoundError(f"Yahoo Finance MCP server was not found: {server_path}")

    history = await load_history(checkpointer, checkpoint_config)
    prior_history = history_messages(history)
    selected_tools: list[dict[str, Any]] = []
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        cwd=str(server_path.parent),
    )

    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            model = get_azure_chat_model().bind_tools(
                [mcp_tool_to_aoai_tool(tool) for tool in tools]
            )
            messages: list[Any] = [SystemMessage(SYSTEM_PROMPT), *prior_history, HumanMessage(question)]

            for _ in range(max_tool_rounds):
                response = await model.ainvoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    answer = str(response.content)
                    history.append(
                        {
                            "timestamp": datetime.now(UTC).isoformat(),
                            "question": question,
                            "answer": answer,
                            "selected_tools": selected_tools,
                        }
                    )
                    next_checkpoint_config = await save_history(
                        checkpointer, checkpoint_config, history
                    )
                    return answer, next_checkpoint_config

                for tool_call in response.tool_calls:
                    print(f"Calling tool: {tool_call['name']} with arguments: {tool_call['args']}")
                    selected_tools.append(
                        {"name": tool_call["name"], "arguments": tool_call["args"]}
                    )
                    result = await session.call_tool(tool_call["name"], tool_call["args"])
                    print(f"Tool {tool_call['name']} returned result: {result}")
                    messages.append(
                        ToolMessage(
                            content=call_result_to_text(result),
                            tool_call_id=tool_call["id"],
                        )
                    )

    raise RuntimeError(f"AOAI requested more than {max_tool_rounds} rounds of tool calls.")


async def run_conversation(initial_question: str | None) -> None:
    """Run one or more questions with a process-scoped SQLite checkpoint."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        checkpoint_config: dict[str, Any] = {
            "configurable": {"thread_id": "cli-session", "checkpoint_ns": ""}
        }
        question = initial_question

        while True:
            if question is None:
                question = input("質問 (exit で終了): ").strip()
            if question.strip().lower() in {"exit", "quit"}:
                return
            answer, checkpoint_config = await answer_question(
                question, checkpointer, checkpoint_config
            )
            print(answer)
            question = None


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a financial question using the local Yahoo Finance MCP server."
    )
    parser.add_argument("question", nargs="?", help="Natural-language financial question")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    asyncio.run(run_conversation(args.question))


if __name__ == "__main__":
    main()