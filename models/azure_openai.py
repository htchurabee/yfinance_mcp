import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable '{name}' is not set. Add it to your .env file or shell environment.")
    return value


@lru_cache

def get_azure_chat_model(model_name: str | None = None, temperature: float = 0.0) -> AzureChatOpenAI:
    """Create and cache an AzureChatOpenAI client using values from the environment."""
    endpoint = _get_required_env("AZURE_OPENAI_ENDPOINT")
    api_key = _get_required_env("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    deployment_name = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment_name:
        raise ValueError(
            "AZURE_OPENAI_DEPLOYMENT_NAME is not set. "
            "Add it to your .env file or pass model_name to get_azure_chat_model()."
        )

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=deployment_name,
        model=deployment_name,
        temperature=temperature,
    )


azure_chat_model = get_azure_chat_model()
