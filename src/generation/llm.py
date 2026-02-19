"""
src/generation/llm.py
LLM factory: ollama (default) | openai | anthropic
"""
from __future__ import annotations
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from config.settings import get_settings
from src.utils.logger import logger

# Available models per provider shown in the UI
PROVIDER_MODELS: dict[str, list[str]] = {
    "ollama": [
        "llama3.2",
        "llama3.1",
        "llama3.2:1b",
        "mistral",
        "gemma2",
        "phi3",
        "qwen2.5",
        "deepseek-r1",
        "codellama",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ],
}

OLLAMA_EMBED_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "snowflake-arctic-embed",
]


@lru_cache(maxsize=4)
def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    streaming: bool = True,
) -> BaseChatModel:
    settings = get_settings()
    provider = provider or settings.llm_provider
    model = model or settings.llm_model
    logger.info(f"Initializing LLM: provider={provider} model={model}")

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            openai_api_key=settings.openai_api_key,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            streaming=streaming,
            anthropic_api_key=settings.anthropic_api_key,
        )

    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose: ollama | openai | anthropic")
