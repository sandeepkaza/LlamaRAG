"""
src/ingestion/embedder.py
Embedding factory: ollama (default) | sentence-transformers | openai
"""
from __future__ import annotations
from functools import lru_cache
from langchain_core.embeddings import Embeddings
from config.settings import get_settings
from src.utils.logger import logger


@lru_cache(maxsize=1)
def get_embeddings(provider: str | None = None, model: str | None = None) -> Embeddings:
    """Return a cached LangChain Embeddings instance."""
    settings = get_settings()
    provider = provider or settings.embedding_provider
    model = model or settings.embedding_model

    logger.info(f"Initializing embeddings: provider={provider} model={model}")

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=model,
            base_url=settings.ollama_base_url,
        )

    elif provider == "sentence-transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=settings.openai_api_key,
        )

    else:
        raise ValueError(f"Unknown embedding provider '{provider}'. Choose: ollama | sentence-transformers | openai")


def get_embedding_dimension(embeddings: Embeddings) -> int:
    """Probe embedding dimension with a test string."""
    return len(embeddings.embed_query("probe"))
