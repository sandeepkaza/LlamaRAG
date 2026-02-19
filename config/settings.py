"""
config/settings.py - Centralized configuration via pydantic-settings.
All values read from .env file or environment variables.
"""
from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_title: str = "RAG System"
    log_level: str = "INFO"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # LLM
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    llm_model: str = "llama3.2"

    # Optional API keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Embeddings
    embedding_provider: Literal["ollama", "sentence-transformers", "openai"] = "ollama"
    embedding_model: str = "nomic-embed-text"

    # Vector DB
    vector_db: Literal["chroma", "faiss", "pinecone", "qdrant"] = "chroma"
    chroma_persist_dir: str = "./data/chroma_db"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "rag-index"
    pinecone_environment: str = "us-east-1"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_strategy: Literal["recursive", "sentence", "token"] = "recursive"

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_strategy: Literal["similarity", "mmr", "hybrid"] = "similarity"


@lru_cache
def get_settings() -> Settings:
    return Settings()
