"""
src/ingestion/chunker.py
Three chunking strategies: recursive | sentence | token
"""
from __future__ import annotations
from typing import List, Literal
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from config.settings import get_settings
from src.utils.logger import logger

ChunkStrategy = Literal["recursive", "sentence", "token"]


def _build_splitter(strategy: ChunkStrategy, chunk_size: int, chunk_overlap: int):
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    elif strategy == "sentence":
        # Fallback to recursive with sentence-friendly separators if NLTK unavailable
        try:
            from langchain_text_splitters import NLTKTextSplitter
            return NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[". ", "! ", "? ", "\n\n", "\n", " "],
            )
    elif strategy == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")


def chunk_documents(
    documents: List[Document],
    strategy: ChunkStrategy | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    if not documents:
        return []

    settings = get_settings()
    strategy = strategy or settings.chunk_strategy
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    logger.info(f"Chunking {len(documents)} doc(s) | strategy={strategy} size={chunk_size} overlap={chunk_overlap}")

    splitter = _build_splitter(strategy, chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Add chunk index per source
    counter: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_index"] = counter.get(src, 0)
        counter[src] = counter.get(src, 0) + 1

    logger.success(f"Created {len(chunks)} chunks from {len(documents)} document(s)")
    return chunks


def chunk_text(text: str, strategy: ChunkStrategy = "recursive",
               chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    return _build_splitter(strategy, chunk_size, chunk_overlap).split_text(text)
