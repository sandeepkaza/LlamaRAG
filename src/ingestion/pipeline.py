"""
src/ingestion/pipeline.py
End-to-end ingestion: Load → Chunk → Embed → Store
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from src.ingestion.document_loader import load_document, load_directory, load_from_url, SUPPORTED_EXTENSIONS
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import get_embeddings
from src.retrieval.vector_store import get_vector_store, add_documents
from src.utils.logger import logger
from config.settings import get_settings


def ingest_documents(
    file_paths: List[str | Path],
    collection_name: str = "default",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    chunk_strategy: Optional[str] = None,
) -> dict:
    """Ingest a list of files. Returns a summary dict."""
    settings = get_settings()
    t0 = time.perf_counter()

    all_docs: List[Document] = []
    for fp in file_paths:
        try:
            all_docs.extend(load_document(fp))
        except Exception as exc:
            logger.warning(f"Skipping {fp}: {exc}")

    if not all_docs:
        return {"status": "error", "message": "No documents could be loaded."}

    chunks = chunk_documents(
        all_docs,
        strategy=chunk_strategy or settings.chunk_strategy,
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
    )

    embeddings = get_embeddings()
    store = get_vector_store(embeddings, collection_name=collection_name)
    add_documents(store, chunks)

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "status": "success",
        "files_processed": len(file_paths),
        "documents_loaded": len(all_docs),
        "chunks_created": len(chunks),
        "collection": collection_name,
        "elapsed_seconds": elapsed,
    }
    logger.success(f"Ingestion complete: {summary}")
    return summary


def ingest_directory(directory: str | Path, collection_name: str = "default", **kwargs) -> dict:
    directory = Path(directory)
    files = [p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    logger.info(f"Ingesting {len(files)} file(s) from '{directory}'")
    return ingest_documents(files, collection_name=collection_name, **kwargs)


def ingest_url(url: str, collection_name: str = "default", **kwargs) -> dict:
    t0 = time.perf_counter()
    docs = load_from_url(url)
    chunks = chunk_documents(docs)
    embeddings = get_embeddings()
    store = get_vector_store(embeddings, collection_name=collection_name)
    add_documents(store, chunks)
    return {
        "status": "success",
        "url": url,
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "collection": collection_name,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
