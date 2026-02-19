"""
src/ingestion/document_loader.py
Loads PDF, DOCX, TXT, MD, HTML, CSV, XLSX into LangChain Documents.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from src.utils.logger import logger

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".csv", ".xlsx", ".xls"}


def load_document(file_path: str | Path) -> List[Document]:
    """Load a single document. Returns list of LangChain Documents."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    logger.info(f"Loading '{path.name}'")

    try:
        docs = _load_by_extension(path, ext)
        for doc in docs:
            doc.metadata.setdefault("source", str(path))
            doc.metadata.setdefault("filename", path.name)
            doc.metadata.setdefault("file_type", ext.lstrip("."))
        logger.success(f"Loaded {len(docs)} page(s) from '{path.name}'")
        return docs
    except Exception as exc:
        logger.error(f"Failed to load '{path.name}': {exc}")
        raise


def _load_by_extension(path: Path, ext: str) -> List[Document]:
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(str(path)).load()

    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(str(path)).load()

    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(str(path), encoding="utf-8").load()

    elif ext == ".md":
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        return UnstructuredMarkdownLoader(str(path)).load()

    elif ext in (".html", ".htm"):
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        return UnstructuredHTMLLoader(str(path)).load()

    elif ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(str(path)).load()

    elif ext in (".xlsx", ".xls"):
        from langchain_community.document_loaders import UnstructuredExcelLoader
        return UnstructuredExcelLoader(str(path)).load()

    else:
        raise ValueError(f"No loader for extension '{ext}'")


def load_directory(directory: str | Path) -> List[Document]:
    """Load all supported documents from a directory recursively."""
    directory = Path(directory)
    all_docs: List[Document] = []
    files = [p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    logger.info(f"Found {len(files)} supported file(s) in '{directory}'")
    for fp in files:
        try:
            all_docs.extend(load_document(fp))
        except Exception as exc:
            logger.warning(f"Skipping '{fp.name}': {exc}")
    logger.success(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def load_from_url(url: str) -> List[Document]:
    """Scrape a webpage and return as Documents."""
    from langchain_community.document_loaders import WebBaseLoader
    logger.info(f"Loading URL: {url}")
    docs = WebBaseLoader(url).load()
    for doc in docs:
        doc.metadata["source"] = url
        doc.metadata["filename"] = url
    return docs
