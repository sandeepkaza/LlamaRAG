"""
src/retrieval/vector_store.py
Vector store factory: chroma (default) | faiss | pinecone | qdrant
"""
from __future__ import annotations
import os
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from config.settings import get_settings
from src.utils.logger import logger


def get_vector_store(
    embeddings: Embeddings,
    collection_name: str = "default",
    vector_db: str | None = None,
) -> VectorStore:
    settings = get_settings()
    backend = vector_db or settings.vector_db
    logger.info(f"Vector store: backend={backend} collection={collection_name}")

    # ── ChromaDB ──────────────────────────────────────────────
    if backend == "chroma":
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

    # ── FAISS ─────────────────────────────────────────────────
    elif backend == "faiss":
        from langchain_community.vectorstores import FAISS
        faiss_path = f"./data/faiss_{collection_name}"
        if os.path.exists(faiss_path):
            logger.info(f"Loading FAISS index from '{faiss_path}'")
            return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        # Bootstrap with a dummy doc — real docs added via add_documents()
        store = FAISS.from_texts(["__init__"], embeddings)
        return store

    # ── Pinecone ──────────────────────────────────────────────
    elif backend == "pinecone":
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index_name = settings.pinecone_index_name
        if index_name not in [i.name for i in pc.list_indexes()]:
            from src.ingestion.embedder import get_embedding_dimension
            pc.create_index(
                name=index_name,
                dimension=get_embedding_dimension(embeddings),
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
            )
        return PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings, namespace=collection_name)

    # ── Qdrant ────────────────────────────────────────────────
    elif backend == "qdrant":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from langchain_qdrant import QdrantVectorStore
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            from src.ingestion.embedder import get_embedding_dimension
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=get_embedding_dimension(embeddings), distance=Distance.COSINE),
            )
        return QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)

    else:
        raise ValueError(f"Unknown vector_db '{backend}'. Choose: chroma | faiss | pinecone | qdrant")


def add_documents(store: VectorStore, documents: List[Document]) -> List[str]:
    logger.info(f"Storing {len(documents)} chunks…")
    ids = store.add_documents(documents)
    if hasattr(store, "save_local"):
        store.save_local("./data/faiss_default")
    logger.success(f"Stored {len(ids)} vectors")
    return ids


def delete_collection(collection_name: str, vector_db: str | None = None) -> None:
    settings = get_settings()
    backend = vector_db or settings.vector_db
    logger.warning(f"Deleting collection '{collection_name}' from {backend}")
    if backend == "chroma":
        import chromadb
        chromadb.PersistentClient(path=settings.chroma_persist_dir).delete_collection(collection_name)
    elif backend == "qdrant":
        from qdrant_client import QdrantClient
        QdrantClient(url=settings.qdrant_url).delete_collection(collection_name)
    else:
        logger.warning(f"Delete not implemented for '{backend}'")
