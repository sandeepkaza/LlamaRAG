"""
src/retrieval/retriever.py
Retrieval strategies: similarity | mmr | hybrid
"""
from __future__ import annotations
from typing import List, Literal, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from config.settings import get_settings
from src.utils.logger import logger

RetrievalStrategy = Literal["similarity", "mmr", "hybrid"]


def retrieve(
    query: str,
    store: VectorStore,
    top_k: Optional[int] = None,
    strategy: Optional[RetrievalStrategy] = None,
    score_threshold: float = 0.0,
) -> List[Document]:
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    strategy = strategy or settings.retrieval_strategy

    logger.info(f"Retrieving top_k={top_k} strategy={strategy} query='{query[:60]}…'")

    if strategy == "similarity":
        docs = _similarity_search(store, query, top_k, score_threshold)
    elif strategy == "mmr":
        docs = _mmr_search(store, query, top_k)
    elif strategy == "hybrid":
        docs = _hybrid_search(store, query, top_k)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    logger.info(f"Retrieved {len(docs)} chunk(s)")
    return docs


def _similarity_search(store, query, top_k, score_threshold) -> List[Document]:
    try:
        results = store.similarity_search_with_relevance_scores(query, k=top_k, score_threshold=score_threshold)
        docs = []
        for doc, score in results:
            doc.metadata["retrieval_score"] = round(score, 4)
            doc.metadata["retrieval_strategy"] = "similarity"
            docs.append(doc)
        return docs
    except Exception:
        return store.similarity_search(query, k=top_k)


def _mmr_search(store, query, top_k) -> List[Document]:
    docs = store.max_marginal_relevance_search(query, k=top_k, fetch_k=min(top_k * 3, 20), lambda_mult=0.5)
    for i, doc in enumerate(docs):
        doc.metadata["retrieval_strategy"] = "mmr"
        doc.metadata["mmr_rank"] = i
    return docs


def _hybrid_search(store, query, top_k) -> List[Document]:
    semantic = _similarity_search(store, query, top_k * 2, 0.0)
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [d.page_content.lower().split() for d in semantic]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
        max_s = max(scores) or 1.0
        combined = []
        for i, doc in enumerate(semantic):
            sem = doc.metadata.get("retrieval_score", 0.5)
            bm25_n = scores[i] / max_s
            doc.metadata["retrieval_score"] = round(0.5 * sem + 0.5 * bm25_n, 4)
            doc.metadata["retrieval_strategy"] = "hybrid"
            combined.append((doc.metadata["retrieval_score"], doc))
        combined.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in combined[:top_k]]
    except ImportError:
        for doc in semantic:
            doc.metadata["retrieval_strategy"] = "hybrid_fallback"
        return semantic[:top_k]


def format_context(documents: List[Document], max_chars: int = 6000) -> str:
    """Format retrieved docs into a context string for the LLM."""
    if not documents:
        return ""
    parts, total = [], 0
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        block = f"[Source {i}: {source}]\n{doc.page_content.strip()}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(block[:remaining] + "…")
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)
