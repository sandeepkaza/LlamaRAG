"""
src/generation/rag_chain.py
Conversational RAG chain with streaming, history, and source citations.
"""
from __future__ import annotations
from typing import Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from src.generation.llm import get_llm
from src.generation.prompts import RAG_PROMPT, CONDENSE_PROMPT, NO_CONTEXT_RESPONSE
from src.retrieval.retriever import retrieve, format_context
from src.utils.logger import logger
from config.settings import get_settings


class RAGChain:
    """
    Conversational RAG chain.

    result = chain.invoke("What is the refund policy?")
    result["answer"]   # str
    result["sources"]  # List[Document]

    for chunk in chain.stream("Summarise chapter 3"):
        print(chunk, end="", flush=True)
    sources = chain.last_sources
    """

    def __init__(
        self,
        vector_store: VectorStore,
        collection_name: str = "default",
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.store = vector_store
        self.collection_name = collection_name
        settings = get_settings()
        self.top_k = top_k or settings.retrieval_top_k
        self.strategy = strategy or settings.retrieval_strategy
        self._llm = get_llm(provider=llm_provider, model=llm_model)
        self._parser = StrOutputParser()
        self.last_sources: List[Document] = []

    def invoke(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> dict:
        chat_history = chat_history or []
        standalone = self._condense(question, chat_history)
        sources = retrieve(standalone, self.store, top_k=self.top_k, strategy=self.strategy)
        self.last_sources = sources

        if not sources:
            return {"answer": NO_CONTEXT_RESPONSE, "sources": [], "standalone_question": standalone, "context": ""}

        context = format_context(sources)
        chain = RAG_PROMPT | self._llm | self._parser
        answer = chain.invoke({"question": standalone, "context": context, "chat_history": chat_history})
        logger.info(f"Answer generated ({len(answer)} chars) from {len(sources)} sources")
        return {"answer": answer, "sources": sources, "standalone_question": standalone, "context": context}

    def stream(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> Iterator[str]:
        chat_history = chat_history or []
        standalone = self._condense(question, chat_history)
        sources = retrieve(standalone, self.store, top_k=self.top_k, strategy=self.strategy)
        self.last_sources = sources

        if not sources:
            yield NO_CONTEXT_RESPONSE
            return

        context = format_context(sources)
        chain = RAG_PROMPT | self._llm | self._parser
        yield from chain.stream({"question": standalone, "context": context, "chat_history": chat_history})

    def _condense(self, question: str, chat_history: List[BaseMessage]) -> str:
        if not chat_history:
            return question
        try:
            chain = CONDENSE_PROMPT | self._llm | self._parser
            return chain.invoke({"question": question, "chat_history": chat_history})
        except Exception as exc:
            logger.warning(f"Condensation failed: {exc}")
            return question


def build_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert [{role, content}] dicts to LangChain messages."""
    result = []
    for msg in history:
        if msg["role"] == "user":
            result.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            result.append(AIMessage(content=msg["content"]))
    return result
