import pytest
from langchain_core.documents import Document
from src.retrieval.retriever import format_context

@pytest.fixture
def docs():
    return [
        Document(page_content="Refunds within 30 days.", metadata={"filename": "policy.pdf", "retrieval_score": 0.9}),
        Document(page_content="Shipping 3-5 days.", metadata={"filename": "faq.txt", "retrieval_score": 0.7}),
    ]

class TestFormatContext:
    def test_includes_sources(self, docs):
        ctx = format_context(docs)
        assert "Source 1" in ctx and "Source 2" in ctx

    def test_max_chars(self, docs):
        ctx = format_context(docs, max_chars=50)
        assert len(ctx) < 500

    def test_empty(self):
        assert format_context([]) == ""
