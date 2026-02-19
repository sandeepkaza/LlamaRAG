import pytest
from pathlib import Path
from langchain_core.documents import Document
from src.ingestion.document_loader import load_document, SUPPORTED_EXTENSIONS
from src.ingestion.chunker import chunk_documents, chunk_text

@pytest.fixture
def sample_txt(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello world. " * 200)
    return f

@pytest.fixture
def sample_docs():
    return [
        Document(page_content="First doc. " * 50, metadata={"source": "a.txt"}),
        Document(page_content="Second doc. " * 50, metadata={"source": "b.txt"}),
    ]

class TestLoader:
    def test_load_txt(self, sample_txt):
        docs = load_document(sample_txt)
        assert len(docs) >= 1 and "Hello" in docs[0].page_content

    def test_metadata(self, sample_txt):
        docs = load_document(sample_txt)
        assert docs[0].metadata["file_type"] == "txt"

    def test_unsupported(self, tmp_path):
        bad = tmp_path / "x.xyz"; bad.write_text("x")
        with pytest.raises(ValueError): load_document(bad)

    def test_extensions(self):
        assert {".pdf", ".docx", ".txt"}.issubset(SUPPORTED_EXTENSIONS)

class TestChunker:
    def test_creates_more_chunks(self, sample_docs):
        chunks = chunk_documents(sample_docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > len(sample_docs)

    def test_chunk_index_added(self, sample_docs):
        chunks = chunk_documents(sample_docs, chunk_size=100, chunk_overlap=0)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_empty(self):
        assert chunk_documents([]) == []

    def test_chunk_text(self):
        parts = chunk_text("word " * 500, chunk_size=100)
        assert len(parts) > 1
