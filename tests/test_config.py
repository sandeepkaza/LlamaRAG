from config.settings import Settings

def test_defaults():
    s = Settings()
    assert s.llm_provider == "ollama"
    assert s.embedding_provider == "ollama"
    assert s.vector_db == "chroma"
    assert s.chunk_size > 0
    assert s.chunk_overlap < s.chunk_size

def test_ollama_url():
    s = Settings()
    assert s.ollama_base_url.startswith("http")
