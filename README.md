# 🦙 RAG System — Ollama Edition

A **fully local, production-ready RAG system** — no API keys, no cloud, no cost.  
Powered by **Ollama** for both LLM inference and embeddings, with a beautiful Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=llama)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Details |
|---|---|
| **100% Local** | No API keys needed — runs entirely on your machine |
| **LLM Providers** | Ollama (default) · OpenAI · Anthropic |
| **Embedding Models** | Ollama · sentence-transformers · OpenAI |
| **Vector DBs** | ChromaDB (default) · FAISS · Pinecone · Qdrant |
| **Document Formats** | PDF · DOCX · TXT · Markdown · HTML · CSV · XLSX · URL |
| **Chunking Strategies** | Recursive · Sentence · Token |
| **Retrieval Strategies** | Similarity · MMR · Hybrid (BM25 + semantic) |
| **UI** | Streamlit app with streaming, sources, setup guide |
| **CLI** | `ingest`, `query`, `models`, `setup`, `ui`, `info` |
| **Tests** | Pytest suite |
| **Docker** | One-command deployment with Ollama bundled |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌──────────────────────────────────────────────┐
│           Streamlit UI / CLI                  │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│              RAG Chain                        │
│  1. Condense multi-turn question              │
│  2. Retrieve top-K chunks                     │
│  3. Format context + citations                │
│  4. Stream answer from Ollama                 │
└──────┬───────────────────────────┬────────────┘
       │                           │
       ▼                           ▼
┌─────────────┐           ┌──────────────────┐
│  ChromaDB   │           │  Ollama (local)   │
│  (vectors)  │           │  llama3.2 / etc.  │
└─────────────┘           └──────────────────┘
       ▲
       │  ingestion
┌──────┴──────────────────────────────────────┐
│  Load → Chunk → Embed (nomic-embed-text)    │
│  PDF / DOCX / TXT / MD / HTML / CSV / URL   │
└─────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
rag-system/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py   # Multi-format loader
│   │   ├── chunker.py           # recursive|sentence|token
│   │   ├── embedder.py          # ollama|sentence-transformers|openai
│   │   └── pipeline.py          # load→chunk→embed→store
│   ├── retrieval/
│   │   ├── vector_store.py      # chroma|faiss|pinecone|qdrant
│   │   └── retriever.py         # similarity|mmr|hybrid
│   ├── generation/
│   │   ├── llm.py               # ollama|openai|anthropic
│   │   ├── prompts.py           # RAG + condense prompts
│   │   └── rag_chain.py         # streaming conversational chain
│   ├── ui/
│   │   └── app.py               # Streamlit app (4 tabs)
│   └── utils/
│       └── logger.py
├── config/
│   └── settings.py              # pydantic-settings
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_config.py
├── data/raw/                    # Drop documents here
├── cli.py                       # Typer CLI
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed

### 1. Clone & Install

```bash
git clone https://github.com/sandeepkaza/LlamaRAG.git
cd LlamaRAG

python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Start Ollama & Pull Models

```bash
# Terminal 1 — keep this running
ollama serve

# Terminal 2 — pull the models (one time)
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Configure

```bash
cp .env.example .env
```

The defaults work out of the box for Ollama — no changes needed!

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text

VECTOR_DB=chroma
```

### 4. Run

```bash
streamlit run src/ui/app.py
```

Open http://localhost:8501 → **Ingest** tab → upload docs → **Chat** tab → ask questions! 🎉

---

## 🦙 Recommended Ollama Models

### LLM Models

```bash
ollama pull llama3.2          # Best balance (default)
ollama pull llama3.2:1b       # Fastest, lowest RAM
ollama pull llama3.1          # Higher quality, more RAM
ollama pull mistral           # Great for European languages
ollama pull gemma2            # Google's model, very capable
ollama pull phi3              # Microsoft, very efficient
ollama pull qwen2.5           # Great for multilingual
ollama pull codellama         # Optimized for code
ollama pull deepseek-r1       # Strong reasoning
```

### Embedding Models

```bash
ollama pull nomic-embed-text     # Best quality (default)
ollama pull mxbai-embed-large    # High quality, larger
ollama pull all-minilm           # Fastest, smallest
```

### Good Combos by Use Case

| Use Case | LLM | Embedding |
|---|---|---|
| General Q&A | `llama3.2` | `nomic-embed-text` |
| Low RAM (<8GB) | `llama3.2:1b` | `all-minilm` |
| High quality | `llama3.1` | `nomic-embed-text` |
| Code assistant | `codellama` | `nomic-embed-text` |
| Reasoning | `deepseek-r1` | `nomic-embed-text` |

---

## 🖥️ Streamlit UI

The app has 4 tabs:

**💬 Chat** — Multi-turn Q&A with streaming responses and source citations  
**📥 Ingest** — Upload files or paste a URL, with chunking controls  
**⚙️ Setup** — Live setup checklist showing what's installed and running  
**📊 Stats** — Ingestion history with charts  

Sidebar controls let you switch LLM model, embedding model, vector DB, retrieval strategy, and top-K in real time.

---

## ⌨️ CLI Usage

```bash
# Check Ollama status
python cli.py setup

# List pulled models
python cli.py models

# Ingest a directory
python cli.py ingest --path data/raw/

# Ingest a single file
python cli.py ingest --file report.pdf --collection finance

# Ingest from URL
python cli.py ingest --url https://docs.example.com/page

# Query
python cli.py query "What is the main finding?"

# Query with specific model
python cli.py query "Summarize chapter 3" --strategy mmr --top-k 10 -m llama3.1

# Show current config
python cli.py info

# Launch UI
python cli.py ui
```

---

## 🐳 Docker

```bash
cp .env.example .env

# Start app + Ollama together
docker compose up -d

# Pull models inside the container
docker exec rag-ollama ollama pull llama3.2
docker exec rag-ollama ollama pull nomic-embed-text

# View logs
docker compose logs -f app
```

App at http://localhost:8501

---

## ⚠️ Important: Switching Embedding Models

If you change `EMBEDDING_MODEL` or `EMBEDDING_PROVIDER`, you **must** delete the ChromaDB folder and re-ingest all documents. Different models produce vectors with different dimensions — mixing them causes errors.

```bash
# Windows
rmdir /s /q data\chroma_db

# Mac/Linux
rm -rf data/chroma_db
```

Or use the **Reset ChromaDB** button in the sidebar.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 💡 Programmatic Usage

```python
from src.ingestion.pipeline import ingest_documents
from src.ingestion.embedder import get_embeddings
from src.retrieval.vector_store import get_vector_store
from src.generation.rag_chain import RAGChain

# Ingest
ingest_documents(["report.pdf"], collection_name="finance")

# Query
embeddings = get_embeddings()
store = get_vector_store(embeddings, collection_name="finance")
chain = RAGChain(store, top_k=5, strategy="mmr")

result = chain.invoke("What were the Q3 revenues?")
print(result["answer"])

# Streaming
for chunk in chain.stream("Summarize the report"):
    print(chunk, end="", flush=True)
```

---

## 🔧 Troubleshooting

**`Connection refused` / Ollama not running**
```bash
ollama serve
```

**Model not found**
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

**ChromaDB dimension mismatch** — Delete `data/chroma_db/` and re-ingest after changing embedding model.

**Slow first response** — Ollama loads the model into memory on first use. Subsequent queries are much faster.

**Out of memory** — Switch to a smaller model: `ollama pull llama3.2:1b`

---

## 📄 License

MIT

---

## 🤝 Contributing

```bash
pytest tests/ -v
black src/ cli.py
isort src/ cli.py
```
