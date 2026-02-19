"""
src/ui/app.py  –  RAG System (Ollama Edition)
Run: streamlit run src/ui/app.py
"""
from __future__ import annotations
import os
import sys
import shutil
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import get_settings
from src.generation.llm import get_llm, PROVIDER_MODELS, OLLAMA_EMBED_MODELS
from src.generation.rag_chain import RAGChain, build_messages
from src.ingestion.pipeline import ingest_documents, ingest_url
from src.ingestion.embedder import get_embeddings
from src.retrieval.vector_store import get_vector_store, delete_collection

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🦙 RAG System",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f0f1a,#1a1a2e); border-right:1px solid #2a2a3e; }

.user-bubble {
    background: linear-gradient(135deg,#6366f1,#818cf8);
    color: white; border-radius: 18px 18px 4px 18px;
    padding: 12px 16px; margin: 6px 0; max-width: 80%;
    margin-left: auto; box-shadow: 0 2px 12px rgba(99,102,241,.3);
}
.assistant-bubble {
    background: #1e1e2e; color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px; margin: 6px 0; max-width: 85%;
    border: 1px solid #2a2a3e; box-shadow: 0 2px 8px rgba(0,0,0,.3);
}
.stat-card {
    background:#1e1e2e; border:1px solid #2a2a3e;
    border-radius:12px; padding:16px; text-align:center;
}
.stat-number { font-size:1.8rem; font-weight:700; color:#818cf8; }
.stat-label  { font-size:.78rem; color:#94a3b8; margin-top:4px; }

.setup-box {
    background: #1a1a2e; border: 1px solid #6366f1;
    border-radius: 12px; padding: 20px; margin: 10px 0;
}
.ok-badge   { color: #22c55e; font-weight: 600; }
.warn-badge { color: #f59e0b; font-weight: 600; }
.err-badge  { color: #ef4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        messages=[],
        collection="default",
        ingestion_stats=[],
        llm_provider="ollama",
        llm_model="llama3.2",
        embed_provider="ollama",
        embed_model="nomic-embed-text",
        vector_db="chroma",
        retrieval_strategy="similarity",
        top_k=5,
        show_sources=True,
        streaming=True,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init()


# ── Ollama health check ───────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def check_ollama() -> tuple[bool, list[str]]:
    """Returns (is_running, list_of_pulled_models)."""
    try:
        import ollama
        models = ollama.list()
        names = [m.model for m in models.models]
        return True, names
    except Exception:
        return False, []


# ── RAG chain builder ─────────────────────────────────────────────────────────
def build_chain() -> RAGChain | None:
    try:
        # Clear embedding cache if settings changed
        get_embeddings.cache_clear()
        embeddings = get_embeddings(
            provider=st.session_state.embed_provider,
            model=st.session_state.embed_model,
        )
        store = get_vector_store(
            embeddings,
            collection_name=st.session_state.collection,
            vector_db=st.session_state.vector_db,
        )
        get_llm.cache_clear()
        return RAGChain(
            vector_store=store,
            collection_name=st.session_state.collection,
            top_k=st.session_state.top_k,
            strategy=st.session_state.retrieval_strategy,
            llm_provider=st.session_state.llm_provider,
            llm_model=st.session_state.llm_model,
        )
    except Exception as exc:
        st.error(f"❌ Failed to initialize: {exc}")
        return None


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🦙 RAG System")

    # ── Ollama status ──────────────────────────────────────────────
    ollama_ok, pulled_models = check_ollama()
    if ollama_ok:
        st.markdown(f'<span class="ok-badge">● Ollama running</span>', unsafe_allow_html=True)
        if pulled_models:
            st.caption(f"{len(pulled_models)} model(s) available")
    else:
        st.markdown('<span class="err-badge">● Ollama not running</span>', unsafe_allow_html=True)
        st.caption("Run: `ollama serve`")

    st.divider()

    # ── LLM Settings ──────────────────────────────────────────────
    st.markdown("### 🤖 LLM")
    provider = st.selectbox("Provider", ["ollama", "openai", "anthropic"],
                            index=["ollama", "openai", "anthropic"].index(st.session_state.llm_provider))
    st.session_state.llm_provider = provider

    if provider == "ollama":
        model_options = PROVIDER_MODELS["ollama"]
        # Put pulled models first
        if pulled_models:
            pulled_clean = [m.split(":")[0] if ":" not in m else m for m in pulled_models]
            known_pulled = [m for m in model_options if m in pulled_clean or m in pulled_models]
            others = [m for m in model_options if m not in known_pulled]
            model_options = known_pulled + others
        model = st.selectbox("Model", model_options)
        if not ollama_ok:
            st.warning("⚠️ Start Ollama first: `ollama serve`")
        elif pulled_models and model not in [m.split(":")[0] for m in pulled_models] and model not in pulled_models:
            st.warning(f"⚠️ Pull this model first:\n```\nollama pull {model}\n```")
    else:
        model = st.selectbox("Model", PROVIDER_MODELS[provider])
        if provider == "openai":
            api_key = st.text_input("OpenAI API Key", type="password",
                                    value=os.environ.get("OPENAI_API_KEY", ""))
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            api_key = st.text_input("Anthropic API Key", type="password",
                                    value=os.environ.get("ANTHROPIC_API_KEY", ""))
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key

    st.session_state.llm_model = model

    st.divider()

    # ── Embedding Settings ────────────────────────────────────────
    st.markdown("### 🔢 Embeddings")
    embed_provider = st.selectbox("Embed Provider", ["ollama", "sentence-transformers", "openai"],
                                  index=["ollama", "sentence-transformers", "openai"].index(
                                      st.session_state.embed_provider))

    if embed_provider == "ollama":
        embed_model_opts = OLLAMA_EMBED_MODELS
        if pulled_models:
            pulled_embed = [m for m in pulled_models if any(e in m for e in ["embed", "minilm", "arctic"])]
            embed_model_opts = list(dict.fromkeys(pulled_embed + OLLAMA_EMBED_MODELS))
        embed_model = st.selectbox("Embed Model", embed_model_opts)
        if embed_provider == "ollama" and pulled_models and embed_model not in pulled_models:
            st.warning(f"⚠️ Pull: `ollama pull {embed_model}`")
    elif embed_provider == "sentence-transformers":
        embed_model = st.selectbox("Embed Model", ["all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5", "all-mpnet-base-v2"])
    else:
        embed_model = st.selectbox("Embed Model", ["text-embedding-3-small", "text-embedding-3-large"])

    if embed_provider != st.session_state.embed_provider or embed_model != st.session_state.embed_model:
        st.warning("⚠️ Embedding model changed! Delete `data/chroma_db` and re-ingest documents.")

    st.session_state.embed_provider = embed_provider
    st.session_state.embed_model = embed_model

    st.divider()

    # ── Vector DB ─────────────────────────────────────────────────
    st.markdown("### 🗄️ Vector DB")
    st.session_state.vector_db = st.selectbox("Backend", ["chroma", "faiss", "pinecone", "qdrant"])
    st.session_state.collection = st.text_input("Collection", value=st.session_state.collection)

    st.divider()

    # ── Retrieval ─────────────────────────────────────────────────
    st.markdown("### 🎯 Retrieval")
    st.session_state.retrieval_strategy = st.selectbox(
        "Strategy", ["similarity", "mmr", "hybrid"],
        help="similarity=precise, mmr=diverse, hybrid=keyword+semantic"
    )
    st.session_state.top_k = st.slider("Top-K", 1, 20, st.session_state.top_k)

    st.divider()

    # ── Preferences ───────────────────────────────────────────────
    st.session_state.show_sources = st.toggle("Show sources", value=st.session_state.show_sources)
    st.session_state.streaming = st.toggle("Streaming", value=st.session_state.streaming)

    st.divider()

    # ── Quick Setup Guide ─────────────────────────────────────────
    with st.expander("🚀 Quick Setup"):
        st.markdown(f"""
**1. Install Ollama**
[ollama.com/download](https://ollama.com/download)

**2. Start Ollama**
```
ollama serve
```

**3. Pull models**
```
ollama pull llama3.2
ollama pull nomic-embed-text
```

**4. Ingest & Chat!**
Go to the **Ingest** tab → upload docs → **Chat** tab → ask questions
        """)

    with st.expander("⚠️ Danger Zone"):
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("🔥 Delete collection", use_container_width=True, type="primary"):
            try:
                delete_collection(st.session_state.collection, st.session_state.vector_db)
                st.success("Deleted!")
            except Exception as exc:
                st.error(str(exc))
        if st.button("🗂️ Reset ChromaDB", use_container_width=True):
            settings = get_settings()
            if Path(settings.chroma_persist_dir).exists():
                shutil.rmtree(settings.chroma_persist_dir)
                st.success("ChromaDB cleared! Re-ingest your documents.")
            else:
                st.info("Nothing to clear.")


# ════════════════════════════════════════════════════════════════════
# MAIN TABS
# ════════════════════════════════════════════════════════════════════
tab_chat, tab_ingest, tab_setup, tab_stats = st.tabs(["💬 Chat", "📥 Ingest", "⚙️ Setup", "📊 Stats"])


# ── Source renderer helper ────────────────────────────────────────────────────
def render_sources(sources: list):
    if not sources:
        return
    st.markdown("**📚 Sources:**")
    cols = st.columns(min(len(sources), 3))
    for i, src in enumerate(sources):
        if isinstance(src, dict):
            name, score, preview = src.get("filename", "?"), src.get("score"), src.get("preview", "")
        else:
            name  = src.metadata.get("filename", src.metadata.get("source", "?"))
            score = src.metadata.get("retrieval_score")
            preview = src.page_content[:300]
        label = f"📄 {name}" + (f" ({score:.2f})" if isinstance(score, float) else "")
        with cols[i % len(cols)]:
            with st.expander(label):
                st.caption(preview + "…")


# ════════════════════════════════════════════════════════════════════
# TAB: CHAT
# ════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 Chat with your documents")

    if not ollama_ok and st.session_state.llm_provider == "ollama":
        st.error("🔴 Ollama is not running. Start it with `ollama serve` then refresh.")
        st.stop()

    # Render history
    for msg in st.session_state.messages:
        css_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        icon = "🧑" if msg["role"] == "user" else "🦙"
        st.markdown(f'<div class="{css_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources") and st.session_state.show_sources and msg["role"] == "assistant":
            render_sources(msg["sources"])

    question = st.chat_input("Ask a question about your documents…")

    if question:
        st.markdown(f'<div class="user-bubble">🧑 {question}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": question})

        chain = build_chain()
        if chain is None:
            st.stop()

        chat_history = build_messages(st.session_state.messages[:-1])

        try:
            if st.session_state.streaming:
                placeholder = st.empty()
                full = ""
                for chunk in chain.stream(question, chat_history=chat_history):
                    full += chunk
                    placeholder.markdown(f'<div class="assistant-bubble">🦙 {full}▌</div>', unsafe_allow_html=True)
                placeholder.markdown(f'<div class="assistant-bubble">🦙 {full}</div>', unsafe_allow_html=True)
                sources = chain.last_sources
            else:
                with st.spinner("🦙 Thinking…"):
                    result = chain.invoke(question, chat_history=chat_history)
                full = result["answer"]
                sources = result["sources"]
                st.markdown(f'<div class="assistant-bubble">🦙 {full}</div>', unsafe_allow_html=True)

            if sources and st.session_state.show_sources:
                render_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full,
                "sources": [
                    {
                        "filename": s.metadata.get("filename", s.metadata.get("source", "?")),
                        "score": s.metadata.get("retrieval_score"),
                        "preview": s.page_content[:300],
                    }
                    for s in sources
                ],
            })

        except Exception as exc:
            err = str(exc)
            if "connection" in err.lower() or "refused" in err.lower():
                st.error("🔴 Cannot connect to Ollama. Is it running? Try: `ollama serve`")
            elif "model" in err.lower() and "not found" in err.lower():
                st.error(f"🔴 Model not found. Pull it first:\n```\nollama pull {st.session_state.llm_model}\n```")
            else:
                st.error(f"❌ Error: {exc}")


# ════════════════════════════════════════════════════════════════════
# TAB: INGEST
# ════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.markdown("### 📥 Add Documents to Knowledge Base")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 📁 Upload Files")
        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "md", "html", "htm", "csv", "xlsx"],
            help="Supported: PDF, DOCX, TXT, MD, HTML, CSV, XLSX",
        )

        with st.expander("⚙️ Chunking options"):
            c1, c2, c3 = st.columns(3)
            with c1: chunk_strategy = st.selectbox("Strategy", ["recursive", "sentence", "token"])
            with c2: chunk_size = st.number_input("Size", 100, 4000, 1000, 100)
            with c3: chunk_overlap = st.number_input("Overlap", 0, 500, 200, 50)

        if uploaded and st.button("🚀 Ingest Files", type="primary", use_container_width=True):
            with st.spinner(f"Processing {len(uploaded)} file(s)… (embedding with Ollama may take a moment)"):
                tmp = tempfile.mkdtemp()
                paths = []
                for f in uploaded:
                    dest = Path(tmp) / f.name
                    dest.write_bytes(f.getbuffer())
                    paths.append(dest)

                try:
                    get_embeddings.cache_clear()
                    summary = ingest_documents(
                        paths,
                        collection_name=st.session_state.collection,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_strategy=chunk_strategy,
                    )
                    st.session_state.ingestion_stats.append(summary)
                    if summary["status"] == "success":
                        st.success(
                            f"✅ **{summary['files_processed']}** file(s) → "
                            f"**{summary['chunks_created']}** chunks in **{summary['elapsed_seconds']}s**"
                        )
                    else:
                        st.error(f"❌ {summary.get('message')}")
                except Exception as exc:
                    err = str(exc)
                    if "connection" in err.lower() or "refused" in err.lower():
                        st.error("🔴 Ollama not running. Start it with `ollama serve`")
                    elif "not found" in err.lower():
                        st.error(f"🔴 Embedding model not pulled. Run:\n```\nollama pull {st.session_state.embed_model}\n```")
                    else:
                        st.error(f"❌ {exc}")

        st.markdown("---")
        st.markdown("#### 🌐 Ingest from URL")
        url_in = st.text_input("URL", placeholder="https://example.com/docs/page")
        if url_in and st.button("🌐 Fetch & Ingest", use_container_width=True):
            with st.spinner("Fetching…"):
                try:
                    get_embeddings.cache_clear()
                    summary = ingest_url(url_in, collection_name=st.session_state.collection)
                    st.session_state.ingestion_stats.append(summary)
                    st.success(f"✅ {summary['chunks_created']} chunks in {summary['elapsed_seconds']}s")
                except Exception as exc:
                    st.error(f"❌ {exc}")

    with col2:
        st.markdown("#### 🦙 Ollama Status")
        if ollama_ok:
            st.success("✅ Ollama is running")
            if pulled_models:
                st.markdown("**Pulled models:**")
                for m in pulled_models:
                    icon = "📝" if "embed" in m or "minilm" in m or "arctic" in m else "🦙"
                    st.markdown(f"- {icon} `{m}`")
            else:
                st.warning("No models pulled yet.")
                st.code("ollama pull llama3.2\nollama pull nomic-embed-text")
        else:
            st.error("🔴 Ollama not running")
            st.code("ollama serve")

        st.markdown("#### 📝 How it works")
        st.markdown("""
**Load** → Parse PDF/DOCX/TXT/etc.  
**Chunk** → Split into overlapping pieces  
**Embed** → Ollama converts text to vectors  
**Store** → Saved to ChromaDB locally  
**Query** → Retrieve + Llama3.2 answers  
        """)


# ════════════════════════════════════════════════════════════════════
# TAB: SETUP GUIDE
# ════════════════════════════════════════════════════════════════════
with tab_setup:
    st.markdown("### ⚙️ Setup Guide")

    ollama_ok, pulled = check_ollama()
    llm_model = st.session_state.llm_model
    embed_model = st.session_state.embed_model

    llm_pulled   = any(llm_model in m or m in llm_model for m in pulled)
    embed_pulled = any(embed_model in m or m in embed_model for m in pulled)

    st.markdown("#### Step 1: Install Ollama")
    st.markdown("Download from **[ollama.com/download](https://ollama.com/download)** for Windows/Mac/Linux")

    st.markdown("#### Step 2: Start Ollama")
    col_a, col_b = st.columns(2)
    with col_a:
        st.code("ollama serve", language="bash")
    with col_b:
        if ollama_ok:
            st.markdown('<p class="ok-badge">✅ Ollama is running!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="err-badge">❌ Not running</p>', unsafe_allow_html=True)

    st.markdown("#### Step 3: Pull the LLM model")
    col_a, col_b = st.columns(2)
    with col_a:
        st.code(f"ollama pull {llm_model}", language="bash")
    with col_b:
        if llm_pulled:
            st.markdown(f'<p class="ok-badge">✅ {llm_model} is ready!</p>', unsafe_allow_html=True)
        elif ollama_ok:
            st.markdown(f'<p class="warn-badge">⚠️ Not pulled yet</p>', unsafe_allow_html=True)

    st.markdown("#### Step 4: Pull the embedding model")
    col_a, col_b = st.columns(2)
    with col_a:
        st.code(f"ollama pull {embed_model}", language="bash")
    with col_b:
        if embed_pulled:
            st.markdown(f'<p class="ok-badge">✅ {embed_model} is ready!</p>', unsafe_allow_html=True)
        elif ollama_ok:
            st.markdown(f'<p class="warn-badge">⚠️ Not pulled yet</p>', unsafe_allow_html=True)

    st.markdown("#### Step 5: Configure .env")
    settings = get_settings()
    st.code(f"""
LLM_PROVIDER=ollama
LLM_MODEL={llm_model}
OLLAMA_BASE_URL={settings.ollama_base_url}

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL={embed_model}

VECTOR_DB=chroma
CHROMA_PERSIST_DIR=./data/chroma_db
    """.strip(), language="dotenv")

    st.markdown("#### Step 6: Run")
    st.code("streamlit run src/ui/app.py", language="bash")

    st.markdown("---")
    st.markdown("#### 🔄 Switching embedding models")
    st.warning(
        "If you change the embedding model, you **must** delete `data/chroma_db/` and re-ingest all documents. "
        "Embedding dimensions differ between models — mixing them causes errors. "
        "Use the **Reset ChromaDB** button in the sidebar."
    )

    st.markdown("#### 📦 Popular Ollama model combos")
    st.markdown("""
| Use Case | LLM | Embedding |
|---|---|---|
| Best quality | `llama3.1` | `nomic-embed-text` |
| Fast & light | `llama3.2:1b` | `all-minilm` |
| Coding | `codellama` | `nomic-embed-text` |
| Privacy-first | `phi3` | `all-minilm` |
    """)


# ════════════════════════════════════════════════════════════════════
# TAB: STATS
# ════════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown("### 📊 Ingestion Statistics")

    if not st.session_state.ingestion_stats:
        st.info("No documents ingested yet. Go to the **Ingest** tab to add documents.")
    else:
        stats = st.session_state.ingestion_stats
        total_files  = sum(s.get("files_processed", 1) for s in stats)
        total_chunks = sum(s.get("chunks_created", 0) for s in stats)
        total_docs   = sum(s.get("documents_loaded", 1) for s in stats)
        total_time   = sum(s.get("elapsed_seconds", 0) for s in stats)

        c1, c2, c3, c4 = st.columns(4)
        for col, num, label in [
            (c1, total_files, "Files"),
            (c2, total_docs, "Pages"),
            (c3, total_chunks, "Chunks"),
            (c4, f"{round(total_time,1)}s", "Total Time"),
        ]:
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{num}</div>'
                            f'<div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        import pandas as pd
        df = pd.DataFrame(stats)
        st.dataframe(df, use_container_width=True)
        if "chunks_created" in df.columns:
            st.bar_chart(df["chunks_created"])
