#!/usr/bin/env python3
"""
cli.py – RAG System CLI (Ollama Edition)

python cli.py ingest --path data/raw/
python cli.py ingest --file report.pdf
python cli.py query "What is the refund policy?"
python cli.py models          # list pulled Ollama models
python cli.py setup           # check setup status
python cli.py ui              # launch Streamlit
"""
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

sys.path.insert(0, str(Path(__file__).parent))

app = typer.Typer(name="rag", help="🦙 RAG System CLI", rich_markup_mode="rich", no_args_is_help=True)
console = Console()


@app.command()
def setup():
    """⚙️ Check Ollama setup status."""
    try:
        import ollama
        models = ollama.list()
        names = [m.model for m in models.models]
        console.print(Panel(
            f"[green]✅ Ollama is running[/green]\n\n"
            f"Pulled models:\n" + "\n".join(f"  • {n}" for n in names),
            title="Ollama Status", border_style="green"
        ))
    except Exception:
        console.print(Panel(
            "[red]❌ Ollama is not running[/red]\n\n"
            "Start it with: [bold]ollama serve[/bold]\n"
            "Then pull models:\n"
            "  [bold]ollama pull llama3.2[/bold]\n"
            "  [bold]ollama pull nomic-embed-text[/bold]",
            title="Ollama Status", border_style="red"
        ))


@app.command()
def models():
    """🦙 List all pulled Ollama models."""
    try:
        import ollama
        result = ollama.list()
        table = Table(title="Pulled Ollama Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="green")
        for m in result.models:
            size = f"{m.size / 1e9:.1f} GB" if m.size else "?"
            table.add_row(m.model, size)
        console.print(table)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")


@app.command()
def ingest(
    path: str = typer.Option(None, "--path", "-p", help="Directory to ingest"),
    file: str = typer.Option(None, "--file", "-f", help="Single file"),
    url: str = typer.Option(None, "--url", "-u", help="URL to ingest"),
    collection: str = typer.Option("default", "--collection", "-c"),
    chunk_size: int = typer.Option(1000),
    chunk_overlap: int = typer.Option(200),
    strategy: str = typer.Option("recursive"),
):
    """📥 Ingest documents into the vector store."""
    from src.ingestion.pipeline import ingest_documents, ingest_directory, ingest_url as _ingest_url

    if not any([path, file, url]):
        console.print("[red]Provide --path, --file, or --url[/red]")
        raise typer.Exit(1)

    with console.status("[bold green]Ingesting…"):
        if path:
            summary = ingest_directory(path, collection_name=collection)
        elif file:
            summary = ingest_documents([file], collection_name=collection,
                                        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                        chunk_strategy=strategy)
        else:
            summary = _ingest_url(url, collection_name=collection)

    if summary["status"] == "success":
        console.print(Panel(
            f"[green]✅ Done![/green]\n\n"
            f"Chunks: [bold]{summary['chunks_created']}[/bold]\n"
            f"Time: [bold]{summary['elapsed_seconds']}s[/bold]\n"
            f"Collection: [bold]{summary['collection']}[/bold]",
            title="Ingestion Summary", border_style="green"
        ))
    else:
        console.print(f"[red]❌ {summary.get('message')}[/red]")


@app.command()
def query(
    question: str = typer.Argument(...),
    collection: str = typer.Option("default", "-c"),
    top_k: int = typer.Option(5, "-k"),
    strategy: str = typer.Option("similarity", "-s"),
    model: str = typer.Option(None, "-m", help="Ollama model override"),
):
    """💬 Ask a question."""
    from src.ingestion.embedder import get_embeddings
    from src.retrieval.vector_store import get_vector_store
    from src.generation.rag_chain import RAGChain

    with console.status("Thinking…"):
        embeddings = get_embeddings()
        store = get_vector_store(embeddings, collection_name=collection)
        chain = RAGChain(store, top_k=top_k, strategy=strategy, llm_model=model)
        result = chain.invoke(question)

    console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="blue"))

    if result["sources"]:
        table = Table(title="Sources")
        table.add_column("File", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Preview", style="dim")
        for doc in result["sources"]:
            name  = doc.metadata.get("filename", "?")
            score = doc.metadata.get("retrieval_score", "—")
            score = f"{score:.3f}" if isinstance(score, float) else str(score)
            table.add_row(name, score, doc.page_content[:80].replace("\n", " ") + "…")
        console.print(table)


@app.command()
def ui():
    """🖥️ Launch Streamlit UI."""
    import subprocess
    subprocess.run(["streamlit", "run", "src/ui/app.py"])


@app.command()
def info():
    """ℹ️ Show current config."""
    from config.settings import get_settings
    s = get_settings()
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    for k, v in [
        ("LLM Provider", s.llm_provider), ("LLM Model", s.llm_model),
        ("Embed Provider", s.embedding_provider), ("Embed Model", s.embedding_model),
        ("Vector DB", s.vector_db), ("Chunk Size", str(s.chunk_size)),
        ("Chunk Overlap", str(s.chunk_overlap)), ("Strategy", s.chunk_strategy),
        ("Top-K", str(s.retrieval_top_k)), ("Ollama URL", s.ollama_base_url),
    ]:
        table.add_row(k, v)
    console.print(table)


if __name__ == "__main__":
    app()
