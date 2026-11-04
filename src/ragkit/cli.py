"""`rag` command line: ingest, ask, eval, serve, status."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragkit.config import Settings
from ragkit.pipeline import Pipeline

app = typer.Typer(
    help="Self-hosted RAG over private documents, with citations and refusal.", no_args_is_help=True
)
console = Console()


def _pipeline() -> Pipeline:
    return Pipeline.from_settings(Settings())


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, help="File or folder (pdf, docx, html, md, txt)"),
) -> None:
    """Extract, chunk, embed and index documents."""
    report = _pipeline().ingest(path)
    console.print(f"[green]Indexed[/] {report.documents} document(s), {report.chunks} chunk(s).")
    for s in report.skipped:
        console.print(f"[yellow]Skipped (no text):[/] {s}")


@app.command()
def ask(
    question: str,
    json_out: bool = typer.Option(False, "--json", help="Print the answer as JSON"),
    show_sources: bool = typer.Option(True, help="Show cited passages"),
) -> None:
    """Ask a question. The answer cites its sources or refuses."""
    ans = _pipeline().answerer.ask(question)
    if json_out:
        payload = {
            "question": ans.question,
            "answer": ans.text,
            "refused": ans.refused,
            "model": ans.model,
            "citations": [{"ref": c.citation, "text": c.text} for c in ans.citations],
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    style = "yellow" if ans.refused else "bold"
    console.print(f"[{style}]{ans.text}[/]")
    if show_sources and ans.citations:
        console.print()
        for i, c in enumerate(ans.citations, start=1):
            console.print(f"[dim][{i}][/] {c.citation}")


@app.command(name="eval")
def eval_cmd(
    cases: Path = typer.Argument(..., exists=True, help="YAML eval set"),
    min_accuracy: float = typer.Option(0.0, help="Exit 1 if answer accuracy is below this"),
) -> None:
    """Run an eval set and print retrieval / citation / refusal metrics."""
    from ragkit.evaluate import evaluate, load_cases

    report = evaluate(_pipeline().answerer, load_cases(cases))
    table = Table(title=f"Eval — {report.n} cases")
    table.add_column("metric")
    table.add_column("value", justify="right")
    summary = report.summary()
    for k, v in summary.items():
        table.add_row(k, f"{v:.0f}" if k == "cases" else f"{v:.0%}")
    console.print(table)
    for r in report.results:
        mark = (
            "[green]✓[/]"
            if (r.answer_ok and not r.refused) or (r.refused and r.case.expect_refusal)
            else "[red]✗[/]"
        )
        console.print(f"{mark} {r.case.question}\n   [dim]{r.answer[:140]}[/]")
    if summary["answer_accuracy"] < min_accuracy:
        raise typer.Exit(code=1)


@app.command()
def status() -> None:
    """Show the indexed documents."""
    p = _pipeline()
    table = Table(title=f"{p.store.count()} chunks · store={p.settings.store_url} · embed={p.embedder.name}")
    table.add_column("source")
    table.add_column("title")
    table.add_column("chunks", justify="right")
    for _, source, title, n in p.store.documents():
        table.add_row(source, title, str(n))
    console.print(table)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the HTTP API."""
    import uvicorn

    uvicorn.run("ragkit.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
