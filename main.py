"""
Who's Grading the Analysts?
A multi-agent RAG system that fact-checks Wall Street analyst reports
against SEC filings using LLM-as-a-judge.

Usage:
    python main.py
"""

from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]Who's Grading the Analysts?[/bold cyan]\n"
        "[dim]Multi-agent RAG · SEC EDGAR · LLM-as-a-Judge[/dim]",
        border_style="cyan",
    ))

    console.print("\n[yellow]Step 1 scaffold complete.[/yellow] Next up:")
    console.print("  [dim]→ Step 2: SEC EDGAR data pull[/dim]")
    console.print("  [dim]→ Step 3: Chunk + embed into Chroma[/dim]")
    console.print("  [dim]→ Step 4: RAG query end-to-end[/dim]\n")


if __name__ == "__main__":
    main()
