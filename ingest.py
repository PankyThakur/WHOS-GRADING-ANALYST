"""
Ingest pipeline — one command to load a company's SEC filings into Chroma.

Usage:
    python ingest.py AAPL
    python ingest.py MSFT --form 10-K --count 2
"""

import argparse
import sys

from rich.console import Console
from rich.progress import track

from mcps.sec_edgar import get_cik, get_filings, download_filing_text
from utils.chunker import chunk_text
from utils.embeddings import ingest
from utils.bm25_index import build_index, save_index

console = Console()


def run(ticker: str, form_type: str = "10-Q", count: int = 4):
    console.rule(f"[bold cyan]Ingesting {ticker} {form_type} filings[/bold cyan]")

    # ── 1. Ticker → CIK ─────────────────────────────────────
    console.print(f"\n[yellow]→[/yellow] Looking up CIK for [bold]{ticker}[/bold]...")
    cik = get_cik(ticker)
    console.print(f"  CIK: [green]{cik}[/green]")

    # ── 2. CIK → filing list ─────────────────────────────────
    console.print(f"\n[yellow]→[/yellow] Fetching {count} recent {form_type} filings...")
    filings = get_filings(cik, form_type=form_type, count=count)
    for f in filings:
        console.print(f"  [dim]{f['date']}  {f['accession_number']}[/dim]")

    # ── 3. Download → chunk → embed ──────────────────────────
    total_chunks = 0
    all_chunks   = []   # collect all chunks for BM25 index

    for filing in track(filings, description="Processing filings..."):
        source_id = f"{ticker}_{form_type}_{filing['date']}"

        console.print(f"\n[yellow]→[/yellow] Downloading {filing['date']}...")
        text = download_filing_text(filing["accession_number"], cik, filing["primary_document"])
        console.print(f"  Downloaded [green]{len(text):,}[/green] chars")

        chunks = chunk_text(text, source_id=source_id)
        console.print(f"  Split into [green]{len(chunks)}[/green] chunks")

        n = ingest(chunks)
        console.print(f"  Upserted [green]{n}[/green] chunks into Chroma ✓")
        total_chunks += n
        all_chunks.extend(chunks)

    # ── 4. Build BM25 index over all chunks ──────────────────
    console.print(f"\n[yellow]→[/yellow] Building BM25 keyword index...")
    bm25_index = build_index(all_chunks)
    save_index(bm25_index, all_chunks)
    console.print(f"  BM25 index saved ✓ ({len(all_chunks)} chunks indexed)")

    console.rule()
    console.print(
        f"\n[bold green]Done![/bold green] "
        f"{total_chunks} total chunks from {len(filings)} filings "
        f"in Chroma + BM25 index built.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC filings into Chroma")
    parser.add_argument("ticker",          help="Stock ticker, e.g. AAPL")
    parser.add_argument("--form",  default="10-Q", help="Form type (default: 10-Q)")
    parser.add_argument("--count", default=4, type=int, help="Number of filings (default: 4)")
    args = parser.parse_args()

    try:
        run(args.ticker, form_type=args.form, count=args.count)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
