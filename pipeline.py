"""
End-to-end pipeline — fact-checks an analyst report against SEC filings.

Day 2 Step 3: Claims now processed in parallel via ThreadPoolExecutor.

Why threads (not processes)?
- LLM calls are I/O-bound: the thread blocks waiting for OpenRouter HTTP response.
- The GIL releases during I/O, so threads genuinely run concurrently here.
- Processes would add serialization overhead for no benefit on I/O-bound work.
- In production this would be Celery + Redis: same fan-out pattern, but tasks
  survive restarts, can be retried, and are visible in a dashboard.

Usage:
    python pipeline.py "Apple Services revenue grew 12% YoY in Q3 2026"
    python pipeline.py --file report.txt
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.claim_extractor import extract_claims
from agents.rag_retriever   import retrieve
from agents.judge           import judge

console = Console()

MAX_WORKERS = 5   # max parallel claims; keeps OpenRouter rate limits happy

VERDICT_COLORS = {
    "SUPPORTED":             "green",
    "CONTRADICTED":          "red",
    "INSUFFICIENT_EVIDENCE": "yellow",
    "OUT_OF_SCOPE":          "dim",
}

VERDICT_ICONS = {
    "SUPPORTED":             "✅",
    "CONTRADICTED":          "❌",
    "INSUFFICIENT_EVIDENCE": "⚠️",
    "OUT_OF_SCOPE":          "🚫",
}


# ── Per-claim worker ─────────────────────────────────────────────────────────

def _process_claim(claim_obj: dict) -> dict:
    """
    Retrieve evidence + judge a single claim.
    Runs in its own thread — no shared mutable state touched here.

    Returns a result dict merging claim_obj fields with verdict fields.
    """
    claim = claim_obj["claim"]

    retrieval = retrieve(claim)

    if retrieval.get("verdict") == "OUT_OF_SCOPE":
        return {
            **claim_obj,
            "verdict":      "OUT_OF_SCOPE",
            "confidence":   0,
            "reasoning":    "No relevant filing data found.",
            "key_evidence": "-",
            "best_distance": retrieval["best_distance"],
        }

    verdict = judge(claim, retrieval["chunks"])

    return {
        **claim_obj,
        **verdict,
        "best_distance": retrieval["best_distance"],
        "n_chunks":      len(retrieval["chunks"]),
    }


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(analyst_report: str):
    console.rule("[bold cyan]Who's Grading the Analysts?[/bold cyan]")

    # ── Step 1: Extract claims (sequential — one LLM call, fast) ────────────
    console.print("\n[yellow]→[/yellow] Extracting claims from analyst report...")
    claims = extract_claims(analyst_report)
    console.print(f"  Found [green]{len(claims)}[/green] verifiable claim(s)\n")

    if not claims:
        console.print("[red]No verifiable claims found.[/red]")
        return []

    # ── Step 2+3: Retrieve + Judge — all claims in parallel ─────────────────
    console.print(
        f"[yellow]→[/yellow] Processing {len(claims)} claims in parallel "
        f"(max {MAX_WORKERS} workers)...\n"
    )

    # index map so we can restore original order after parallel execution
    indexed_claims = list(enumerate(claims))   # [(0, claim_obj), ...]

    results_map: dict[int, dict] = {}
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all claims at once — workers pick them up immediately
        future_to_index = {
            executor.submit(_process_claim, claim_obj): idx
            for idx, claim_obj in indexed_claims
        }

        # Print results as each future completes (not necessarily in order)
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
            except Exception as exc:
                # Don't let one failed claim kill the whole pipeline
                claim_text = claims[idx]["claim"]
                console.print(f"  [red]✗ Claim {idx+1} failed:[/red] {exc}")
                result = {
                    **claims[idx],
                    "verdict":      "INSUFFICIENT_EVIDENCE",
                    "confidence":   0,
                    "reasoning":    f"Processing error: {exc}",
                    "key_evidence": "-",
                }

            results_map[idx] = result

            # Live progress line as each claim finishes
            color  = VERDICT_COLORS.get(result["verdict"], "white")
            icon   = VERDICT_ICONS.get(result["verdict"], "?")
            conf   = f"{result['confidence']:.0%}" if result.get("confidence") else "-"
            dist   = result.get("best_distance", 0)
            chunks = result.get("n_chunks", 0)
            console.print(
                f"  [{color}]{icon} Claim {idx+1}[/{color}]  "
                f"{result['verdict']} ({conf})  "
                f"[dim]chunks={chunks} dist={dist:.3f}[/dim]"
            )

    elapsed = time.perf_counter() - t_start

    # Restore original claim order for the summary table
    results = [results_map[i] for i in range(len(claims))]

    # ── Summary table ────────────────────────────────────────────────────────
    console.print(f"\n[dim]⏱  Finished in {elapsed:.1f}s ({len(claims)} claims)[/dim]")
    console.rule("[bold]Results Summary[/bold]")

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Claim",      ratio=4)
    table.add_column("Period",     ratio=1)
    table.add_column("Verdict",    ratio=2)
    table.add_column("Confidence", ratio=1, justify="right")

    for r in results:
        color = VERDICT_COLORS.get(r["verdict"], "white")
        icon  = VERDICT_ICONS.get(r["verdict"], "?")
        conf  = f"{r['confidence']:.0%}" if r.get("confidence") else "-"
        table.add_row(
            r["claim"][:80],
            r.get("period") or "-",
            f"[{color}]{icon} {r['verdict']}[/{color}]",
            conf,
        )

    console.print(table)

    # ── Key evidence per claim ────────────────────────────────────────────────
    console.print()
    for i, r in enumerate(results, 1):
        if r.get("key_evidence") and r["key_evidence"] != "-":
            console.print(Panel(
                f"[dim]{r['key_evidence']}[/dim]",
                title=f"[bold]Claim {i} key evidence[/bold]",
                border_style="dim",
            ))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fact-check analyst claims against SEC filings")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("claim", nargs="?", help="Analyst report text (inline)")
    group.add_argument("--file", help="Path to a text file containing the analyst report")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            report = f.read()
    else:
        report = args.claim

    try:
        run_pipeline(report)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise
