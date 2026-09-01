"""
Evaluation runner — measures pipeline accuracy against a labeled claim set.

Why do we need this?
Every time you tweak a prompt, change chunking, or adjust retrieval params,
you need to know: did that change help or hurt? Without a labeled test set
you're flying blind — you eyeball two runs and guess. With an eval set you
get a number: accuracy went from 58% to 75%. That number is defensible.

Usage:
    python eval/run_eval.py

Output:
    - Per-claim result (expected vs predicted, ✅ / ❌)
    - Overall accuracy
    - Precision / Recall / F1 per verdict class (like a classification report)

Ground truth: eval/claims.json — hand-labeled claims where the correct
verdict is known from the actual Apple SEC filings we ingested.
"""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Add project root to path so we can import pipeline internals
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import _process_claim

console   = Console()
EVAL_PATH = Path(__file__).parent / "claims.json"


# ── Helpers ──────────────────────────────────────────────────────────────────

VERDICT_COLORS = {
    "SUPPORTED":             "green",
    "CONTRADICTED":          "red",
    "INSUFFICIENT_EVIDENCE": "yellow",
    "OUT_OF_SCOPE":          "dim",
}

def _color(verdict: str) -> str:
    return VERDICT_COLORS.get(verdict, "white")


# ── Core eval logic ───────────────────────────────────────────────────────────

def run_eval():
    console.rule("[bold cyan]Evaluation Run — Who's Grading the Analysts?[/bold cyan]")

    with open(EVAL_PATH) as f:
        eval_claims = json.load(f)

    console.print(
        f"\n[yellow]→[/yellow] Running [bold]{len(eval_claims)}[/bold] eval claims sequentially...\n"
    )

    # ── Run claims one by one — eval prioritises reproducibility over speed ──
    results = []

    for ec in eval_claims:
        try:
            result = _process_claim(ec)
        except Exception as exc:
            console.print(f"  [red]✗ {ec['id']} failed: {exc}[/red]")
            result = {
                **ec,
                "verdict":      "INSUFFICIENT_EVIDENCE",
                "confidence":   0,
                "key_evidence": "-",
            }

        expected  = ec["expected_verdict"]
        predicted = result["verdict"]
        correct   = expected == predicted
        icon      = "✅" if correct else "❌"
        col       = _color(predicted)

        console.print(
            f"  {icon} [{ec['id']}]  "
            f"expected=[bold]{expected}[/bold]  "
            f"got=[{col}]{predicted}[/{col}]"
        )

        results.append({
            **result,
            "expected_verdict": expected,
            "eval_id":          ec["id"],
            "notes":            ec.get("notes", ""),
        })

    _print_report(results)
    return results


def _print_report(results: list[dict]):
    console.rule("[bold]Evaluation Report[/bold]")

    # ── Per-claim table ───────────────────────────────────────────────────────
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("ID",        ratio=1)
    table.add_column("Claim",     ratio=5)
    table.add_column("Expected",  ratio=2)
    table.add_column("Predicted", ratio=2)
    table.add_column("",          ratio=1, justify="center")

    n_correct = 0
    for r in results:
        expected  = r["expected_verdict"]
        predicted = r["verdict"]
        correct   = expected == predicted
        if correct:
            n_correct += 1
        table.add_row(
            r["eval_id"],
            r["claim"][:65],
            f"[{_color(expected)}]{expected}[/{_color(expected)}]",
            f"[{_color(predicted)}]{predicted}[/{_color(predicted)}]",
            "✅" if correct else "❌",
        )

    console.print(table)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    accuracy = n_correct / len(results) if results else 0
    console.print(
        f"\n[bold]Overall accuracy:[/bold] "
        f"[green]{n_correct}[/green] / {len(results)} = "
        f"[bold {'green' if accuracy >= 0.7 else 'yellow'}]{accuracy:.0%}[/bold {'green' if accuracy >= 0.7 else 'yellow'}]\n"
    )

    # ── Precision / Recall / F1 per class ────────────────────────────────────
    active = [v for v in ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
              if any(r["expected_verdict"] == v for r in results)]

    metrics = Table(show_header=True, header_style="bold cyan")
    metrics.add_column("Verdict",   ratio=3)
    metrics.add_column("Precision", ratio=1, justify="right")
    metrics.add_column("Recall",    ratio=1, justify="right")
    metrics.add_column("F1",        ratio=1, justify="right")
    metrics.add_column("Support",   ratio=1, justify="right")

    for verdict in active:
        tp      = sum(1 for r in results if r["expected_verdict"] == verdict and r["verdict"] == verdict)
        fp      = sum(1 for r in results if r["expected_verdict"] != verdict and r["verdict"] == verdict)
        fn      = sum(1 for r in results if r["expected_verdict"] == verdict and r["verdict"] != verdict)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        col = _color(verdict)
        metrics.add_row(
            f"[{col}]{verdict}[/{col}]",
            f"{precision:.0%}",
            f"{recall:.0%}",
            f"{f1:.0%}",
            str(support),
        )

    console.print(metrics)

    # ── Failure analysis ──────────────────────────────────────────────────────
    failures = [r for r in results if r["expected_verdict"] != r["verdict"]]
    if failures:
        console.print(f"\n[bold red]Failures ({len(failures)}):[/bold red]")
        for r in failures:
            console.print(
                f"  [dim]{r['eval_id']}[/dim] "
                f"expected [bold]{r['expected_verdict']}[/bold] "
                f"→ got [bold]{r['verdict']}[/bold]\n"
                f"    Claim:    {r['claim']}\n"
                f"    Evidence: {r.get('key_evidence','?')[:120]}\n"
            )


if __name__ == "__main__":
    run_eval()
