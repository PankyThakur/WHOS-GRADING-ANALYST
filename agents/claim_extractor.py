"""
Agent 1 — Claim Extractor

Given a raw analyst report (text), extracts factual claims
that can be verified against SEC filings.

Model: GLM-5.2 via OpenRouter
"""

import json
import re
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EXTRACTOR_MODEL

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

SYSTEM_PROMPT = """You are a financial fact-checker.
Given an analyst report, extract every verifiable factual claim.

Return ONLY a JSON array of claims, each with:
- "claim": the exact claim text
- "type": one of ["revenue", "earnings", "guidance", "debt", "margins", "other"]
- "period": the time period referenced (e.g. "Q3 2026", "FY2023") or null

Only include claims that can be verified against SEC filings (10-Q, 10-K).
Exclude opinions, price targets, and buy/sell recommendations.

Granularity rule: one claim = one verifiable fact. Split compound sentences.
"Revenue was $X and margin was Y%" → two separate claims.
Never merge two distinct facts into one claim entry.

CRITICAL: "X, up from Y" constructions contain TWO claims — always split them:
  "iPhone revenue was $85.3B, up from $69.1B" →
    claim 1: "iPhone revenue was $85.3B"   (current period)
    claim 2: "iPhone revenue was $69.1B"   (prior period)
  "Operating cash flow was $53.9B, up from $29.9B a year ago" →
    claim 1: "Operating cash flow was $53.9B"    (current period)
    claim 2: "Operating cash flow was $29.9B"    (prior period, one year ago)

Example output:
[
  {"claim": "Apple Services revenue was $26.3B", "type": "revenue", "period": "Q3 2026"},
  {"claim": "Apple Services revenue was $23.2B in the prior year", "type": "revenue", "period": "Q3 2025"},
  {"claim": "Net income was $24.8 billion", "type": "earnings", "period": "Q3 2026"}
]"""


def _parse_json(text: str) -> list:
    """Extract JSON from LLM response, handles markdown code blocks."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find the first [ ... ] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def extract_claims(analyst_report: str) -> list[dict]:
    """
    Extract verifiable factual claims from an analyst report.

    Args:
        analyst_report: Raw text of the analyst report.

    Returns:
        List of claim dicts with keys: claim, type, period.
    """
    response = client.chat.completions.create(
        model=EXTRACTOR_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Extract claims from this analyst report:\n\n{analyst_report}"},
        ],
        temperature=0.1,   # low temp = consistent structured output
    )

    raw = response.choices[0].message.content
    return _parse_json(raw)
