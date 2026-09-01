"""
Agent 3 — Judge

Given a claim + retrieved SEC filing evidence, renders a verdict.
Different model family from the extractor to reduce self-preference bias.

Model: Gemini Flash via OpenRouter
"""

import json
import re
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, JUDGE_MODEL

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

SYSTEM_PROMPT = """You are a strict financial auditor fact-checking Wall Street analyst claims against SEC filings.

You will receive:
1. A claim made by an analyst
2. Evidence chunks pulled directly from the company's SEC filings

Your job: determine if the SEC filings support, contradict, or are silent on the claim.

Respond ONLY with a JSON object:
{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one paragraph citing specific numbers from the evidence>",
  "key_evidence": "<the single most relevant sentence from the filing chunks>"
}

Rules:
- SUPPORTED: filing explicitly confirms the claim with matching numbers/facts
- CONTRADICTED: filing contains numbers/facts that directly conflict with the claim
- INSUFFICIENT_EVIDENCE: filing doesn't clearly address the claim either way
- Be strict. Partial matches → INSUFFICIENT_EVIDENCE, not SUPPORTED.
- Always cite specific numbers. Never invent data.

Confidence calibration — use the FULL range, not just 0.95/0.98:
- 0.99: exact number match in the filing (e.g. "$85,269" matches "$85.3B")
- 0.90–0.98: strong match with minor rounding or unit differences
- 0.75–0.89: match is likely but evidence is indirect or partial
- 0.50–0.74: genuinely ambiguous — evidence exists but doesn't clearly confirm or deny
- 0.30–0.49: weak signal, mostly inferring
- Below 0.30: essentially guessing
Do NOT anchor to 0.95 or 0.98. Calibrate per claim based on how directly the evidence addresses it."""


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, handles markdown code blocks."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _build_evidence_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable evidence block."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        dist_str = f"{chunk['distance']:.3f}" if 'distance' in chunk else f"rrf={chunk.get('rrf_score', 0):.4f}"
        lines.append(f"[Chunk {i} | Source: {chunk['source']} | {dist_str}]")
        lines.append(chunk["text"][:800])   # cap each chunk to avoid token blowout
        lines.append("")
    return "\n".join(lines)


def judge(claim: str, evidence_chunks: list[dict]) -> dict:
    """
    Evaluate a claim against SEC filing evidence.

    Args:
        claim:           The factual claim to evaluate.
        evidence_chunks: Retrieved chunks (text, source, distance).

    Returns:
        Dict with keys: verdict, confidence, reasoning, key_evidence.
    """
    evidence_block = _build_evidence_block(evidence_chunks)

    user_message = f"""CLAIM TO VERIFY:
"{claim}"

SEC FILING EVIDENCE:
{evidence_block}

Render your verdict."""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    return _parse_json(raw)
