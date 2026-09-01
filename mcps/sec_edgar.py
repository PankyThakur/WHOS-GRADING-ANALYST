"""
MCP 1 — SEC EDGAR

Fetches 10-Q and 10-K filings for a given company (by ticker).
Uses the free EDGAR REST API — no authentication required, just a User-Agent.
"""

import re
import requests
from config import EDGAR_USER_AGENT, EDGAR_BASE_URL

# EDGAR rate-limits aggressively without a proper User-Agent
HEADERS = {"User-Agent": EDGAR_USER_AGENT, "Accept": "application/json"}


# ── 1. Ticker → CIK ─────────────────────────────────────────

def get_cik(ticker: str) -> str:
    """
    Look up a company's CIK by ticker symbol.

    Args:
        ticker: Stock ticker, e.g. "AAPL"

    Returns:
        Zero-padded 10-digit CIK string, e.g. "0000320193"
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    ticker_upper = ticker.upper()
    for entry in resp.json().values():
        if entry["ticker"] == ticker_upper:
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in EDGAR. Check the symbol.")


# ── 2. CIK → recent filings list ────────────────────────────

def get_filings(cik: str, form_type: str = "10-Q", count: int = 4) -> list[dict]:
    """
    Fetch recent filings metadata for a company.

    Args:
        cik: 10-digit CIK string.
        form_type: "10-Q" or "10-K".
        count: How many recent filings to return.

    Returns:
        List of dicts: accession_number, date, primary_document, url
    """
    url = f"{EDGAR_BASE_URL}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    recent = resp.json()["filings"]["recent"]
    forms            = recent["form"]
    accessions       = recent["accessionNumber"]
    dates            = recent["filingDate"]
    primary_docs     = recent["primaryDocument"]

    results = []
    for i, form in enumerate(forms):
        if form == form_type:
            acc_nodash = accessions[i].replace("-", "")
            cik_int    = int(cik)
            results.append({
                "accession_number": accessions[i],
                "date":             dates[i],
                "primary_document": primary_docs[i],
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_int}/{acc_nodash}/{primary_docs[i]}"
                ),
            })
            if len(results) >= count:
                break

    if not results:
        raise ValueError(f"No {form_type} filings found for CIK {cik}.")

    return results


# ── 3. Accession number → plain text ────────────────────────

def _strip_html(html: str) -> str:
    """
    Extract readable text from an HTML or iXBRL filing document.

    Strategy:
    1. Jump to <body> — skips XBRL schema declarations in the <head>
    2. Drop <script>, <style>, and hidden elements entirely
    3. Strip remaining tags, decode entities, collapse whitespace
    """
    # 1. Skip everything before <body> (XBRL namespace junk lives in <head>)
    body_start = html.lower().find("<body")
    if body_start != -1:
        html = html[body_start:]

    # 2. Drop non-visible blocks wholesale
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>",   " ", html, flags=re.DOTALL | re.IGNORECASE)
    # hidden elements (display:none) — usually XBRL tagging wrappers
    html = re.sub(
        r'<[^>]+style="[^"]*display\s*:\s*none[^"]*"[^>]*>.*?</\w+>',
        " ", html, flags=re.DOTALL | re.IGNORECASE,
    )

    # 3. Strip all remaining tags
    text = re.sub(r"<[^>]+>", " ", html)

    # 4. Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;",  "&")
    text = text.replace("&lt;",   "<")
    text = text.replace("&gt;",   ">")
    text = re.sub(r"&#\d+;",    " ", text)   # numeric entities
    text = re.sub(r"&[a-z]+;",  " ", text)   # any remaining named entities

    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_filing_text(accession_number: str, cik: str, primary_doc: str) -> str:
    """
    Download and return the plain text of a filing's primary document.

    Args:
        accession_number: EDGAR accession number, e.g. "0000320193-24-000085"
        cik: 10-digit CIK string.
        primary_doc: Primary document filename, from get_filings().

    Returns:
        Plain text content of the filing (HTML tags stripped).
    """
    acc_nodash = accession_number.replace("-", "")
    cik_int    = int(cik)

    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_nodash}/{primary_doc}"
    )
    doc_resp = requests.get(
        doc_url,
        headers={**HEADERS, "Accept": "text/html,text/plain"},
        timeout=30,
    )
    doc_resp.raise_for_status()

    raw = doc_resp.text
    return _strip_html(raw) if "<" in raw else raw
