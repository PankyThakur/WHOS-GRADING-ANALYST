"""
Chunker — paragraph-aware semantic chunking for SEC filing text.

Why paragraph-aware instead of word-count sliding window?
- SEC filings use double newlines to separate financial paragraphs, tables,
  and discussion sections. These are natural semantic units.
- A sliding window at word 512 can split "iPhone revenue was $85,269 in Q3
  vs $69,138 last year" across two chunks, leaving each half useless.
- Paragraph boundaries preserve the claim intact in one chunk.

Strategy:
  1. Split on blank lines (\\n\\n) → raw paragraphs
  2. Accumulate paragraphs into a chunk up to MAX_CHUNK_WORDS
  3. Paragraphs larger than MAX_CHUNK_WORDS are split at sentence boundaries
  4. Last OVERLAP_SENTENCES from the previous chunk carry over for continuity
"""

import hashlib
import re


MAX_CHUNK_WORDS    = 500   # flush when a chunk exceeds this
MIN_CHUNK_WORDS    = 80    # don't emit tiny orphan chunks
OVERLAP_SENTENCES  = 2     # sentences carried from prev chunk to next


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    """
    Sentence splitter that handles common abbreviations.
    Splits on '. ', '! ', '? ' followed by a capital letter or end of string.
    Not perfect, but good enough for SEC prose.
    """
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text.strip())
    return [s.strip() for s in raw if s.strip()]


def _emit(sentences: list[str], source_id: str, index: int) -> dict:
    """Join a list of sentences into a chunk dict."""
    text   = " ".join(sentences)
    raw_id = f"{source_id}_{index}"
    return {
        "id":     hashlib.md5(raw_id.encode()).hexdigest(),
        "text":   text,
        "source": source_id,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source_id: str = "unknown",
    # Legacy params kept so ingest.py call-site doesn't break
    chunk_size: int = 512,
    overlap:    int = 64,
) -> list[dict]:
    """
    Split text into paragraph-aware semantic chunks.

    Args:
        text:      Raw filing text (already HTML-stripped).
        source_id: Label for the source doc, e.g. "AAPL_10Q_2026-07-31".
                   Used as metadata in Chroma.

    Returns:
        List of dicts with keys: id, text, source.
        Same format as before — nothing downstream needs to change.
    """
    # ── 1. Split into paragraphs ─────────────────────────────
    raw_paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    chunks  = []   # final output
    current = []   # sentences accumulating in the current chunk
    cur_wc  = 0
    index   = 0

    def flush():
        """Emit current buffer as a chunk, carry overlap into next."""
        nonlocal current, cur_wc, index
        if _word_count(" ".join(current)) >= MIN_CHUNK_WORDS:
            chunks.append(_emit(current, source_id, index))
            index += 1
        # Carry last OVERLAP_SENTENCES into next chunk for continuity
        current = current[-OVERLAP_SENTENCES:] if current else []
        cur_wc  = sum(_word_count(s) for s in current)

    # ── 2. Walk paragraphs ───────────────────────────────────
    for para in paragraphs:
        para_wc = _word_count(para)

        if para_wc > MAX_CHUNK_WORDS:
            # ── 2a. Big paragraph → sentence-level split ─────
            # First flush whatever we've been accumulating
            if cur_wc >= MIN_CHUNK_WORDS:
                flush()
            elif current:
                # tiny leftover — keep it, don't lose overlap
                pass

            for sent in _split_sentences(para):
                sent_wc = _word_count(sent)
                if cur_wc + sent_wc > MAX_CHUNK_WORDS and cur_wc >= MIN_CHUNK_WORDS:
                    flush()
                current.append(sent)
                cur_wc += sent_wc

        else:
            # ── 2b. Normal paragraph ─────────────────────────
            if cur_wc + para_wc > MAX_CHUNK_WORDS and cur_wc >= MIN_CHUNK_WORDS:
                flush()
            # Treat the whole paragraph as one "sentence" in our buffer
            current.append(para)
            cur_wc += para_wc

    # ── 3. Flush remainder ───────────────────────────────────
    if current and _word_count(" ".join(current)) >= MIN_CHUNK_WORDS:
        chunks.append(_emit(current, source_id, index))

    return chunks
