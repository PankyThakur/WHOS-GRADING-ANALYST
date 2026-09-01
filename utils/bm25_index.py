"""
BM25 Index — keyword-based search over SEC filing chunks.

Used alongside Chroma's semantic search in hybrid retrieval.
BM25 excels at finding exact terms: specific dollar amounts,
ticker symbols, accounting line items like "iPhone" or "85,269".

The index is saved to disk as a pickle file so it survives
between runs without re-ingesting everything.
"""

import os
import pickle
from rank_bm25 import BM25Okapi

# Where the BM25 index lives on disk
BM25_INDEX_PATH = "./vectorstore/bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def build_index(chunks: list[dict]) -> BM25Okapi:
    """
    Build a BM25 index from a list of chunks.

    Args:
        chunks: List of dicts with keys: id, text, source.

    Returns:
        Fitted BM25Okapi index.
    """
    corpus = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def save_index(index: BM25Okapi, chunks: list[dict]):
    """
    Persist the BM25 index and the original chunks to disk.
    We save chunks alongside the index because BM25 only stores
    token frequencies — we need the original text for retrieval.
    """
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "chunks": chunks}, f)


def load_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Load the BM25 index and chunks from disk.

    Returns:
        Tuple of (BM25Okapi index, list of chunk dicts).
    """
    if not os.path.exists(BM25_INDEX_PATH):
        raise FileNotFoundError(
            "BM25 index not found. Run ingest.py first to build it."
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["chunks"]


def bm25_search(query: str, n_results: int = 5) -> list[dict]:
    """
    Search the BM25 index for chunks matching a query.

    Args:
        query:     The claim text to search for.
        n_results: Number of top results to return.

    Returns:
        List of dicts with keys: text, source, bm25_score, rank.
    """
    index, chunks = load_index()
    tokens = _tokenize(query)
    scores = index.get_scores(tokens)

    # Pair each chunk with its score and sort descending
    scored = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:n_results]

    return [
        {
            "text":       chunks[i]["text"],
            "source":     chunks[i]["source"],
            "bm25_score": float(score),
            "rank":       rank + 1,
        }
        for rank, (i, score) in enumerate(scored)
    ]
