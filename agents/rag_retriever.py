"""
Agent 2 — RAG Retriever (Hybrid Search)

Combines semantic search (Chroma) with keyword search (BM25)
using Reciprocal Rank Fusion to merge results.

Why hybrid?
- Semantic finds conceptually relevant chunks ("revenue declined")
- BM25 finds exact matches ("85,269" or "iPhone")
- Together they catch what either alone would miss

Option B relevance filter still applies:
if the best semantic distance is above threshold → OUT_OF_SCOPE.
We check semantic first because BM25 always returns something,
even for garbage queries, so it can't be the gatekeeper.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL
from utils.bm25_index import bm25_search

# Distance threshold for Option B relevance filter (L2, semantic only)
RELEVANCE_THRESHOLD = 1.2

# RRF constant — controls how much rank position matters vs raw score
# 60 is the standard value from the original RRF paper
RRF_K = 60


def get_collection():
    """Return the Chroma collection (must already exist — run ingest.py first)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
    )


def _reciprocal_rank_fusion(
    semantic_chunks: list[dict],
    bm25_chunks:     list[dict],
    n_results:       int = 5,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score for a chunk = 1/(k + rank_in_semantic) + 1/(k + rank_in_bm25)
    Chunks appearing in both lists get a boost.
    Chunks only in one list still contribute via that list's rank.

    Returns top n_results chunks by combined RRF score.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    # Score semantic results — use first 200 chars as dedup key
    for rank, chunk in enumerate(semantic_chunks, 1):
        key = chunk["text"][:200]
        scores[key]    = scores.get(key, 0) + 1 / (RRF_K + rank)
        chunk_map[key] = chunk

    # Score BM25 results
    for rank, chunk in enumerate(bm25_chunks, 1):
        key = chunk["text"][:200]
        scores[key]    = scores.get(key, 0) + 1 / (RRF_K + rank)
        chunk_map[key] = chunk

    # Sort by combined RRF score descending
    top_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:n_results]

    return [
        {**chunk_map[k], "rrf_score": scores[k]}
        for k in top_keys
    ]


def retrieve(claim: str, n_results: int = 12) -> dict:
    """
    Hybrid retrieval: semantic + BM25 merged via RRF.

    Returns either:
      {"chunks": [...], "best_distance": float}   ← proceed to judge
      {"verdict": "OUT_OF_SCOPE", ...}             ← skip judge entirely

    Each chunk dict has: text, source, rrf_score.
    """
    # ── Semantic search via Chroma ───────────────────────────
    collection = get_collection()
    results    = collection.query(
        query_texts=[claim],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    best_distance = min(distances)

    # ── Option B: relevance filter ───────────────────────────
    if best_distance > RELEVANCE_THRESHOLD:
        return {
            "verdict":       "OUT_OF_SCOPE",
            "chunks":        [],
            "best_distance": best_distance,
        }

    semantic_chunks = [
        {
            "text":     doc,
            "source":   meta.get("source", "unknown"),
            "distance": dist,
            "rank":     rank + 1,
        }
        for rank, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances)
        )
    ]

    # ── BM25 keyword search ──────────────────────────────────
    try:
        keyword_chunks = bm25_search(claim, n_results=n_results)
    except FileNotFoundError:
        # BM25 index not built yet — fall back to semantic only
        keyword_chunks = []

    # ── Merge with Reciprocal Rank Fusion ────────────────────
    if keyword_chunks:
        merged = _reciprocal_rank_fusion(semantic_chunks, keyword_chunks, n_results)
    else:
        merged = semantic_chunks

    return {
        "chunks":        merged,
        "best_distance": best_distance,
    }
