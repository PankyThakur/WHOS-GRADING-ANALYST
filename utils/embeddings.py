"""
Embeddings — pushes chunked text into Chroma.

Chroma handles the actual embedding via its built-in
SentenceTransformerEmbeddingFunction, so we never manually
call an embedding model here.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL


def get_collection():
    """
    Return the Chroma collection, creating it if it doesn't exist.
    Embedding model is baked into the collection at creation time.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
    )


def ingest(chunks: list[dict]) -> int:
    """
    Upsert chunks into the Chroma collection.

    Uses upsert (not add) so re-running ingest on the same filing
    is idempotent — no duplicate chunks.

    Args:
        chunks: List of dicts with keys: id, text, source.

    Returns:
        Number of chunks upserted.
    """
    if not chunks:
        return 0

    collection = get_collection()

    collection.upsert(
        ids        = [c["id"]     for c in chunks],
        documents  = [c["text"]   for c in chunks],
        metadatas  = [{"source": c["source"]} for c in chunks],
    )

    return len(chunks)
