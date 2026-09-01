"""
Central config — loads from .env and exposes typed constants.
Every other module imports from here; nothing else touches os.environ.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from the project root, regardless of where the script is run from
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── OpenRouter ───────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

# ── Models ───────────────────────────────────────────────────
EXTRACTOR_MODEL: str = os.getenv("EXTRACTOR_MODEL", "thudm/glm-4-9b")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-4-5")

# ── Embeddings ───────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chroma ───────────────────────────────────────────────────
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./vectorstore")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "sec_filings")

# ── SEC EDGAR ────────────────────────────────────────────────
EDGAR_USER_AGENT: str = os.environ["EDGAR_USER_AGENT"]
EDGAR_BASE_URL: str = "https://data.sec.gov"
