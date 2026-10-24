"""
app/config.py
─────────────
Single place for every setting of MedRAG.

Every value can be overridden with an environment variable (see .env.example),
so the same code runs on a laptop, in Docker or in the evaluation script.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()  # read .env once, before any setting is used


def _float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to the default."""
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to the default."""
    return int(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    """Read a boolean flag: '1', 'true', 'yes' and 'on' mean True."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ── Storage ───────────────────────────────────────────────────────────────────
# Absolute paths, anchored to the project folder: the API, the UI and the scripts
# then use the same data whatever folder they are started from, and ChromaDB never
# gets a relative path (a relative path was part of a Windows bug hunt, see vector_store.py).
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.abspath(os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data")))
CHROMA_DIR    = os.getenv("CHROMA_PERSIST_DIR", os.path.join(DATA_DIR, "chroma_db"))
SUMMARIES_DIR = os.getenv("SUMMARIES_DIR", os.path.join(DATA_DIR, "summaries"))
UPLOADS_DIR   = os.getenv("UPLOADS_DIR", os.path.join(DATA_DIR, "uploads"))
MODELS_DIR    = os.getenv("MODELS_DIR", os.path.join(DATA_DIR, "models"))

# ── LLMs (Groq) ───────────────────────────────────────────────────────────────
# MODEL_FAST rewrites queries and judges groundedness; MODEL_STRONG answers.
# Groq retires models regularly (llama3-8b-8192 in v2; llama-3.1-8b-instant and
# llama-3.3-70b-versatile on 16 Aug 2026). Defaults follow Groq's recommended
# replacements; check them with: python -m scripts.check_llm
MODEL_FAST   = os.getenv("MODEL_FAST", "openai/gpt-oss-20b")
MODEL_STRONG = os.getenv("MODEL_STRONG", "openai/gpt-oss-120b")

# Language of the patient records. Queries are rewritten INTO this language for
# retrieval (query translation), so an English or Arabic question still matches a
# French record; the answer itself is given in the user's language.
DOCUMENT_LANGUAGE = os.getenv("DOCUMENT_LANGUAGE", "français")
LLM_TIMEOUT  = _int("LLM_TIMEOUT", 60)        # seconds per Groq call
LLM_RETRIES  = _int("LLM_RETRIES", 3)         # automatic retries on 429 / 5xx

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = _int("CHUNK_SIZE", 600)       # characters per chunk
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 80)

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_CANDIDATES = _int("TOP_K_CANDIDATES", 12)   # chunks fetched by each retriever
RERANK_TOP_N     = _int("RERANK_TOP_N", 8)        # fused chunks sent to the cross-encoder (CPU cost cap)
TOP_K_CONTEXT    = _int("TOP_K_CONTEXT", 5)       # chunks finally given to the LLM
WEIGHT_DENSE     = _float("WEIGHT_DENSE", 0.7)    # ChromaDB share in rank fusion
WEIGHT_BM25      = _float("WEIGHT_BM25", 0.3)     # BM25 share in rank fusion
RRF_K            = _int("RRF_K", 60)              # standard Reciprocal Rank Fusion constant
MULTI_QUERY      = _bool("MULTI_QUERY", True)     # search with original AND rewritten query

# ── Relevance gate (cross-encoder reranking) ──────────────────────────────────
# Measured on the 32 dev questions, raw queries, same pipeline (see README, Evaluation):
#   FlashRank ms-marco-MultiBERT-L-12          : balanced accuracy 50% (chance), batch-dependent scores
#   cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 : ROC-AUC 0.838, leave-one-out 73%, ~140 ms/question (CPU)
#   BAAI/bge-reranker-v2-m3                    : ROC-AUC 0.967, leave-one-out 88%, ~1.3 s/question (CPU)
# bge is the default (quality); set RERANKER_MODEL to the mmarco model for a ~10x faster gate.
RERANK_ENABLED   = _bool("RERANK_ENABLED", True)
RERANKER_MODEL   = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# Chunks scoring below this are dropped; if none is left, MedRAG refuses WITHOUT calling the LLM.
# Recall-oriented on purpose: the gate removes obvious off-topic questions and keeps
# >= 95% of answerable ones; the LLM's refusal rule handles harder cases.
# Re-calibrate after any model change: python -m scripts.evaluate --calibrate --rewrite
RELEVANCE_THRESHOLD = _float("RELEVANCE_THRESHOLD", 0.0034)  # bge, calibrated on dev (rewritten queries)

# ── Hallucination mitigation ──────────────────────────────────────────────────
GROUNDING_CHECK = _bool("GROUNDING_CHECK", True)  # verify the answer against its sources

# ── Upload limits (retrieval security) ────────────────────────────────────────
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 20)
MAX_PDF_PAGES = _int("MAX_PDF_PAGES", 500)

# ── Startup ───────────────────────────────────────────────────────────────────
WARMUP_MODELS = _bool("WARMUP_MODELS", True)  # load embeddings + reranker when the API starts

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)


def ensure_dirs() -> None:
    """Create every data folder so a fresh clone works without manual mkdir."""
    for path in (DATA_DIR, CHROMA_DIR, SUMMARIES_DIR, UPLOADS_DIR, MODELS_DIR):
        os.makedirs(path, exist_ok=True)
