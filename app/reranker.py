"""
app/reranker.py
───────────────
Cross-encoder reranking: the relevance gate of MedRAG.

Hybrid retrieval (dense + BM25) only ranks chunks against each other; it cannot
say whether ANY chunk actually answers the question. A cross-encoder reads the
question and the chunk together and returns a relevance score in [0, 1], which
MedRAG uses as its relevance gate (see rag_pipeline.py).

Two backends, chosen from the model name in RERANKER_MODEL:
  - "org/name" (contains "/")  -> Hugging Face model via sentence-transformers CrossEncoder
                                  (default: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1,
                                  trained on mMARCO = MS MARCO translated into 14 languages,
                                  French and Arabic included)
  - short FlashRank name        -> FlashRank ONNX model (e.g. ms-marco-MultiBERT-L-12)

Why not FlashRank MultiBERT by default: on our evaluation corpus its score did not
depend on relevance (an off-topic French question scored 0.999 against medical
chunks) and changed with the other passages in the batch (0.0033 vs 0.6675 for the
same pair). See scripts/diagnose_reranker.py and the README "Evaluation" section.
"""

import logging
from functools import lru_cache

from langchain_core.documents import Document

from app import config

log = logging.getLogger(__name__)


def _is_hf_model(name: str) -> bool:
    """Hugging Face ids look like 'org/name'; FlashRank ids have no slash."""
    return "/" in name


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load the Hugging Face cross-encoder once (downloaded to the HF cache on first use)."""
    import torch
    from sentence_transformers import CrossEncoder  # lazy: tests run without the real model

    log.info("Loading cross-encoder %s", config.RERANKER_MODEL)
    # Sigmoid set explicitly: scores are always probabilities in [0, 1], whatever the
    # model's saved config says, so thresholds keep the same meaning across models.
    return CrossEncoder(config.RERANKER_MODEL, max_length=512, activation_fn=torch.nn.Sigmoid())


@lru_cache(maxsize=1)
def _get_ranker():
    """Load the FlashRank model once (downloaded to data/models on first use)."""
    from flashrank import Ranker  # lazy: tests run without it

    log.info("Loading FlashRank reranker %s from %s", config.RERANKER_MODEL, config.MODELS_DIR)
    return Ranker(model_name=config.RERANKER_MODEL, cache_dir=config.MODELS_DIR, max_length=512)


def rerank(query: str, docs: list[Document]) -> list[tuple[Document, float]]:
    """
    Score every candidate chunk against the query, best first.

    Returns (document, score) pairs. If reranking is disabled the fused order
    is kept and every score is None, so the gate is skipped (logged loudly).
    """
    if not docs:
        return []
    if not config.RERANK_ENABLED:
        log.warning("Reranking disabled: no relevance gate is applied")
        return [(d, None) for d in docs]

    if _is_hf_model(config.RERANKER_MODEL):
        # One (question, chunk) pair per document; predict() returns one score per pair.
        scores = _get_cross_encoder().predict([(query, d.page_content) for d in docs], show_progress_bar=False)
        scored = [(d, float(s)) for d, s in zip(docs, scores)]
    else:
        from flashrank import RerankRequest

        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        results = _get_ranker().rerank(RerankRequest(query=query, passages=passages))
        scored = [(docs[r["id"]], float(r["score"])) for r in results]  # map back to Documents

    return sorted(scored, key=lambda x: x[1], reverse=True)
