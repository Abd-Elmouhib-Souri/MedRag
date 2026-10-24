"""
scripts/check_models.py
───────────────────────
Smoke test for the REAL local models (pytest uses fakes):
downloads them once, then checks they load and behave sensibly.

    python -m scripts.check_models
"""

import time

from langchain_core.documents import Document

from app import config
from app.embeddings import get_embeddings_model
from app.reranker import rerank

QUESTION = "Quelle est la dose de warfarine ?"

# Three passages with a known expected order: relevant > same domain > off-topic.
PASSAGES = [
    "Traitement actuel : Warfarine 5 mg le soir, objectif INR entre 2 et 3.",  # answers the question
    "Spirométrie du 3 juin 2026 : VEMS à 78 % de la valeur théorique.",        # medical, wrong topic
    "Recette du couscous : semoule, agneau, pois chiches et légumes.",         # off-topic
]


def main() -> None:
    config.ensure_dirs()  # creates data/models where FlashRank stores its files

    # 1) Embeddings: first call downloads the model into the Hugging Face cache
    t0 = time.perf_counter()
    vector = get_embeddings_model().embed_query(QUESTION)
    print(f"embeddings OK: {config.EMBEDDING_MODEL} -> {len(vector)} dims "
          f"({time.perf_counter() - t0:.1f}s)")

    # 2) Reranker: first call downloads the FlashRank model into data/models
    t0 = time.perf_counter()
    scored = rerank(QUESTION, [Document(page_content=p) for p in PASSAGES])  # sorted best first
    print(f"reranker OK: {config.RERANKER_MODEL} ({time.perf_counter() - t0:.1f}s)")
    for doc, score in scored:
        print(f"  {score:.4f}  {doc.page_content[:60]}")


if __name__ == "__main__":
    main()
