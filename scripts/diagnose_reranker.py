"""
scripts/diagnose_reranker.py
────────────────────────────
Is the reranker usable as a relevance gate? Controlled experiments on the
evaluation index, one variable each, for whatever RERANKER_MODEL is set:

  A. batch vs one-by-one   -> the score of a pair must not depend on the other passages
  B. full chunk vs short   -> does passage LENGTH drive the score?
  C. on-topic vs off-topic -> the same chunks against a question they answer and one they don't

Run after `python -m scripts.evaluate --calibrate` (which builds eval/chroma_db):
    python -m scripts.diagnose_reranker
"""

from pathlib import Path

from langchain_core.documents import Document

from app import config

config.CHROMA_DIR = str(Path("eval/chroma_db").resolve())  # same index as the calibration run

from app.reranker import rerank                 # noqa: E402  (import after the path is set)
from app.vector_store import get_all_documents  # noqa: E402

OFF_TOPIC = "Donne-moi une recette de couscous."            # every score SHOULD be low
ON_TOPIC = "Quel est le traitement de fond de l'asthme ?"   # chunk 2 answers it
COLLECTION = "dossier_gharbi_salma"


def score(query: str, passages: list[str]) -> list[float]:
    """Score passages in ONE call, returned in the input order."""
    docs = [Document(page_content=p, metadata={"i": i}) for i, p in enumerate(passages)]
    by_i = {d.metadata["i"]: s for d, s in rerank(query, docs)}
    return [by_i[i] for i in range(len(passages))]


def main() -> None:
    chunks = [d.page_content for d in get_all_documents(COLLECTION)]
    print(f"Reranker: {config.RERANKER_MODEL}  |  collection: {COLLECTION}  |  {len(chunks)} chunks\n")

    batch = score(OFF_TOPIC, chunks)
    single = [score(OFF_TOPIC, [c])[0] for c in chunks]
    print(f"A. batch vs one-by-one, query {OFF_TOPIC!r} (same numbers = batch-independent)")
    for c, b, s in zip(chunks, batch, single):
        print(f"   batch {b:.4f} | single {s:.4f} | {len(c):4d} chars | {c[:45]!r}")

    print("\nB. full chunk vs first 80 chars (big drop = length/filler drives the score)")
    for c, full, cut in zip(chunks, single, score(OFF_TOPIC, [c[:80] for c in chunks])):
        print(f"   full {full:.4f} -> short {cut:.4f} | {c[:45]!r}")

    print(f"\nC. on-topic {ON_TOPIC!r} vs off-topic (a good gate separates them)")
    for c, on, off in zip(chunks, score(ON_TOPIC, chunks), single):
        print(f"   on-topic {on:.4f} | off-topic {off:.4f} | {c[:45]!r}")


if __name__ == "__main__":
    main()
