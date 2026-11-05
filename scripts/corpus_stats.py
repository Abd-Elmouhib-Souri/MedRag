"""
scripts/corpus_stats.py
───────────────────────
Size of what is indexed: documents, pages and chunks, read straight from ChromaDB.

    python -m scripts.corpus_stats                      # the demo index (data/chroma_db)
    python -m scripts.corpus_stats --chroma eval/chroma_db

Definitions (so the numbers can be quoted exactly):
  - document = one ChromaDB collection = one uploaded PDF (MedRAG indexes one PDF per collection);
  - page     = a PDF page that produced at least one chunk (blank pages produce none);
  - chunk    = one stored passage (600 characters, 80 overlap), quarantined ones included,
               shown separately because retrieval never uses them.
No model and no Groq key needed: only ChromaDB's document store is read.
"""

import argparse
from pathlib import Path

import chromadb

from app import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chroma", default=config.CHROMA_DIR, help="ChromaDB folder (default: the demo index)")
    args = parser.parse_args()

    path = Path(args.chroma).resolve()
    if not path.exists():
        raise SystemExit(f"No ChromaDB folder at {path}: index a PDF in the app first.")
    client = chromadb.PersistentClient(path=str(path))  # read-only use: we only call get()

    total_docs = total_pages = total_chunks = total_quarantined = 0
    print(f"ChromaDB: {path}\n")
    print(f"{'collection':<34}{'pages':>7}{'chunks':>8}{'quarantined':>13}")
    for col in sorted(client.list_collections(), key=lambda c: c.name):
        metas = client.get_collection(col.name).get(include=["metadatas"])["metadatas"] or []
        if not metas:
            print(f"{col.name:<34}{'-':>7}{0:>8}{'-':>13}   (empty, not counted)")
            continue
        pages = {m.get("page") for m in metas if m}                 # distinct page numbers
        quarantined = sum(bool(m and m.get("suspicious")) for m in metas)
        print(f"{col.name:<34}{len(pages):>7}{len(metas):>8}{quarantined:>13}")
        total_docs += 1
        total_pages += len(pages)
        total_chunks += len(metas)
        total_quarantined += quarantined

    print(f"\nTOTAL: {total_docs} documents, {total_pages} pages, {total_chunks} chunks "
          f"({total_quarantined} quarantined for prompt injection)")


if __name__ == "__main__":
    main()
