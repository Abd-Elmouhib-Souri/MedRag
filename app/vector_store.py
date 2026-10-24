"""
app/vector_store.py
───────────────────
ChromaDB collections: one isolated collection per patient document.

Isolation matters for privacy: a question about patient A can never retrieve
chunks of patient B, because each query targets one collection only.

ChromaDB 1.5.x workaround (seen on Windows during evaluation): the first query after
a write can fail with "Error creating hnsw segment reader: Nothing found on disk",
because the vector index is not flushed to disk yet; deleting a collection can
trigger the same failure on other collections. So MedRAG (1) re-ingests by clearing
the records of an existing collection instead of deleting the collection, and
(2) if the index stays unreadable, answers with an exact in-memory search over the
same chunks (dense_search), so the bug can slow a question down but never break it.
"""

import logging
import re
import time

import chromadb
import chromadb.errors
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app import config
from app.embeddings import get_embeddings_model
from app.guardrails import screen_text

log = logging.getLogger(__name__)

DEFAULT_COLLECTION = "medrag_default"

# BM25 needs every chunk of a collection in memory; we cache them per
# collection instead of reloading them from ChromaDB on every question.
_DOCS_CACHE: dict[str, list[Document]] = {}
_VEC_CACHE: dict = {}  # collection -> numpy matrix of chunk vectors (fallback search only)


def _sanitize_collection_name(name: str) -> str:
    """
    ChromaDB naming rules: 3-63 characters, starts and ends with an
    alphanumeric character, only letters, digits, underscores and dashes.
    """
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    name = name.strip("_-")[:63]
    if len(name) < 3:
        name = name + "_doc"
    return name


def _client() -> chromadb.ClientAPI:
    """Persistent ChromaDB client on disk."""
    config.ensure_dirs()
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def create_vector_store(chunks: list, collection_name: str = DEFAULT_COLLECTION) -> tuple[Chroma, int]:
    """
    Embed the chunks and store them in their own collection.

    Each chunk is screened for prompt injection; suspicious chunks are stored
    with suspicious=True so retrieval can exclude them (and the API can report
    them). Re-indexing the same name replaces the old collection.

    Returns the vector store and the number of suspicious chunks.
    """
    safe_name = _sanitize_collection_name(collection_name)
    _clear_records(safe_name)  # re-ingest = clean replace, no duplicates (see module note)

    documents, suspicious = [], 0
    for chunk in chunks:
        screen = screen_text(chunk["text"])
        suspicious += screen.suspicious
        documents.append(Document(
            page_content=chunk["text"],
            metadata={
                "page":       chunk["page"],
                "has_alert":  chunk["has_alert"],
                "has_dosage": chunk["has_dosage"],
                "suspicious": screen.suspicious,
            },
        ))
        if screen.suspicious:
            log.warning("Prompt-injection pattern on page %s: %r", chunk["page"], screen.match)

    log.info("Indexing %d chunks into '%s'", len(documents), safe_name)
    store = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings_model(),
        client=_client(),
        collection_name=safe_name,
    )
    _DOCS_CACHE[safe_name] = documents
    return store, suspicious


def _clear_records(safe_name: str) -> None:
    """Remove every record of an existing collection, keeping the collection itself."""
    _DOCS_CACHE.pop(safe_name, None)
    _VEC_CACHE.pop(safe_name, None)
    if safe_name not in {col.name for col in _client().list_collections()}:
        return  # new collection: nothing to clear
    collection = _client().get_collection(safe_name)
    ids = collection.get(include=[])["ids"]  # ids only, no embeddings or text
    if ids:
        collection.delete(ids=ids)
        log.info("Cleared %d old chunks from '%s' before re-indexing", len(ids), safe_name)


def _exact_search(collection_name: str, docs: list[Document], query: str, k: int) -> list[Document]:
    """
    Exact nearest-neighbour search in memory (L2 distance, Chroma's default metric).

    Per-patient collections are small (tens to a few thousand chunks), so an exact
    scan is fast and needs no HNSW index file. Chunk vectors are cached per collection.
    """
    import numpy as np

    model = get_embeddings_model()
    key = _sanitize_collection_name(collection_name)
    if key not in _VEC_CACHE or len(_VEC_CACHE[key]) != len(docs):
        _VEC_CACHE[key] = np.asarray(model.embed_documents([d.page_content for d in docs]), dtype=np.float32)
    q = np.asarray(model.embed_query(query), dtype=np.float32)
    distances = np.linalg.norm(_VEC_CACHE[key] - q, axis=1)  # one distance per chunk
    return [docs[i] for i in np.argsort(distances)[:k]]       # closest first


def dense_search(store: Chroma, collection_name: str, docs: list[Document], query: str, k: int,
                 retries: int = 2) -> list[Document]:
    """
    Dense search through ChromaDB, with a safety net for a ChromaDB 1.5.x bug.

    ChromaDB sometimes cannot open its vector index ("Error creating hnsw segment
    reader: Nothing found on disk"; seen on Windows, see module note). We retry
    briefly, then fall back to an exact in-memory search over the same chunks
    (`docs`, read from ChromaDB's document store, which is not affected), so a
    question never fails because of this bug. Any OTHER error is raised.
    """
    for attempt in range(retries + 1):
        try:
            return store.similarity_search(query, k=k)
        except chromadb.errors.InternalError as exc:
            if "Nothing found on disk" not in str(exc):
                raise  # a different problem: never hide it
            if attempt < retries:
                time.sleep(0.25 * 2 ** attempt)  # 0.25 s, 0.5 s
    log.warning("ChromaDB vector index unreadable for '%s'; using exact in-memory search", collection_name)
    return _exact_search(collection_name, docs, query, k)


def get_vector_store(collection_name: str = DEFAULT_COLLECTION) -> Chroma:
    """Open an existing collection without recomputing embeddings."""
    return Chroma(
        client=_client(),
        embedding_function=get_embeddings_model(),
        collection_name=_sanitize_collection_name(collection_name),
    )


def get_all_documents(collection_name: str) -> list[Document]:
    """Every chunk of a collection (for BM25), cached after the first load."""
    safe_name = _sanitize_collection_name(collection_name)
    if safe_name not in _DOCS_CACHE:
        raw = get_vector_store(safe_name).get()
        _DOCS_CACHE[safe_name] = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw.get("documents") or [], raw.get("metadatas") or [])
        ]
    return _DOCS_CACHE[safe_name]


def list_collections() -> list:
    """Names of every indexed document (used by /collections and the sidebar)."""
    try:
        return sorted(col.name for col in _client().list_collections())
    except Exception as exc:
        log.warning("Cannot list ChromaDB collections: %s", exc)
        return []


def delete_collection(collection_name: str, quiet: bool = False) -> bool:
    """Delete a collection and its cached chunks; returns False if it did not exist."""
    safe_name = _sanitize_collection_name(collection_name)
    _DOCS_CACHE.pop(safe_name, None)
    _VEC_CACHE.pop(safe_name, None)
    try:
        _client().delete_collection(safe_name)
        log.info("Collection '%s' deleted", safe_name)
        return True
    except Exception as exc:
        if not quiet:
            log.warning("Cannot delete collection '%s': %s", collection_name, exc)
        return False
