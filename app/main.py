"""
app/main.py
───────────
MedRAG REST API (FastAPI). Interactive docs: http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from app import config
from app.document_loader import load_pdf, split_text_with_metadata
from app.guardrails import UploadError, safe_collection_name, screen_text, validate_and_store_pdf
from app.patient_summary import (delete_patient_summary, generate_patient_summary,
                                 load_all_summaries, load_patient_summary)
from app.rag_pipeline import query_compare, query_medrag
from app.report_generator import generate_pdf_report
from app.vector_store import create_vector_store, delete_collection, list_collections

log = logging.getLogger("medrag.api")
config.ensure_dirs()  # a fresh clone works without creating folders by hand

def _warm_up_models() -> None:
    """
    Load the embedding model and the reranker once, at startup.

    Without this, the FIRST question pays the loading time (bge-reranker-v2-m3 is
    ~2.3 GB: about 30-60 s on a laptop) and the UI request times out.
    """
    import time

    from langchain_core.documents import Document

    from app.embeddings import get_embeddings_model
    from app.reranker import rerank

    t0 = time.perf_counter()
    try:
        get_embeddings_model().embed_query("warm-up")
        if config.RERANK_ENABLED:
            rerank("warm-up", [Document(page_content="warm-up")])
        log.info("Models ready in %.1f s (embeddings + %s)", time.perf_counter() - t0, config.RERANKER_MODEL)
    except Exception:  # never block the API: the first question will load them instead
        log.exception("Model warm-up failed; models will load on the first question")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Runs once when uvicorn starts the API (before it accepts requests)."""
    if config.WARMUP_MODELS:
        _warm_up_models()
    yield


app = FastAPI(
    title="MedRAG API",
    version="3.0",
    description="Trustworthy clinical RAG: hybrid retrieval, relevance gate, grounding check, clinical alerts.",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """Log the full traceback server-side; return a short JSON error to the client."""
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"[:300]})


# ── Request models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    collection_name: Optional[str] = None


class CompareRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    collection_names: List[str]


class ReportRequest(BaseModel):
    conversation: List[dict]
    collection_name: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    """Health check + the active safety settings (useful when reading logs or results)."""
    return {
        "message": "MedRAG API — Clinical Intelligence",
        "version": app.version,
        "settings": {
            "llm": config.MODEL_STRONG, "llm_fast": config.MODEL_FAST,
            "reranker": config.RERANKER_MODEL if config.RERANK_ENABLED else None,
            "relevance_threshold": config.RELEVANCE_THRESHOLD,
            "grounding_check": config.GROUNDING_CHECK,
            "fusion": f"weighted RRF dense {config.WEIGHT_DENSE} / bm25 {config.WEIGHT_BM25}",
        },
    }


@app.get("/collections")
async def get_collections():
    """Every indexed document."""
    names = list_collections()
    return {"collections": names, "count": len(names)}


@app.get("/summaries")
async def get_all_summaries():
    """Every auto-generated patient card."""
    summaries = load_all_summaries()
    return {"summaries": summaries, "count": len(summaries)}


@app.get("/summaries/{collection_name}")
async def get_summary(collection_name: str):
    """One patient card."""
    summary = load_patient_summary(collection_name)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Patient card '{collection_name}' not found. Re-index the PDF.")
    return summary


@app.post("/ingest")
def ingest_document(file: UploadFile = File(...), collection_name: Optional[str] = Form(None)):
    """
    Upload a PDF (multipart/form-data), index it in its own collection and
    generate the patient card.

    Security: the server never opens a client-supplied path. The upload is
    checked (real PDF, size, pages, not encrypted) and stored under a random name.
    """
    try:
        stored = validate_and_store_pdf(file.file.read(), file.filename)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    col_name = safe_collection_name(collection_name or file.filename)
    pages = load_pdf(str(stored))
    chunks = split_text_with_metadata(pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text in this PDF (scanned image?).")

    _, suspicious = create_vector_store(chunks, collection_name=col_name)
    # The patient card is generated from clean chunks only (no injected text reaches the LLM)
    clean_chunks = [c for c in chunks if not screen_text(c["text"]).suspicious]
    summary = generate_patient_summary(collection_name=col_name, chunks=clean_chunks)
    return {
        "status": "ok",
        "collection_name": col_name,
        "pages": len(pages),
        "chunks_indexed": len(chunks),
        "suspicious_chunks": suspicious,  # prompt-injection attempts found and quarantined
        "summary": summary,
    }


@app.post("/ask")
def ask_medrag(request: QueryRequest):
    """Question on one patient record: answer, sources, alerts, grounding, retrieval details."""
    if not request.collection_name:
        raise HTTPException(status_code=400, detail="collection_name is required.")
    result = query_medrag(question=request.question, collection_name=request.collection_name)
    return {"question": request.question, **result}


@app.post("/compare")
def compare_patients(request: CompareRequest):
    """Compare 2+ patient records on the same question."""
    if len(request.collection_names) < 2:
        raise HTTPException(status_code=400, detail="At least 2 collections are required.")
    result = query_compare(question=request.question, collection_names=request.collection_names)
    return {"question": request.question, **result}


@app.post("/report")
async def export_report(req: ReportRequest):
    """PDF report of the conversation."""
    if not req.conversation:
        raise HTTPException(status_code=400, detail="Nothing to export.")
    pdf_bytes = generate_pdf_report(conversation=req.conversation, collection_name=req.collection_name)
    filename = f"rapport_medrag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.delete("/collections/{collection_name}")
async def remove_collection(collection_name: str):
    """Delete a document and its patient card."""
    success = delete_collection(collection_name)
    delete_patient_summary(collection_name)
    if success:
        return {"status": "ok", "deleted": collection_name}
    raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
