"""End-to-end pipeline tests through the API (offline fakes from conftest)."""

from langchain_core.documents import Document

import app.rag_pipeline as rp


def _ingest(client, pdf, name="patient_test"):
    r = client.post("/ingest", files={"file": (f"{name}.pdf", pdf, "application/pdf")},
                    data={"collection_name": name})
    assert r.status_code == 200, r.text
    return r.json()


def test_ingest_counts_chunks_and_quarantines_injection(client, patient_pdf):
    data = _ingest(client, patient_pdf)
    assert data["pages"] == 3 and data["chunks_indexed"] >= 3
    assert data["suspicious_chunks"] == 1                     # page 3 attack found at ingestion
    assert "patient_test" in client.get("/collections").json()["collections"]


def test_in_scope_question_is_answered_with_sources(client, patient_pdf):
    _ingest(client, patient_pdf)
    r = client.post("/ask", json={"question": "Quelle est la dose de metformine ?",
                                  "collection_name": "patient_test"}).json()
    assert not r["refused"]
    assert "Metformine 1000 mg" in r["answer"] and "[Page 2]" in r["answer"]
    assert any(s["page"] == 2 for s in r["sources"])
    assert r["retrieval"]["blocked_chunks"] == 1
    assert all("Ignore previous" not in s["content"] for s in r["sources"])  # attack never reaches the LLM


def test_out_of_scope_question_is_refused_without_calling_the_llm(client, patient_pdf, monkeypatch):
    _ingest(client, patient_pdf)
    calls = []
    monkeypatch.setattr(rp, "get_llm_chain", lambda: calls.append(1))  # would crash if used
    r = client.post("/ask", json={"question": "Qui a gagné la coupe du monde 2022 ?",
                                  "collection_name": "patient_test"}).json()
    assert r["refused"] and r["answer"].startswith("Information non trouvée")
    assert r["sources"] == [] and calls == []


def test_grounding_flags_unsupported_sentences(monkeypatch):
    from app.grounding import check_grounding
    result = check_grounding("Metformine 1000 mg [Page 2]. Dose INVENTÉE de 5 g.", "<document>...</document>")
    assert result.checked and result.grounded is False and len(result.unsupported) == 1


def test_compare_two_patients(client, patient_pdf):
    _ingest(client, patient_pdf, "patient_a")
    _ingest(client, patient_pdf, "patient_b")
    r = client.post("/compare", json={"question": "Quelle est la dose de metformine ?",
                                      "collection_names": ["patient_a", "patient_b"]}).json()
    assert not r["refused"] and set(r["per_patient"]) == {"patient_a", "patient_b"}


def test_api_rejects_bad_requests(client):
    assert client.post("/ingest", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}).status_code == 400
    assert client.post("/ask", json={"question": "dose ?"}).status_code == 400
    assert client.post("/compare", json={"question": "dose ?", "collection_names": ["a"]}).status_code == 400


def test_rrf_rewards_chunks_found_by_both_retrievers():
    a, b, c = (Document(page_content=t) for t in ("A", "B", "C"))
    fused = rp._rrf_fuse([([a, b], 0.7), ([c, b], 0.3)], limit=3)
    assert fused[0].page_content == "B"  # B is in both lists, so it wins


def test_dense_search_falls_back_when_chroma_index_is_unreadable(client, patient_pdf, monkeypatch):
    """ChromaDB 1.5.x bug: 'Nothing found on disk'. The question must still be answered."""
    import chromadb.errors

    import app.vector_store as vs

    _ingest(client, patient_pdf)

    def broken(self, *args, **kwargs):  # every Chroma vector query fails like on Windows
        raise chromadb.errors.InternalError("Error creating hnsw segment reader: Nothing found on disk")

    monkeypatch.setattr(vs.Chroma, "similarity_search", broken)
    monkeypatch.setattr(vs.time, "sleep", lambda s: None)  # skip the retry waits
    r = client.post("/ask", json={"question": "Quelle est la dose de metformine ?",
                                  "collection_name": "patient_test"}).json()
    assert not r["refused"] and "Metformine 1000 mg" in r["answer"]
    assert all("Ignore previous" not in s["content"] for s in r["sources"])  # quarantine still applies


def test_dense_search_does_not_hide_other_chroma_errors(monkeypatch):
    import chromadb.errors
    import pytest

    import app.vector_store as vs

    class Store:
        def similarity_search(self, query, k):
            raise chromadb.errors.InternalError("disk full")

    with pytest.raises(chromadb.errors.InternalError):
        vs.dense_search(Store(), "any_collection", [], "question", 3)


def test_gate_keeps_chunk_when_the_rewrite_is_bad(client, patient_pdf, monkeypatch):
    """A bad LLM rewrite must not hide a chunk that the user's own question finds."""
    _ingest(client, patient_pdf)
    monkeypatch.setattr(rp, "rewrite_query", lambda q, strict=False: "zzz qqq www")  # useless rewrite
    r = client.post("/ask", json={"question": "Quelle est la dose de metformine ?",
                                  "collection_name": "patient_test"}).json()
    assert not r["refused"] and "Metformine 1000 mg" in r["answer"]
