"""
Tests fonctionnels MedRAG — exécution sans pytest.
Couvre : ingestion, Q&A, alertes, comparaison, fiches, export PDF, API.
"""
import json
import os
import sys
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
SAMPLE_DIR = os.path.join("data", "sample_docs")

results = []


def record(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    results.append({"test": name, "status": status, "detail": detail})
    icon = "[OK]" if ok else "[X]"
    print(f"{icon} [{status}] {name}")
    if detail:
        print(f"    → {detail[:300]}")


def test_alert_detector():
    from app.alert_detector import detect_alerts

    alert = detect_alerts(
        answer="Interaction médicamenteuse détectée avec warfarine.",
        context="Patient sous warfarine 5 mg. INR 2.4.",
    )
    ok = alert.has_alert and "warfarine" in " ".join(alert.high_risk_drugs_found).lower()
    record("Alert detector (unitaire)", ok, f"level={alert.level}, drugs={alert.high_risk_drugs_found}")


def test_document_loader():
    from app.document_loader import load_pdf, split_text_with_metadata

    pdf = os.path.join(SAMPLE_DIR, "dossier_01_benali_karim_diabete_hta.pdf")
    if not os.path.exists(pdf):
        record("Document loader", False, f"PDF manquant: {pdf}")
        return None

    pages = load_pdf(pdf)
    chunks = split_text_with_metadata(pages)
    ok = len(pages) >= 1 and len(chunks) >= 3
    record("Document loader + chunking", ok, f"{len(pages)} pages, {len(chunks)} chunks")
    return pdf


def test_embeddings():
    from app.embeddings import get_embeddings_model

    model = get_embeddings_model()
    vecs = model.embed_documents(["Metformine 1000 mg", "Diabète type 2"])
    ok = len(vecs) == 2 and len(vecs[0]) == 384
    record("Embeddings multilingues", ok, f"dim={len(vecs[0])}")


def test_ingest_local(pdf_path: str, collection_name: str):
    from app.document_loader import load_pdf, split_text_with_metadata
    from app.vector_store import create_vector_store, list_collections
    from app.patient_summary import generate_patient_summary

    pages = load_pdf(pdf_path)
    chunks = split_text_with_metadata(pages)
    create_vector_store(chunks, collection_name=collection_name)
    summary = generate_patient_summary(collection_name=collection_name, chunks=chunks)

    cols = list_collections()
    safe = collection_name.replace(".pdf", "")
    ok = any(safe in c or collection_name in c for c in cols)
    record(f"Ingestion locale — {collection_name}", ok, f"chunks={len(chunks)}, summary_keys={list(summary.keys())[:4]}")
    return summary


def test_rag_local(collection_name: str):
    from app.rag_pipeline import query_medrag

    result = query_medrag(
        question="Quel est le diagnostic principal et les allergies du patient ?",
        collection_name=collection_name,
    )
    answer = result.get("answer", "")
    ok = len(answer) > 20 and len(result.get("sources", [])) > 0
    record(f"RAG Q&A — {collection_name}", ok, answer[:200])
    return result


def test_compare_local(cols: list):
    from app.rag_pipeline import query_compare

    result = query_compare(
        question="Quel patient a le risque cardiovasculaire ou rénal le plus élevé ?",
        collection_names=cols,
    )
    answer = result.get("answer", "")
    ok = len(answer) > 50 and result.get("mode") == "comparison"
    record("RAG comparaison inter-patients", ok, answer[:200])
    return result


def test_report_pdf():
    from app.report_generator import generate_pdf_report

    conversation = [
        {"role": "user", "content": "Quel est le diagnostic ?"},
        {
            "role": "assistant",
            "content": "Diabète de type 2 avec hypertension.",
            "sources": [{"page": 1, "content": "DIAGNOSTIC PRINCIPAL : Diabète"}],
            "alert": {"has_alert": False, "alerts": [], "level": "AUCUNE"},
        },
    ]
    pdf_bytes = generate_pdf_report(conversation, collection_name="test_patient")
    ok = pdf_bytes[:4] == b"%PDF"
    record("Export rapport PDF", ok, f"taille={len(pdf_bytes)} bytes")


def wait_for_api(timeout=30):
    for _ in range(timeout):
        try:
            r = requests.get(f"{API_URL}/", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def test_api_endpoints(pdf1: str, pdf2: str, col1: str, col2: str):
    if not wait_for_api():
        record("API — disponibilité", False, f"Backend inaccessible sur {API_URL}")
        return

    record("API — disponibilité", True, API_URL)

    # GET /
    r = requests.get(f"{API_URL}/", timeout=5)
    record("API GET /", r.status_code == 200, r.json().get("message", ""))

    # GET /collections
    r = requests.get(f"{API_URL}/collections", timeout=10)
    record("API GET /collections", r.status_code == 200, f"count={r.json().get('count')}")

    # POST /ingest (2e patient via API si pas déjà fait)
    r = requests.post(
        f"{API_URL}/ingest",
        json={"file_path": pdf2, "collection_name": col2},
        timeout=180,
    )
    record("API POST /ingest", r.status_code == 200, f"chunks={r.json().get('chunks_indexed') if r.ok else r.text[:120]}")

    # GET /summaries
    r = requests.get(f"{API_URL}/summaries", timeout=10)
    record("API GET /summaries", r.status_code == 200, f"count={r.json().get('count')}")

    safe_col1 = col1.replace(".pdf", "")
    r = requests.get(f"{API_URL}/summaries/{col1}", timeout=10)
    record("API GET /summaries/{name}", r.status_code == 200, r.json().get("nom_patient", "")[:80])

    # POST /ask
    r = requests.post(
        f"{API_URL}/ask",
        json={"question": "Quels médicaments prend le patient ?", "collection_name": col1},
        timeout=90,
    )
    data = r.json() if r.ok else {}
    record(
        "API POST /ask",
        r.status_code == 200 and bool(data.get("answer")),
        (data.get("answer") or r.text)[:200],
    )

    # POST /compare
    r = requests.post(
        f"{API_URL}/compare",
        json={
            "question": "Comparez les allergies et traitements des deux patients.",
            "collection_names": [col1, col2],
        },
        timeout=180,
    )
    data = r.json() if r.ok else {}
    record(
        "API POST /compare",
        r.status_code == 200 and data.get("mode") == "comparison",
        (data.get("answer") or r.text)[:200],
    )

    # POST /report
    r = requests.post(
        f"{API_URL}/report",
        json={
            "conversation": [
                {"role": "user", "content": "Test export"},
                {"role": "assistant", "content": "Réponse test", "sources": [], "alert": None},
            ],
            "collection_name": safe_col1,
        },
        timeout=30,
    )
    record("API POST /report", r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"), f"size={len(r.content)}")


def main():
    print("\n" + "=" * 60)
    print("MEDRAG — TESTS FONCTIONNELS")
    print("=" * 60 + "\n")

    if not os.getenv("GROQ_API_KEY"):
        print("[X] GROQ_API_KEY manquante dans .env")
        sys.exit(1)

    # Créer PDFs si besoin
    if not os.path.exists(os.path.join(SAMPLE_DIR, "dossier_01_benali_karim_diabete_hta.pdf")):
        from scripts.create_test_pdfs import main as create_pdfs
        create_pdfs()

    pdf1 = os.path.join(SAMPLE_DIR, "dossier_01_benali_karim_diabete_hta.pdf")
    pdf2 = os.path.join(SAMPLE_DIR, "dossier_02_trabelsi_nadia_irc.pdf")
    col1 = "dossier_01_benali_karim_diabete_hta.pdf"
    col2 = "dossier_02_trabelsi_nadia_irc.pdf"

    # Tests unitaires / modules
    test_alert_detector()
    pdf = test_document_loader()
    test_embeddings()
    test_report_pdf()

    if pdf:
        test_ingest_local(pdf1, col1)
        test_ingest_local(pdf2, col2)
        test_rag_local(col1)
        test_compare_local([col1, col2])

    # Tests API (nécessite uvicorn en arrière-plan)
    test_api_endpoints(pdf1, pdf2, col1, col2)

    # Résumé
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print("\n" + "=" * 60)
    print(f"RÉSULTAT : {passed} PASS / {failed} FAIL / {len(results)} total")
    print("=" * 60)

    report_path = os.path.join("data", f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Rapport sauvegardé : {report_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
