

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from datetime import datetime
import os

from app.rag_pipeline      import query_medrag, query_compare
from app.document_loader   import load_pdf, split_text_with_metadata
from app.vector_store      import get_vector_store, create_vector_store, list_collections, delete_collection
from app.report_generator  import generate_pdf_report
from app.patient_summary   import generate_patient_summary, load_patient_summary, load_all_summaries, delete_patient_summary

load_dotenv()

app = FastAPI(
    title="MedRAG API",
    description="API de recherche médicale intelligente — Multi-documents, Alertes Cliniques, Comparaison inter-patients"
)

# ── Modèles ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None

class CompareRequest(BaseModel):
    question: str
    collection_names: List[str]       # 2 collections minimum

class IngestRequest(BaseModel):
    file_path: str
    collection_name: Optional[str] = None

class ReportRequest(BaseModel):
    conversation: List[dict]
    collection_name: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {
        "message": "MedRAG API — Clinical Intelligence",
        "version": "2.0",
        "endpoints": [
            "GET    /collections              — lister les documents indexés",
            "GET    /summaries                — toutes les fiches patients",
            "GET    /summaries/{name}         — fiche d'un patient",
            "POST   /ingest                   — indexer un PDF + générer fiche",
            "POST   /ask                      — question sur un dossier",
            "POST   /compare                  — comparaison inter-patients",
            "POST   /report                   — exporter rapport PDF",
            "DELETE /collections/{name}       — supprimer un document",
        ]
    }


@app.get("/collections")
async def get_collections():
    """Liste toutes les collections ChromaDB disponibles."""
    try:
        names = list_collections()
        return {"collections": names, "count": len(names)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summaries")
async def get_all_summaries():
    """Retourne toutes les fiches patients générées automatiquement."""
    try:
        summaries = load_all_summaries()
        return {"summaries": summaries, "count": len(summaries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summaries/{collection_name}")
async def get_summary(collection_name: str):
    """Retourne la fiche d'un patient spécifique."""
    summary = load_patient_summary(collection_name)
    if not summary:
        raise HTTPException(
            status_code=404,
            detail=f"Fiche patient '{collection_name}' non trouvée. Réindexez le document."
        )
    return summary


@app.post("/ingest")
async def ingest_document(req: IngestRequest):
    """
    Indexe un PDF dans sa propre collection ChromaDB
    et génère automatiquement la fiche patient (Couche 3).
    """
    try:
        col_name = req.collection_name if req.collection_name else os.path.basename(req.file_path)

        pages  = load_pdf(req.file_path)
        chunks = split_text_with_metadata(pages)

        if not chunks:
            raise HTTPException(status_code=400, detail="Aucun texte extractible trouvé dans le PDF.")

        # Indexation ChromaDB
        create_vector_store(chunks, collection_name=col_name)

        # Génération fiche patient automatique (Couche 3)
        summary = generate_patient_summary(collection_name=col_name, chunks=chunks)

        return {
            "status":          "ok",
            "collection_name": col_name,
            "chunks_indexed":  len(chunks),
            "summary":         summary
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur d'ingestion : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_medrag(request: QueryRequest):
    """
    Pose une question à un dossier médical.
    Retourne la réponse + sources + alertes cliniques détectées.
    """
    if not request.collection_name:
        raise HTTPException(
            status_code=400,
            detail="Veuillez sélectionner un document (collection_name requis)."
        )
    result = query_medrag(
        question=request.question,
        collection_name=request.collection_name
    )
    return {
        "question": request.question,
        "answer":   result["answer"],
        "sources":  result["sources"],
        "alert":    result["alert"]
    }


@app.post("/compare")
async def compare_patients(request: CompareRequest):
    """
    Couche 2 : Compare plusieurs dossiers patients sur une question donnée.
    Exemple : {'question': 'Lequel a le risque cardiovasculaire le plus élevé ?',
               'collection_names': ['dossier_01_benali', 'dossier_02_trabelsi']}
    """
    if len(request.collection_names) < 2:
        raise HTTPException(
            status_code=400,
            detail="Au moins 2 collections requises pour la comparaison."
        )
    result = query_compare(
        question=request.question,
        collection_names=request.collection_names
    )
    return {
        "question":    request.question,
        "answer":      result["answer"],
        "per_patient": result["per_patient"],
        "alert":       result["alert"],
        "mode":        "comparison"
    }


@app.post("/report")
async def export_report(req: ReportRequest):
    """Génère un rapport PDF médical depuis l'historique de conversation."""
    try:
        if not req.conversation:
            raise HTTPException(status_code=400, detail="Aucun échange à exporter.")

        pdf_bytes = generate_pdf_report(
            conversation=req.conversation,
            collection_name=req.collection_name
        )
        filename = f"rapport_medrag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur rapport : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}")
async def remove_collection(collection_name: str):
    """Supprime une collection ChromaDB et la fiche patient associée."""
    success = delete_collection(collection_name)
    delete_patient_summary(collection_name)
    if success:
        return {"status": "ok", "deleted": collection_name}
    raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' introuvable.")