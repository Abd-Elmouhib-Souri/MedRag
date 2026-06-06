"""
app/vector_store.py
───────────────────
Gestion des collections ChromaDB — Multi-documents isolés.

CORRECTION v2 : emojis dans print() remplacés par du texte
pour éviter UnicodeEncodeError sur Windows (cp1252).
"""

import os
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.embeddings import get_embeddings_model
import chromadb

CHROMA_DIR         = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
DEFAULT_COLLECTION = "medrag_default"


def _sanitize_collection_name(name: str) -> str:
    """
    ChromaDB impose des règles strictes sur les noms de collections :
    - Entre 3 et 63 caractères
    - Commence et finit par un alphanumérique
    - Contient uniquement lettres, chiffres, underscores et tirets
    """
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    name = name.strip("_-")
    name = name[:63]
    if len(name) < 3:
        name = name + "_doc"
    return name


def create_vector_store(chunks: list, collection_name: str = DEFAULT_COLLECTION) -> Chroma:
    """
    Prend une liste de chunks, génère leurs embeddings
    et les stocke dans une collection ChromaDB dédiée.
    Chaque PDF obtient sa propre collection isolée.
    """
    embeddings_model = get_embeddings_model()
    safe_name        = _sanitize_collection_name(collection_name)

    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "page":       chunk["page"],
                "has_alert":  chunk["has_alert"],
                "has_dosage": chunk["has_dosage"]
            }
        )
        documents.append(doc)

    # CORRECTION : print sans emoji (evite UnicodeEncodeError Windows cp1252)
    print(f"[INFO] Indexation de {len(documents)} fragments dans '{safe_name}' ({CHROMA_DIR})...")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        persist_directory=CHROMA_DIR,
        collection_name=safe_name
    )

    print(f"[OK] Collection '{safe_name}' creee avec succes.")
    return vector_store


def get_vector_store(collection_name: str = DEFAULT_COLLECTION) -> Chroma:
    """
    Recharge une collection ChromaDB existante depuis le disque
    sans recalculer les embeddings.
    """
    embeddings_model = get_embeddings_model()
    safe_name        = _sanitize_collection_name(collection_name)

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings_model,
        collection_name=safe_name
    )


def list_collections() -> list:
    """
    Retourne la liste de toutes les collections ChromaDB existantes.
    Utilisé par l'endpoint /collections et la sidebar Streamlit.
    """
    try:
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        # CORRECTION : print sans emoji
        print(f"[WARN] Impossible de lister les collections ChromaDB : {e}")
        return []


def delete_collection(collection_name: str) -> bool:
    """
    Supprime une collection ChromaDB.
    Utile pour retirer un document indexé sans tout réinitialiser.
    """
    try:
        safe_name = _sanitize_collection_name(collection_name)
        client    = chromadb.PersistentClient(path=CHROMA_DIR)
        client.delete_collection(safe_name)
        # CORRECTION : print sans emoji
        print(f"[OK] Collection '{safe_name}' supprimee.")
        return True
    except Exception as e:
        # CORRECTION : print sans emoji
        print(f"[WARN] Erreur suppression collection '{collection_name}' : {e}")
        return False