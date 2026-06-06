import os
from dotenv import load_dotenv

# Charger les variables du fichier .env
load_dotenv()

from app.document_loader import load_pdf, split_text_with_metadata
from app.vector_store import create_vector_store

def run_test():
    pdf_path = "data/sample_docs/test1.pdf"
    
    if not os.path.exists(pdf_path):
        # Sécurité si tu as gardé le nom d'origine
        if os.path.exists("data/sample_docs/DOSSIER MÉDICAL DE SYNTHÈSE.pdf"):
            pdf_path = "data/sample_docs/DOSSIER MÉDICAL DE SYNTHÈSE.pdf"
        else:
            print("❌ Erreur : Place ton PDF dans data/sample_docs/")
            return

    print("⏳ [1/3] Traitement du fichier PDF...")
    pages = load_pdf(pdf_path)
    chunks = split_text_with_metadata(pages)
    
    print("⏳ [2/3] Insertion et vectorisation dans ChromaDB...")
    db = create_vector_store(chunks)
    print("✅ Base de données vectorielle sauvegardée sur le disque !")

    print("\n🔍 [3/3] Test de recherche sémantique en cours...")
    query = "Quelles sont les contre-indications ou allergies graves de ce patient ?"
    print(f"❓ Question posée : '{query}'")
    
    # On demande à ChromaDB de nous sortir les 2 morceaux les plus pertinents (k=2)
    results = db.similarity_search(query, k=2)
    
    print("\n🎯 ÉLÉMENTS EXTRAITS PAR CHROMADB :")
    print("=" * 60)
    for i, doc in enumerate(results):
        print(f"\n📄 Fragment {i+1} — Trouvé en Page {doc.metadata.get('page')}")
        print(f"⚠️ Alerte critique détectée ? : {doc.metadata.get('has_alert')}")
        print(f"📝 Extrait du texte :\n{doc.page_content}")
        print("-" * 40)
    print("=" * 60)

if __name__ == "__main__":
    run_test()