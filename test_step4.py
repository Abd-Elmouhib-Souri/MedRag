import os
from dotenv import load_dotenv
load_dotenv()

from app.vector_store import get_vector_store
from app.llm_chain import get_llm_chain

def run_integration_test():
    print("📦 Chargement de l'index ChromaDB...")
    db = get_vector_store()
    
    query = "Quelles sont les contre-indications ou allergies graves de ce patient ?"
    print(f"❓ Question posée : '{query}'")

    # 1. Simulation du pipeline : Recherche sémantique
    print("🔍 Recherche des fragments pertinents...")
    docs = db.similarity_search(query, k=2)
    
    # 2. Agrégation du contexte textuel
    context_str = ""
    for doc in docs:
        context_str += f"[Page {doc.metadata.get('page')}] {doc.page_content}\n\n"

    # 3. Initialisation et appel du LLM
    print("🧠 Initialisation du LLM LLaMA 3 sur Groq...")
    try:
        chain = get_llm_chain()
        print("🚀 Lancement de la génération de la réponse...")
        
        response = chain.invoke({
            "context": context_str,
            "question": query
        })
        
        print("\n🤖 RÉPONSE FINALE SYNTHÉTISÉE PAR MEDRAG :")
        print("=" * 60)
        print(response)
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Erreur pendant le test : {e}")

if __name__ == "__main__":
    run_integration_test()