from dotenv import load_dotenv
load_dotenv()

from app.rag_pipeline import query_medrag

def test_industrial_pipeline():
    print("🚀 ========================================================")
    print("🎯 TEST DU MOTEUR RAG HYBRIDE PROFESSIONNEL")
    print("======================================================== 🚀\n")
    
    question = "Quel est le nom du patient, son âge et qui est son médecin traitant ?"
    print(f"❓ Question posée : '{question}'\n")
    
    try:
        # Appel de notre moteur hybride
        result = query_medrag(question)
        
        print("🤖 RÉPONSE DE L'IA :")
        print("-" * 60)
        print(result["answer"])
        print("-" * 60)
        
        print("\n📄 SOURCES UTILISÉES (FUSION HYBRIDE) :")
        for i, source in enumerate(result["sources"], 1):
            print(f"  [{i}] Page : {source['page']}")
            preview = source['content'].replace('\n', ' ').strip()[:90]
            print(f"      Extrait : {preview}...\n")
            
    except Exception as e:
        print(f"❌ Erreur lors du test : {e}")

if __name__ == "__main__":
    test_industrial_pipeline()