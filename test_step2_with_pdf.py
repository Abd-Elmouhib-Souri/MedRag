import os
from app.document_loader import load_pdf, split_text_with_metadata
from app.embeddings import get_embeddings_model

def run_pdf_embedding_test():
    # 1. Détection du fichier PDF
    pdf_path = "data/sample_docs/test1.pdf"
    if not os.path.exists(pdf_path):
        if os.path.exists("data/sample_docs/DOSSIER MÉDICAL DE SYNTHÈSE.pdf"):
            pdf_path = "data/sample_docs/DOSSIER MÉDICAL DE SYNTHÈSE.pdf"
        else:
            print("❌ Erreur : Aucun PDF trouvé dans data/sample_docs/")
            return

    print(f"📖 [Étape 1] Lecture et découpage du fichier : {os.path.basename(pdf_path)}")
    pages = load_pdf(pdf_path)
    chunks = split_text_with_metadata(pages)
    print(f"✅ {len(chunks)} fragments créés.")

    # 2. Extraction uniquement du texte de chaque fragment
    print("\n🧠 [Étape 2] Chargement du modèle d'embeddings...")
    model = get_embeddings_model()
    
    print("🔄 Vectorisation de tes fragments médicaux en cours...")
    # On extrait juste la clé 'text' de chaque dictionnaire de chunk
    texts_to_embed = [chunk["text"] for chunk in chunks]
    
    # Génération des vecteurs
    vectors = model.embed_documents(texts_to_embed)
    print("✅ Vectorisation terminée avec succès !")

    # 3. Affichage du résultat mathématique
    print("\n📊 TABLEAU DE BORD DES VECTEURS :")
    print("=" * 50)
    print(f"Nombre de vecteurs générés : {len(vectors)} (1 par fragment)")
    print(f"Dimension de chaque vecteur  : {len(vectors[0])} nombres")
    print("=" * 50)

    # Petit aperçu numérique du premier fragment pour le plaisir des yeux
    print(f"\n🔢 Aperçu des 5 premiers nombres (sur 384) du Fragment 1 :")
    print(vectors[0][:5])
    print(f"👉 Ce fragment correspond au texte qui commence par : '{chunks[0]['text'][:40]}...'")

if __name__ == "__main__":
    run_pdf_embedding_test()