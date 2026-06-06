import os
from app.document_loader import load_pdf, split_text_with_metadata

def run_test():
    pdf_path = "data/sample_docs/test1.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Erreur : Place d'abord un fichier nommé 'test.pdf' dans {os.path.dirname(pdf_path)}")
        return

    print("📖 Extraction du PDF en cours...")
    pages = load_pdf(pdf_path)
    print(f"✅ Pages lues avec succès : {len(pages)}")

    print("\n✂️ Découpage en morceaux (Chunks) en cours...")
    chunks = split_text_with_metadata(pages)
    print(f"✅ Nombre total de chunks créés : {len(chunks)}")

    if chunks:
        print("\n🔍 Aperçu du tout premier Chunk :")
        print("-" * 50)
        print(f"Page d'origine : {chunks[0]['page']}")
        print(f"Contient un dosage ? : {chunks[0]['has_dosage']}")
        print(f"Contient une alerte ? : {chunks[0]['has_alert']}")
        print(f"Texte :\n{chunks[0]['text']}")
        print("-" * 50)

if __name__ == "__main__":
    run_test()