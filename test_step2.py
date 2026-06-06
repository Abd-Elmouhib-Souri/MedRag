from app.embeddings import get_embeddings_model

def run_test():
    print("🧠 Chargement du modèle d'embeddings local...")
    print("(Note : Le premier lancement va télécharger le modèle de ~120MB, cela peut prendre 1 à 2 minutes)\n")
    
    model = get_embeddings_model()
    print("✅ Modèle chargé avec succès !")

    # Préparation de phrases de test
    texts = [
        "La dose de Paracétamol est de 500mg",
        "Posologie : 500 milligrammes de Paracétamol",
        "Le ciel est bleu aujourd'hui à Monastir"
    ]
    
    print(
        f"🔄 Vectorisation de {len(texts)} phrases en cours..."
    )
    vectors = model.embed_documents(texts)
    
    print(f"✅ Vectorisation terminée !")
    print(f"📊 Taille d'un vecteur : {len(vectors[0])} dimensions (Attendu : 384)")
    
    if len(vectors[0]) == 384:
        print("\n🚀 ÉTAPE 2 RÉUSSIE ! Le modèle est prêt à alimenter ChromaDB.")

if __name__ == "__main__":
    run_test()