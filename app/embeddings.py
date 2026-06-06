# app/embeddings.py
# CORRECTION : langchain_community.embeddings deprecated dans LangChain 1.3.x
# Utiliser langchain_huggingface à la place
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings_model():
    """
    Charge le modèle d'embeddings multilingue.

    Tourne localement sur le PC — GRATUIT — ~120MB au premier téléchargement.
    Supporte : Français, Anglais, Arabe et 50+ autres langues.

    Pourquoi "paraphrase-multilingual-MiniLM-L12-v2" ?
    -> "multilingual" : supporte 50+ langues dont le français
    -> "MiniLM"       : petit modèle, rapide et économe en ressources
    -> "L12"          : 12 couches transformer (compromis vitesse/précision)
    -> Dimension      : 384 vecteurs par chunk
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )