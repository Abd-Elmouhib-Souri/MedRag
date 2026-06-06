"""
app/llm_chain.py
────────────────
Configuration du LLM de génération de réponses cliniques.

CORRECTION v2 : modèle unifié llama-3.3-70b-versatile
(cohérent avec rag_pipeline.py et patient_summary.py)
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_llm_chain():
    """
    Initialise le modèle LLaMA 3.3 via l'API Groq Cloud
    et configure la structure de consigne (Prompt Template).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("gsk_your_actual"):
        raise ValueError("[ERREUR] GROQ_API_KEY manquante ou non valide dans le fichier .env !")

    # CORRECTION : llama-3.3-70b-versatile (cohérent avec rag_pipeline.py)
    # llama-3.1-8b-instant était trop petit pour les réponses médicales précises
    # Si les réponses sont trop lentes (> 60s), repasser à "llama-3.1-8b-instant"
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )

    system_prompt = (
        "Tu es un assistant médical IA expert, rigoureux et précis.\n"
        "Analyse le contexte clinique fourni (extraits de dossiers ou protocoles) pour répondre à la question.\n\n"
        "RÈGLES STRICTES D'EXÉCUTION :\n"
        "1. Ne réponds QUE sur la base des faits explicitement mentionnés dans le contexte.\n"
        "2. Si le contexte ne contient pas la réponse, dis textuellement : 'Information non trouvée dans les documents fournis.'\n"
        "3. Ne formule aucune supposition ou extrapolation en dehors du texte.\n"
        "4. Structure ta réponse de manière claire (puces ou paragraphes).\n\n"
        "CONTEXTE DE RÉFÉRENCE :\n"
        "{context}\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # Chaîne LCEL : prompt → LLM → texte brut
    chain = prompt | llm | StrOutputParser()
    return chain