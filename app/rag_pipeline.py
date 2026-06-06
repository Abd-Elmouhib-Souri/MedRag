"""
app/rag_pipeline.py
───────────────────
Import EnsembleRetriever compatible toutes versions LangChain.
"""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.vector_store import get_vector_store
from app.llm_chain import get_llm_chain
from app.alert_detector import detect_alerts

# ── Import EnsembleRetriever — compatible toutes versions LangChain ───────────
try:
    # LangChain >= 0.2 / 1.x
    from langchain.retrievers import EnsembleRetriever
    print("[INFO] EnsembleRetriever charge depuis langchain.retrievers")
except ImportError:
    try:
        # Versions intermédiaires
        from langchain_community.retrievers import EnsembleRetriever
        print("[INFO] EnsembleRetriever charge depuis langchain_community.retrievers")
    except ImportError:
        # Fallback manuel si rien ne marche
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from typing import List
        import asyncio

        class EnsembleRetriever(BaseRetriever):
            """Fallback EnsembleRetriever manuel (RRF - Reciprocal Rank Fusion)."""
            retrievers: list
            weights: list

            def _get_relevant_documents(self, query: str) -> List[Document]:
                all_docs = {}
                scores   = {}
                for i, (retriever, weight) in enumerate(zip(self.retrievers, self.weights)):
                    try:
                        docs = retriever.invoke(query)
                        for rank, doc in enumerate(docs):
                            key = doc.page_content[:100]
                            if key not in all_docs:
                                all_docs[key] = doc
                                scores[key]   = 0.0
                            scores[key] += weight * (1.0 / (rank + 1))
                    except Exception as e:
                        print(f"[WARN] Retriever {i} error: {e}")
                sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
                return [all_docs[k] for k in sorted_keys]

        print("[INFO] EnsembleRetriever fallback manuel active")

DEFAULT_COLLECTION = "medrag_default"
MODEL_FAST         = "llama3-8b-8192"
MODEL_STRONG       = "llama-3.3-70b-versatile"


def corriger_et_enrichir_requete(question_utilisateur: str) -> str:
    llm = ChatGroq(model_name=MODEL_FAST, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Tu es un assistant médical expert en recherche documentaire. "
            "Ton seul rôle est de corriger les fautes de frappe, l'orthographe et de reformuler "
            "la question de l'utilisateur pour qu'elle soit parfaite pour une recherche par mots-clés "
            "et sémantique dans un dossier clinique. "
            "Ne réponds PAS à la question. "
            "Renvoie UNIQUEMENT la question corrigée, sans introduction, ni explication, ni guillemets."
        )),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"question": question_utilisateur}).strip()
    except Exception as e:
        print(f"[WARN] Erreur reecriture : {e}")
        return question_utilisateur


def _build_ensemble(vector_store, langchain_docs: list, top_k: int):
    """Construit le retriever hybride ChromaDB 70% + BM25 30%."""
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    bm25_retriever   = BM25Retriever.from_documents(langchain_docs, k=top_k)
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )


def query_medrag(
    question: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION
) -> dict:
    """Pipeline RAG complet — mode dossier unique."""

    question_propre = corriger_et_enrichir_requete(question)

    print("\n" + "="*55)
    print(f"[Q] Question d'origine : {question}")
    print(f"[Q] Question optimisee : {question_propre}")
    print(f"[Q] Collection cible   : {collection_name}")
    print("="*55 + "\n")

    vector_store = get_vector_store(collection_name=collection_name)
    all_docs     = vector_store.get()

    if not all_docs or not all_docs.get("documents"):
        return {
            "answer":  f"Aucun document trouvé dans '{collection_name}'. Veuillez d'abord indexer ce PDF.",
            "sources": [],
            "alert":   {"has_alert": False, "alerts": [], "level": "AUCUNE",
                        "high_risk_drugs_found": [], "categories": []}
        }

    langchain_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]

    ensemble    = _build_ensemble(vector_store, langchain_docs, top_k)
    docs        = ensemble.invoke(question_propre)[:top_k]

    context_str = ""
    sources     = []
    for doc in docs:
        page = doc.metadata.get("page", "Inconnue")
        context_str += f"[Page {page}] {doc.page_content}\n\n"
        sources.append({"page": page, "content": doc.page_content})

    chain        = get_llm_chain()
    answer       = chain.invoke({"context": context_str, "question": question_propre})
    alert_result = detect_alerts(answer=answer, context=context_str)

    if alert_result.has_alert:
        print(f"[ALERT] niveau {alert_result.level} : {alert_result.alerts}")

    return {"answer": answer, "sources": sources, "alert": alert_result.to_dict()}


def query_compare(
    question: str,
    collection_names: list,
    top_k: int = 4
) -> dict:
    """Couche 2 : Comparaison inter-patients."""

    empty_alert = {
        "has_alert": False, "alerts": [], "level": "AUCUNE",
        "high_risk_drugs_found": [], "categories": []
    }

    try:
        question_propre = corriger_et_enrichir_requete(question)

        print("\n" + "="*55)
        print("[COMPARE] MODE COMPARAISON")
        print(f"[COMPARE] Question    : {question_propre}")
        print(f"[COMPARE] Collections : {collection_names}")
        print("="*55 + "\n")

        per_patient  = {}
        combined_ctx = ""

        for col_name in collection_names:
            try:
                vs       = get_vector_store(collection_name=col_name)
                all_docs = vs.get()

                if not all_docs or not all_docs.get("documents"):
                    print(f"[WARN] Collection vide : {col_name}")
                    per_patient[col_name] = {"context": f"Aucun document dans '{col_name}'.", "sources": []}
                    combined_ctx += f"\n\nDOSSIER : {col_name}\nAucun document disponible.\n"
                    continue

                langchain_docs = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
                ]

                ensemble = _build_ensemble(vs, langchain_docs, top_k)
                docs     = ensemble.invoke(question_propre)[:top_k]

                patient_ctx     = ""
                patient_sources = []
                for doc in docs:
                    page = doc.metadata.get("page", "?")
                    patient_ctx += f"[Page {page}] {doc.page_content}\n"
                    patient_sources.append({"page": page, "content": doc.page_content})

                per_patient[col_name] = {"context": patient_ctx, "sources": patient_sources}
                label         = col_name.replace("_", " ").title()
                combined_ctx += f"\n\n{'='*40}\nDOSSIER : {label}\n{'='*40}\n{patient_ctx}"

            except Exception as e:
                print(f"[WARN] Erreur collection {col_name} : {e}")
                per_patient[col_name] = {"context": f"Erreur : {str(e)}", "sources": []}
                combined_ctx += f"\n\nDOSSIER : {col_name}\nErreur : {str(e)}\n"

        if not combined_ctx.strip():
            return {
                "answer":      "Aucun document disponible dans les collections selectionnees.",
                "per_patient": per_patient,
                "alert":       empty_alert,
                "mode":        "comparison"
            }

        llm = ChatGroq(model_name=MODEL_STRONG, temperature=0)

        compare_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Tu es un médecin expert en analyse comparative de dossiers cliniques. "
                "Compare objectivement les dossiers patients fournis et réponds à la question. "
                "Structure ta réponse en 3 parties : "
                "1) Analyse par patient, "
                "2) Tableau comparatif des points clés, "
                "3) Conclusion avec recommandation claire. "
                "Base-toi UNIQUEMENT sur les informations fournies. "
                "Si une information manque, indique 'Non renseigné'."
            )),
            ("user", "Question : {question}\n\nDossiers :\n{context}")
        ])

        chain        = compare_prompt | llm | StrOutputParser()
        answer       = chain.invoke({"question": question_propre, "context": combined_ctx})
        alert_result = detect_alerts(answer=answer, context=combined_ctx)

        return {
            "answer":      answer,
            "per_patient": per_patient,
            "alert":       alert_result.to_dict(),
            "mode":        "comparison"
        }

    except Exception as e:
        print(f"[ERROR] query_compare : {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer":      f"Erreur lors de la comparaison : {str(e)}",
            "per_patient": {},
            "alert":       empty_alert,
            "mode":        "comparison"
        }