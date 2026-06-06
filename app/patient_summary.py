"""
app/patient_summary.py
──────────────────────
Couche 3 : Génération automatique d'une fiche de synthèse patient.

Au moment de l'indexation d'un PDF, interroge le LLM pour extraire :
  - Diagnostic principal
  - Médicaments actifs
  - Allergies connues
  - Antécédents majeurs
  - Dernière consultation

La fiche est stockée dans un fichier JSON local (./data/summaries/)
et rechargée par l'UI dans la sidebar.

CORRECTIONS v2 :
  - Prompt JSON plus directif (pas de markdown, pas de texte parasite)
  - Retry automatique jusqu'à 3 tentatives
  - Extraction JSON robuste avec regex en fallback
  - Encodage UTF-8 explicite partout
"""

import os
import re
import json
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

SUMMARIES_DIR = os.getenv("SUMMARIES_DIR", "./data/summaries")

# Prompt très directif — force le JSON pur sans texte parasite
FICHE_PROMPT = """Analyse ce dossier médical et extrais les informations dans le JSON ci-dessous.
IMPORTANT : Réponds UNIQUEMENT avec le JSON. Aucun texte avant. Aucun texte après. Aucun markdown. Aucun backtick.

Dossier médical :
{text}

JSON (remplis chaque champ, mets null si absent, [] pour les listes vides) :
{{
  "nom_patient": null,
  "age": null,
  "diagnostic_principal": null,
  "diagnostics_secondaires": [],
  "medicaments_actifs": [],
  "allergies": [],
  "antecedents": [],
  "derniere_consultation": null,
  "medecin_referent": null,
  "groupe_sanguin": null,
  "resume_clinique": null
}}"""


def _get_summary_path(collection_name: str) -> Path:
    Path(SUMMARIES_DIR).mkdir(parents=True, exist_ok=True)
    safe = collection_name.replace("/", "_").replace("\\", "_")
    return Path(SUMMARIES_DIR) / f"{safe}.json"


def generate_patient_summary(collection_name: str, chunks: list) -> dict:
    """
    Génère une fiche synthèse patient avec retry automatique (max 3 tentatives).

    Args:
        collection_name : nom de la collection (= identifiant patient)
        chunks          : liste de dicts {"text": str, "page": int, ...}

    Returns:
        dict avec les champs de la fiche patient
        _generated = True  si succès
        _generated = False si échec après 3 tentatives (fallback JSON)
    """
    # Prendre les 15 premiers chunks pour couvrir l'en-tête du dossier
    sample_text = "\n\n".join([c["text"] for c in chunks[:15]])

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024
    )

    MAX_RETRIES = 3
    last_error  = None

    for attempt in range(MAX_RETRIES):
        try:
            # Appel direct au LLM (sans ChatPromptTemplate pour éviter
            # les couches supplémentaires qui peuvent ajouter du texte)
            raw = llm.invoke(FICHE_PROMPT.format(text=sample_text[:3000]))
            raw = raw.content if hasattr(raw, "content") else str(raw)
            raw = raw.strip()

            # Nettoyage étape 1 — supprimer les backticks markdown
            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw = part
                        break

            # Nettoyage étape 2 — extraire le JSON même s'il y a du texte autour
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                raw = json_match.group()

            summary = json.loads(raw)
            summary["collection_name"] = collection_name
            summary["_generated"]      = True

            # Sauvegarde locale
            path = _get_summary_path(collection_name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"[OK] Fiche patient generee (tentative {attempt + 1}/{MAX_RETRIES}) : {path}")
            return summary

        except json.JSONDecodeError as e:
            last_error = f"JSONDecodeError : {str(e)}"
            print(f"[WARN] Tentative {attempt + 1}/{MAX_RETRIES} - JSON invalide : {last_error[:80]}")
        except Exception as e:
            last_error = str(e)
            print(f"[WARN] Tentative {attempt + 1}/{MAX_RETRIES} echouee : {last_error[:80]}")

    # ── Fallback après 3 échecs ────────────────────────────────────────────────
    print(f"[WARN] Fiche patient non generee apres {MAX_RETRIES} tentatives - mode fallback")

    fallback = {
        "collection_name":         collection_name,
        "nom_patient":             collection_name.replace("_", " ").title(),
        "age":                     "Non renseigné",
        "diagnostic_principal":    "Non extrait automatiquement",
        "diagnostics_secondaires": [],
        "medicaments_actifs":      [],
        "allergies":               [],
        "antecedents":             [],
        "derniere_consultation":   "Non renseigné",
        "medecin_referent":        "Non renseigné",
        "groupe_sanguin":          "Non renseigné",
        "resume_clinique":         "Fiche générée en mode dégradé — réindexez le PDF pour régénérer.",
        "_generated":              False,
        "_error":                  last_error,
    }

    path = _get_summary_path(collection_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fallback, f, ensure_ascii=False, indent=2)

    return fallback


def load_patient_summary(collection_name: str) -> dict | None:
    """
    Charge la fiche patient depuis le fichier JSON local.
    Retourne None si la fiche n'existe pas encore.
    """
    path = _get_summary_path(collection_name)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_all_summaries() -> dict:
    """
    Charge toutes les fiches patients disponibles.
    Retourne un dict {collection_name: summary_dict}.
    """
    result      = {}
    summary_dir = Path(SUMMARIES_DIR)
    if not summary_dir.exists():
        return result
    for file in summary_dir.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data     = json.load(f)
                col_name = data.get("collection_name", file.stem)
                result[col_name] = data
        except Exception:
            continue
    return result


def delete_patient_summary(collection_name: str) -> bool:
    """Supprime la fiche patient d'une collection."""
    path = _get_summary_path(collection_name)
    if path.exists():
        path.unlink()
        return True
    return False