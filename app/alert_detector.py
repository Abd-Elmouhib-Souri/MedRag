"""
app/alert_detector.py
─────────────────────
Couche 1 : Détection d'alertes cliniques automatique.

Analyse la réponse du LLM et le contexte source pour détecter :
  - Interactions médicamenteuses dangereuses
  - Contre-indications
  - Surdosages
  - Allergies
  - Alertes de surveillance

Retourne un bloc structuré affiché en orange dans l'UI.
"""

import re
from dataclasses import dataclass, field

# ── Dictionnaire de patterns cliniques ────────────────────────────────────────
ALERT_PATTERNS = {
    "INTERACTION": {
        "level": "CRITIQUE",
        "keywords": [
            "interaction", "interactions médicamenteuses", "drug interaction",
            "incompatible", "incompatibilité", "contre-indiqué avec",
            "ne pas associer", "association déconseillée",
        ],
        "icon": "⚡"
    },
    "ALLERGIE": {
        "level": "CRITIQUE",
        "keywords": [
            "allergie", "allergique", "anaphylaxie", "anaphylactique",
            "hypersensibilité", "intolérance", "réaction allergique",
            "urticaire", "choc anaphylactique",
        ],
        "icon": "🚨"
    },
    "SURDOSAGE": {
        "level": "DANGER",
        "keywords": [
            "surdosage", "surdose", "overdose", "toxicité", "toxique",
            "dose excessive", "dépasse la dose", "intoxication",
            "dose maximale dépassée",
        ],
        "icon": "⚠️"
    },
    "CONTRE_INDICATION": {
        "level": "DANGER",
        "keywords": [
            "contre-indication", "contre-indiqué", "contraindication",
            "ne pas utiliser", "interdit", "déconseillé",
            "précaution d'emploi", "prudence",
        ],
        "icon": "🔴"
    },
    "SURVEILLANCE": {
        "level": "ATTENTION",
        "keywords": [
            "surveillance", "surveiller", "monitoring", "contrôle",
            "suivi rapproché", "bilan", "créatinine", "glycémie",
            "INR", "TP", "NFS", "ionogramme", "fonction rénale",
            "insuffisance rénale", "hépatique",
        ],
        "icon": "👁️"
    },
    "URGENCE": {
        "level": "URGENCE",
        "keywords": [
            "urgence", "urgent", "immédiat", "hospitalisation",
            "réanimation", "soins intensifs", "transfert",
            "appeler le 15", "SAMU", "détresse",
        ],
        "icon": "🆘"
    }
}

# Médicaments à haut risque — déclenchent une alerte de surveillance automatique
HIGH_RISK_DRUGS = [
    "warfarine", "coumadine", "héparine", "lovenox", "enoxaparine",
    "méthotrexate", "lithium", "digoxine", "amiodarone", "cordarone",
    "insuline", "metformine", "sulfamide", "glibenclamide",
    "morphine", "tramadol", "oxycodone", "fentanyl",
    "aminoside", "gentamicine", "vancomycine",
    "ciclosporine", "tacrolimus",
    "phenytoine", "carbamazépine", "valproate",
    "clozapine", "olanzapine",
]


@dataclass
class AlertResult:
    has_alert: bool = False
    alerts: list = field(default_factory=list)
    level: str = "AUCUNE"          # AUCUNE / ATTENTION / DANGER / CRITIQUE / URGENCE
    high_risk_drugs_found: list = field(default_factory=list)
    categories: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "has_alert":             self.has_alert,
            "alerts":                self.alerts,
            "level":                 self.level,
            "high_risk_drugs_found": self.high_risk_drugs_found,
            "categories":            self.categories,
        }


# Ordre de sévérité pour déterminer le niveau global
SEVERITY_ORDER = ["AUCUNE", "ATTENTION", "DANGER", "CRITIQUE", "URGENCE"]


def _upgrade_level(current: str, new: str) -> str:
    """Retourne le niveau le plus sévère entre les deux."""
    return new if SEVERITY_ORDER.index(new) > SEVERITY_ORDER.index(current) else current


def detect_alerts(answer: str, context: str = "") -> AlertResult:
    """
    Analyse la réponse du LLM (+ contexte source) et retourne un AlertResult.

    Args:
        answer  : texte de la réponse générée par le LLM
        context : texte brut des chunks sources (optionnel, enrichit la détection)

    Returns:
        AlertResult avec tous les détails des alertes trouvées
    """
    result  = AlertResult()
    full_text = (answer + " " + context).lower()

    # 1. Détection par catégorie de patterns
    for category, config in ALERT_PATTERNS.items():
        found_keywords = [kw for kw in config["keywords"] if kw.lower() in full_text]
        if found_keywords:
            result.has_alert = True
            result.categories.append(category)
            result.level = _upgrade_level(result.level, config["level"])

            # Construire un message d'alerte lisible
            sample = found_keywords[0]
            alert_msg = f"{config['icon']} {category.replace('_', ' ')} — terme détecté : « {sample} »"
            result.alerts.append(alert_msg)

    # 2. Détection médicaments à haut risque
    for drug in HIGH_RISK_DRUGS:
        if drug.lower() in full_text:
            result.high_risk_drugs_found.append(drug)

    if result.high_risk_drugs_found:
        result.has_alert = True
        result.level = _upgrade_level(result.level, "ATTENTION")
        drugs_str = ", ".join(result.high_risk_drugs_found[:3])
        result.alerts.append(f"💊 MÉDICAMENT À RISQUE — surveillance requise : {drugs_str}")

    # 3. Si aucune alerte → niveau AUCUNE
    if not result.has_alert:
        result.level = "AUCUNE"

    return result