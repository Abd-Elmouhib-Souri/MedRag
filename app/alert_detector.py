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
            "INR", "taux de prothrombine", "NFS", "ionogramme", "fonction rénale",
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


# Negation cues: "pas d'allergie", "aucune interaction", "no known allergy"...
# A keyword preceded by one of these in the same sentence (within ~40
# characters) is ignored, so a documented ABSENCE does not raise an alert.
NEGATION_RE = re.compile(
    r"\b(pas\s+d[e']|pas|aucun(e|s)?|sans|absence\s+d[e']|ni|non|jamais|n[ée]gati(f|ve)|"
    r"no|not|without|denies|denied|negative\s+for)\b[^.;\n]{0,40}$",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;\n])\s+")


def _keyword_regex(keyword: str) -> re.Pattern:
    """Whole-word match: 'TP' must not match inside 'http', 'bilan' inside 'bilans' is fine."""
    return re.compile(r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)")


def _affirmed(keyword: str, sentences: list[str]) -> bool:
    """True if the keyword appears at least once WITHOUT a negation just before it."""
    pattern = _keyword_regex(keyword)
    for sentence in sentences:
        for match in pattern.finditer(sentence):
            if not NEGATION_RE.search(sentence[: match.start()]):
                return True
    return False


def detect_alerts(answer: str, context: str = "") -> AlertResult:
    """
    Analyse the LLM answer (+ source context) and return an AlertResult.

    Matching is whole-word and negation-aware: "Allergie : pénicilline" raises
    an ALLERGIE alert, "Aucune allergie connue" does not.
    """
    result = AlertResult()
    full_text = (answer + "\n" + context).lower()
    sentences = [s for s in SENTENCE_SPLIT_RE.split(full_text) if s.strip()]

    # 1. Pattern categories
    for category, cfg in ALERT_PATTERNS.items():
        found = [kw for kw in cfg["keywords"] if _affirmed(kw, sentences)]
        if found:
            result.has_alert = True
            result.categories.append(category)
            result.level = _upgrade_level(result.level, cfg["level"])
            result.alerts.append(f"{cfg['icon']} {category.replace('_', ' ')} — terme détecté : « {found[0]} »")

    # 2. High-risk drugs (a drug name is a fact, so negation also applies: "pas de warfarine")
    result.high_risk_drugs_found = [d for d in HIGH_RISK_DRUGS if _affirmed(d, sentences)]
    if result.high_risk_drugs_found:
        result.has_alert = True
        result.level = _upgrade_level(result.level, "ATTENTION")
        drugs = ", ".join(result.high_risk_drugs_found[:3])
        result.alerts.append(f"💊 MÉDICAMENT À RISQUE — surveillance requise : {drugs}")

    if not result.has_alert:
        result.level = "AUCUNE"
    return result
