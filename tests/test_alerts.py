"""Clinical alert detection: whole-word, negation-aware."""

import pytest

from app.alert_detector import detect_alerts


@pytest.mark.parametrize("text, level, category", [
    ("Allergie : pénicilline (urticaire).", "CRITIQUE", "ALLERGIE"),
    ("Interaction warfarine-aspirine : risque hémorragique.", "CRITIQUE", "INTERACTION"),
    ("Transfert en réanimation immédiat.", "URGENCE", "URGENCE"),
])
def test_affirmed_findings_raise_alerts(text, level, category):
    result = detect_alerts(text)
    assert result.level == level and category in result.categories


@pytest.mark.parametrize("text", [
    "Aucune allergie connue.",
    "Pas d'interaction médicamenteuse retrouvée.",
    "No known drug interaction.",
    "Patient sans allergie.",
])
def test_negated_findings_do_not_alert(text):
    assert detect_alerts(text).level == "AUCUNE"


def test_high_risk_drug_and_its_negation():
    assert "warfarine" in detect_alerts("Traitement : warfarine 5 mg.").high_risk_drugs_found
    assert detect_alerts("Pas de warfarine.").high_risk_drugs_found == []


def test_short_words_match_whole_words_only():
    # "inr" must not fire inside an unrelated word
    assert "SURVEILLANCE" not in detect_alerts("Voir le compte rendu de Inrid.").categories
