"""
tests/test_drug_classes.py
──────────────────────────
Drug-class expansion: detection rules, and its effect on retrieval (offline fakes).
"""

from app import config
from app.drug_classes import expand_query, find_molecules
from app.rag_pipeline import retrieve
from app.vector_store import create_vector_store


def test_class_detected_with_accents_hyphens_plural_and_english():
    # Every spelling a user may type must map to the same molecules
    for text in ("Quel bêtabloquant ?", "un bêta-bloquant", "BETABLOQUANTS", "Which beta blocker?"):
        assert "bisoprolol" in find_molecules(text), text


def test_no_false_hit_inside_other_words():
    # "statistique" must not trigger "statine", "ainsi" must not trigger "ains"
    assert find_molecules("Donne-moi une statistique, ainsi que la date.") == []


def test_arabic_fallback():
    assert "warfarine" in find_molecules("ما هو الدواء المضاد للتخثر؟")


def test_no_class_no_expansion():
    assert expand_query("Quel est le dernier INR ?", "Quel est le dernier INR ?") is None


def test_expansion_keeps_the_question_then_lists_molecules():
    q = expand_query("Quel IEC prend-il ?", "Quel IEC prend-il ?")
    assert q.startswith("Quel IEC prend-il ?") and "ramipril" in q


def _index_record():
    chunks = [{"text": "Bisoprolol 2,5 mg le matin.", "page": 2, "has_alert": False, "has_dosage": True},
              {"text": "Allergie documentée à l'iode.", "page": 1, "has_alert": False, "has_dosage": False}]
    create_vector_store(chunks, collection_name="classes_test")


def test_retrieve_finds_the_molecule_only_when_expansion_is_on(monkeypatch):
    _index_record()
    question = "Quel bêtabloquant prend-il ?"

    monkeypatch.setattr(config, "DRUG_CLASS_EXPANSION", False)
    off = retrieve(question, question, "classes_test")
    assert off.refused  # the fake reranker sees no shared word: the old failure

    # The fake reranker scores the SHARE of query words found in the chunk, so a long
    # molecule list dilutes it (0.09 here); the real cross-encoder is measured by
    # scripts/evaluate.py. Here we only test the plumbing, with a low threshold.
    monkeypatch.setattr(config, "RELEVANCE_THRESHOLD", 0.05)
    monkeypatch.setattr(config, "DRUG_CLASS_EXPANSION", True)
    on = retrieve(question, question, "classes_test")
    assert len(on.queries) == 2 and "bisoprolol" in on.queries[-1]
    assert off.top_score == 0 and on.top_score > 0             # the molecule is now matched
    assert not on.refused and on.kept[0][0].metadata["page"] == 2
