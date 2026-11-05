"""
app/drug_classes.py
───────────────────
Drug-CLASS query expansion.

The problem (seen once on the held-out test set): a record lists molecules
("Bisoprolol 2,5 mg"), the user names a class ("Quel bêtabloquant prend-il ?").
Nothing links the two words: BM25 needs the same token, the multilingual embeddings
do not know pharmacology, and the cross-encoder gave the pair a score under the
relevance gate, so MedRAG refused a question it could answer.

The fix: when a question names a class from the table below, MedRAG searches with
ONE extra query that lists the molecules of that class (rag_pipeline.retrieve).
Retrieval and the gate can then match "bisoprolol" in the record. The LLM still
answers the user's own question, and still must refuse if the record has no
molecule of that class.

Limits, stated honestly:
  - the table is small, hand-written, and French (the language of the records);
  - it covers common classes only; a missing class falls back to the old behaviour;
  - it was written AFTER the test set revealed the problem, so the held-out test set
    can no longer measure it. It is evaluated on new dev questions
    (eval/questions_dev_classes.json), written before this file.
"""

import re
import unicodedata

# Class name (and its common variants) -> molecules (French INN, as written in records).
# Keys are matched after accent/case/hyphen normalisation, so "Bêta-bloquant",
# "betabloquants" and "BÉTABLOQUANT" all hit "betabloquant".
_BETA_BLOCKERS = ["bisoprolol", "aténolol", "métoprolol", "propranolol", "nébivolol", "carvédilol", "acébutolol"]
_AVK = ["warfarine", "acénocoumarol", "fluindione"]
_DOAC = ["apixaban", "rivaroxaban", "dabigatran", "édoxaban"]
_HEPARINS = ["héparine", "énoxaparine", "tinzaparine"]
_ANTIPLATELETS = ["aspirine", "clopidogrel", "ticagrélor", "prasugrel"]
_STATINS = ["atorvastatine", "rosuvastatine", "simvastatine", "pravastatine", "fluvastatine"]
_ACE_INHIBITORS = ["ramipril", "périndopril", "énalapril", "lisinopril", "captopril"]
_SARTANS = ["losartan", "valsartan", "irbésartan", "candésartan", "olmésartan", "telmisartan"]
_CALCIUM_BLOCKERS = ["amlodipine", "nifédipine", "lercanidipine", "diltiazem", "vérapamil"]
_DIURETICS = ["furosémide", "hydrochlorothiazide", "indapamide", "spironolactone"]
_BIGUANIDES = ["metformine"]
_GLIPTINS = ["sitagliptine", "vildagliptine", "saxagliptine", "linagliptine"]
_SULFONYLUREAS = ["gliclazide", "glimépiride", "glibenclamide"]
_SGLT2 = ["empagliflozine", "dapagliflozine", "canagliflozine"]
_GLP1 = ["liraglutide", "dulaglutide", "sémaglutide"]
_ANTIEPILEPTICS = ["valproate", "lévétiracétam", "lamotrigine", "carbamazépine", "éthosuximide", "topiramate"]
_INHALED_STEROIDS = ["budésonide", "béclométasone", "fluticasone"]
_BETA2_AGONISTS = ["salbutamol", "terbutaline", "formotérol", "salmétérol"]
_ANTIBIOTICS = ["amoxicilline", "pénicilline", "azithromycine", "ciprofloxacine", "doxycycline", "céftriaxone"]
_NSAIDS = ["ibuprofène", "kétoprofène", "diclofénac", "naproxène"]
_PPIS = ["oméprazole", "ésoméprazole", "pantoprazole", "lansoprazole"]

DRUG_CLASSES: dict[str, list[str]] = {
    # cardiology
    "betabloquant": _BETA_BLOCKERS, "beta bloquant": _BETA_BLOCKERS, "beta blocker": _BETA_BLOCKERS,
    "anticoagulant": _AVK + _DOAC + _HEPARINS, "blood thinner": _AVK + _DOAC + _HEPARINS,
    "antivitamine k": _AVK, "anti vitamine k": _AVK, "avk": _AVK, "vitamin k antagonist": _AVK,
    "aod": _DOAC, "anticoagulant oral direct": _DOAC, "doac": _DOAC,
    "heparine": _HEPARINS, "hbpm": _HEPARINS,
    "antiagregant": _ANTIPLATELETS, "antiplaquettaire": _ANTIPLATELETS, "antiplatelet": _ANTIPLATELETS,
    "statine": _STATINS, "statin": _STATINS, "hypolipemiant": _STATINS,
    "iec": _ACE_INHIBITORS, "inhibiteur de l enzyme de conversion": _ACE_INHIBITORS,
    "ace inhibitor": _ACE_INHIBITORS,
    "sartan": _SARTANS, "ara2": _SARTANS, "ara ii": _SARTANS, "arb": _SARTANS,
    "antagoniste des recepteurs de l angiotensine": _SARTANS,
    "inhibiteur calcique": _CALCIUM_BLOCKERS, "calcium channel blocker": _CALCIUM_BLOCKERS,
    "diuretique": _DIURETICS, "diuretic": _DIURETICS,
    "antihypertenseur": _ACE_INHIBITORS + _SARTANS + _BETA_BLOCKERS + _CALCIUM_BLOCKERS + _DIURETICS,
    "antihypertensive": _ACE_INHIBITORS + _SARTANS + _BETA_BLOCKERS + _CALCIUM_BLOCKERS + _DIURETICS,
    # diabetes
    "antidiabetique": _BIGUANIDES + _GLIPTINS + _SULFONYLUREAS + _SGLT2 + _GLP1 + ["insuline"],
    "antidiabetic": _BIGUANIDES + _GLIPTINS + _SULFONYLUREAS + _SGLT2 + _GLP1 + ["insuline"],
    "hypoglycemiant": _BIGUANIDES + _GLIPTINS + _SULFONYLUREAS + _SGLT2 + _GLP1 + ["insuline"],
    "biguanide": _BIGUANIDES,
    "gliptine": _GLIPTINS, "dpp 4": _GLIPTINS, "dpp4": _GLIPTINS, "gliptin": _GLIPTINS,
    "sulfamide hypoglycemiant": _SULFONYLUREAS, "sulfonylurea": _SULFONYLUREAS,
    "isglt2": _SGLT2, "sglt2": _SGLT2, "glp 1": _GLP1, "glp1": _GLP1,
    # neurology
    "antiepileptique": _ANTIEPILEPTICS, "anticonvulsivant": _ANTIEPILEPTICS,
    "antiepileptic": _ANTIEPILEPTICS, "anticonvulsant": _ANTIEPILEPTICS,
    # respiratory
    "corticoide inhale": _INHALED_STEROIDS, "corticosteroide inhale": _INHALED_STEROIDS,
    "inhaled corticosteroid": _INHALED_STEROIDS, "inhaled steroid": _INHALED_STEROIDS,
    "bronchodilatateur": _BETA2_AGONISTS + ["tiotropium"], "bronchodilator": _BETA2_AGONISTS + ["tiotropium"],
    "beta 2 agoniste": _BETA2_AGONISTS, "beta2 agoniste": _BETA2_AGONISTS, "beta 2 mimetique": _BETA2_AGONISTS,
    "beta2 mimetique": _BETA2_AGONISTS, "beta agonist": _BETA2_AGONISTS, "beta 2 agonist": _BETA2_AGONISTS,
    # infection, pain, stomach
    "antibiotique": _ANTIBIOTICS, "antibiotic": _ANTIBIOTICS,
    "ains": _NSAIDS, "anti inflammatoire non steroidien": _NSAIDS, "nsaid": _NSAIDS,
    "ipp": _PPIS, "inhibiteur de la pompe a protons": _PPIS, "proton pump inhibitor": _PPIS,
}

# Arabic names, matched as plain substrings (Arabic glues the article "ال" to the word,
# so word boundaries do not work). The rewriting step translates Arabic questions into
# French anyway; these are a fallback for when rewriting fails.
ARABIC_CLASSES: dict[str, list[str]] = {
    "تخثر": _AVK + _DOAC + _HEPARINS,       # مضاد للتخثر / مضادات التخثر = anticoagulant
    "مميع": _AVK + _DOAC + _HEPARINS,       # مميعات الدم = blood thinner
    "حاصر": _BETA_BLOCKERS,                 # حاصرات بيتا = beta blocker
    "ستاتين": _STATINS,
    "صرع": _ANTIEPILEPTICS,                 # مضاد للصرع = antiepileptic
}


def _normalise(text: str) -> str:
    """Lower case, no accents, hyphens/apostrophes -> spaces: 'Bêta-bloquant' -> 'beta bloquant'."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[-'’]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# One regex per key, compiled once: whole words, optional plural "s"/"x"
# ("statines", "iec" but not "fiecal", "statine" but not "statistique").
_PATTERNS = [(re.compile(rf"\b{re.escape(key)}[sx]?\b"), mols) for key, mols in DRUG_CLASSES.items()]


def find_molecules(*texts: str) -> list[str]:
    """
    Molecules of every drug class named in the texts (question and rewrite), in table
    order, without duplicates. Empty list = no class named = no expansion.
    """
    found: list[str] = []
    for text in texts:
        if not text:
            continue
        norm = _normalise(text)
        for pattern, molecules in _PATTERNS:
            if pattern.search(norm):
                found += molecules
        for word, molecules in ARABIC_CLASSES.items():
            if word in text:
                found += molecules
    return list(dict.fromkeys(found))  # dedupe, keep order


def expand_query(question: str, rewritten: str) -> str | None:
    """
    The extra search query, or None when no class is named.

    Built on the rewritten (French) question so the cross-encoder still sees a natural
    question, followed by the molecule names that BM25 and the reranker can match:
    "Quel bêtabloquant prend-il ? Molécules possibles : bisoprolol, aténolol, ..."
    """
    molecules = find_molecules(question, rewritten)
    if not molecules:
        return None
    return f"{rewritten.strip()} Molécules possibles : {', '.join(molecules)}."
