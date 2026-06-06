"""Génère des PDFs médicaux synthétiques pour les tests MedRAG."""
import os
import fitz

SAMPLE_DIR = os.path.join("data", "sample_docs")

PATIENTS = [
    {
        "filename": "dossier_01_benali_karim_diabete_hta.pdf",
        "text": """
DOSSIER MÉDICAL — BENALI KARIM
Âge : 58 ans | Groupe sanguin : A+
Médecin traitant : Dr. Amira Salhi

DIAGNOSTIC PRINCIPAL : Diabète de type 2 + Hypertension artérielle
Antécédents : Obésité (IMC 31), tabagisme sevré en 2020

TRAITEMENT ACTUEL :
- Metformine 1000 mg x2/jour
- Ramipril 5 mg le matin
- Atorvastatine 20 mg le soir

ALLERGIES : Pénicilline (urticaire), Sulfamides (éruption cutanée)

DERNIÈRE CONSULTATION : 12/03/2025
Glycémie à jeun : 1.42 g/L | HbA1c : 7.8%
Tension artérielle : 148/92 mmHg

PRÉCAUTION : Surveillance fonction rénale (créatinine) tous les 6 mois.
Contre-indication : association avec alcool en excès.
""",
    },
    {
        "filename": "dossier_02_trabelsi_nadia_irc.pdf",
        "text": """
DOSSIER MÉDICAL — TRABELSI NADIA
Âge : 67 ans | Groupe sanguin : O-
Médecin traitant : Dr. Hichem Boudhraa

DIAGNOSTIC PRINCIPAL : Insuffisance rénale chronique stade 3
Antécédents : Hypertension, lithiase rénale, anémie chronique

TRAITEMENT ACTUEL :
- Furosémide 40 mg le matin
- Warfarine 5 mg (INR cible 2-3)
- Érythropoïétine 4000 UI/semaine

ALLERGIES : Aucune connue

DERNIÈRE CONSULTATION : 05/04/2025
Créatinine : 185 µmol/L | DFG : 32 mL/min
INR : 2.4

ALERTE : Interaction médicamenteuse — surveiller INR avec warfarine.
Surveillance rapprochée ionogramme et fonction rénale.
""",
    },
]


def main():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    for patient in PATIENTS:
        path = os.path.join(SAMPLE_DIR, patient["filename"])
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), patient["text"].strip(), fontsize=11)
        doc.save(path)
        doc.close()
        print(f"Created {path}")


if __name__ == "__main__":
    main()
