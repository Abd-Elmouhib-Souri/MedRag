import pymupdf  # PyMuPDF (the old "fitz" import name is deprecated)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
import re

# Ces patterns détectent les informations médicales importantes
DRUG_PATTERN = re.compile(
    r'\b(mg|mcg|ml|UI|g/kg|mg/m²|mg/kg)\b',
    re.IGNORECASE
)
ALERT_PATTERN = re.compile(
    r'\b(contre-indication|interaction|allergie|surdosage|'
    r'précaution|avertissement|danger|toxique|ALERTE)\b',
    re.IGNORECASE
)

def load_pdf(file_path: str) -> list[dict]:
    """
    Ouvre un PDF et extrait le texte de chaque page.
    """
    doc = pymupdf.open(file_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()

        pages.append({
            "page": page_num + 1,
            "text": text,
            "has_alert": bool(ALERT_PATTERN.search(text)),
            "has_dosage": bool(DRUG_PATTERN.search(text)),
        })

    doc.close()
    return pages


def split_text_with_metadata(pages: list[dict]) -> list[dict]:
    """
    Découpe chaque page en petits morceaux (chunks) de 600 caractères.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []
    for page in pages:
        sub_chunks = splitter.split_text(page["text"])

        for chunk in sub_chunks:
            if len(chunk.strip()) > 30:  # Ignorer les résidus trop courts
                chunks.append({
                    "text": chunk,
                    "page": page["page"],
                    "has_alert": page["has_alert"],
                    "has_dosage": page["has_dosage"],
                })

    return chunks