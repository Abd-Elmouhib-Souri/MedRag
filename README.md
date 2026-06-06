<div align="center">

# 🏥 MedRAG
### Medical Retrieval-Augmented Generation System

**Python 3.11** · **FastAPI** · **LangChain 1.3.x** · **ChromaDB** · **Streamlit** · **Groq llama-3.3-70b**

**An intelligent medical document assistant — query patient records in natural language, detect critical drug interactions, compare patients, and export structured PDF reports.**

</div>

---

## What it does

Upload any medical PDF and ask questions in plain language. MedRAG retrieves the relevant sections, generates a precise clinical answer, and automatically flags dangerous drug interactions, allergies, or overdose risks.

- **Single document mode** — interrogate one patient record
- **Comparison mode** — compare 2+ patients on the same question
- **Alert detection** — 4 severity levels (ATTENTION → URGENCE)
- **Patient cards** — auto-generated summary from each PDF
- **PDF export** — download the full consultation as a report

> All patient data used in this project is 100% fictional and generated for testing purposes.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| RAG Pipeline | LangChain 1.3.x — EnsembleRetriever |
| Vector Store | ChromaDB (semantic 70%) + BM25 (keyword 30%) |
| Embeddings | sentence-transformers multilingual (FR/EN/AR) |
| LLM | llama-3.3-70b-versatile via Groq API |
| PDF | PyMuPDF (extraction) + ReportLab (generation) |

---

## Project structure

```
medrag/
├── app/
│   ├── main.py               # FastAPI — 9 endpoints
│   ├── rag_pipeline.py       # Hybrid RAG + Query Rewriting + Comparison
│   ├── vector_store.py       # ChromaDB multi-collection management
│   ├── alert_detector.py     # Clinical alert detection engine
│   ├── patient_summary.py    # Auto patient card generation
│   ├── document_loader.py    # PDF extraction + chunking
│   ├── embeddings.py         # HuggingFace multilingual embeddings
│   ├── llm_chain.py          # Groq LLM chain
│   └── report_generator.py   # PDF report (ReportLab)
├── frontend/
│   └── ui.py                 # Streamlit interface
├── tests/
│   ├── test_step1.py         # PDF + chunking
│   ├── test_step2.py         # Embeddings
│   ├── test_step3.py         # ChromaDB
│   ├── test_step4.py         # LLM
│   └── test_step5.py         # Full RAG pipeline
├── data/                     # gitignored — patient data stays local
├── .env.example
└── requirements.txt
```

---

## Getting started

### Prerequisites

- Python 3.11+
- Conda
- A free Groq API key → [console.groq.com](https://console.groq.com)

### 1. Clone and setup

```bash
git clone https://github.com/Abd-Elmouhib-Souri/MedRag.git
cd MedRag

conda create -n medrag python=3.11
conda activate medrag

pip install -r requirements.txt
```

### 2. Configure environment

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
CHROMA_PERSIST_DIR=./data/chroma_db
SUMMARIES_DIR=./data/summaries
```

### 3. Create data directories

```bash
# Linux / Mac
mkdir -p data/chroma_db data/summaries

# Windows PowerShell
New-Item -ItemType Directory -Force -Path data\chroma_db, data\summaries
```

### 4. Start the backend

```bash
# Windows
$env:PYTHONIOENCODING='utf-8'
uvicorn app.main:app --reload --port 8000
```

Swagger docs available at `http://localhost:8000/docs`

### 5. Start the frontend

Open a second terminal:

```bash
conda activate medrag
streamlit run frontend/ui.py
```

Open `http://localhost:8501`

---

## How to test

### Step 1 — Upload a PDF
In the Streamlit sidebar, click **Parcourir**, select any medical PDF, then click **Indexer**.

### Step 2 — Ask questions

```
# Basic
"What is the main diagnosis?"
"Which medications is the patient taking?"
"Does the patient have any allergies?"

# Safety — should trigger alerts
"Can this patient take ibuprofen?"
"Are there any dangerous drug interactions?"

# Typo test — Query Rewriting auto-corrects
"quell son les alergie du patien"
```

### Step 3 — Compare patients
Switch to **Comparaison** mode in the sidebar, select 2 records, then ask:
```
"Which patient has the highest cardiovascular risk?"
"Compare the critical allergies of both patients"
```

### Step 4 — Export
Click **📄 Exporter rapport PDF** in the sidebar to download the full session as a PDF report.

### Run unit tests

```bash
conda activate medrag
python test_step1.py   # PDF extraction + chunking
python test_step2.py   # Embeddings
python test_step3.py   # ChromaDB
python test_step4.py   # LLM connection
python test_step5.py   # Full RAG pipeline
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/collections` | List indexed documents |
| `GET` | `/summaries` | All patient cards |
| `GET` | `/summaries/{name}` | Single patient card |
| `POST` | `/ingest` | Index a PDF |
| `POST` | `/ask` | Query a patient record |
| `POST` | `/compare` | Compare 2+ patients |
| `POST` | `/report` | Export PDF report |
| `DELETE` | `/collections/{name}` | Remove a document |

---

## Author

**Abd Elmouhib Souri**  
Student Engineer — Data Science, AI & Business Intelligence  
ESPRIM — Higher Private School of Engineering of Monastir

- LinkedIn : [linkedin.com/in/abd-elmouhib-souri](https://linkedin.com/in/abd-elmouhib-souri)
- GitHub : [github.com/Abd-Elmouhib-Souri](https://github.com/Abd-Elmouhib-Souri)
- Portfolio : [abd-elmouhib-souri.github.io](https://abd-elmouhib-souri.github.io)
