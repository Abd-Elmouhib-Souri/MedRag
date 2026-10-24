"""
app/embeddings.py
─────────────────
Multilingual sentence embeddings (French, English, Arabic and 50+ languages).

paraphrase-multilingual-MiniLM-L12-v2: 12 transformer layers, 384-dimensional
vectors, ~470 MB download (large multilingual vocabulary), runs locally on CPU for free.
"""

from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def get_embeddings_model():
    """Load the embedding model once per process (it used to reload on every call)."""
    from langchain_huggingface import HuggingFaceEmbeddings  # lazy: tests use a fake model

    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
