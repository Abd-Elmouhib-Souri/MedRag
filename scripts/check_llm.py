"""
scripts/check_llm.py
────────────────────
Smoke test for the Groq LLMs (the pytest suite uses a fake LLM):
  1. lists the chat models your key can use right now,
  2. checks MODEL_FAST rewrites a question and MODEL_STRONG answers from a document.

The API key is read from .env and never printed.

    python -m scripts.check_llm
"""

import os
import time

from app import config
from app.llm_chain import get_llm, get_llm_chain
from app.guardrails import wrap_context
from app.rag_pipeline import rewrite_query


def main() -> None:
    from groq import Groq  # installed with langchain-groq

    # 1) What does Groq offer to this key today? (models get retired: see console.groq.com/docs/deprecations)
    models = sorted(m.id for m in Groq(api_key=os.getenv("GROQ_API_KEY")).models.list().data)
    print("Models available to your key:\n  " + "\n  ".join(models))
    for name, model in (("MODEL_FAST", config.MODEL_FAST), ("MODEL_STRONG", config.MODEL_STRONG)):
        print(f"{name} = {model}  ->  {'available' if model in models else 'NOT AVAILABLE'}")

    # 2) Query rewriting with the fast model (strict: an error is raised, not hidden)
    question = "quell anticoagulan il prend ?"
    t0 = time.perf_counter()
    rewritten = rewrite_query(question, strict=True)
    print(f"\nrewrite ({time.perf_counter() - t0:.1f}s): {question!r} -> {rewritten!r}")

    # 3) A grounded answer with the strong model, from a one-page fictional record
    context = wrap_context([(2, "Traitement actuel : Warfarine 5 mg le soir, objectif INR entre 2 et 3.")])
    t0 = time.perf_counter()
    answer = get_llm_chain().invoke({"context": context, "question": "Quelle est la dose de warfarine ?"})
    print(f"answer  ({time.perf_counter() - t0:.1f}s): {answer!r}")


if __name__ == "__main__":
    main()
