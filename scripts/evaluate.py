"""
scripts/evaluate.py
───────────────────
Measure MedRAG on the fictional evaluation corpus.

Protocol (dev/test split, so the final numbers are not tuned on themselves):

Step 1 — calibrate the relevance gate on the DEV questions (~1 minute):
    python -m scripts.evaluate --calibrate --rewrite   # rewritten questions, as at runtime
    python -m scripts.evaluate --calibrate             # raw questions, no Groq key needed

Step 2 — full pipeline on DEV, to study errors (needs GROQ_API_KEY):
    python -m scripts.evaluate --split dev --threshold <value from step 1>

Step 3 — ONCE, at the end, on the held-out TEST questions (the numbers to report):
    python -m scripts.evaluate --split test --threshold <same value>

Outputs eval/results_<split>.json and eval/RESULTS_<split>.md.
"""

import argparse
import json
import shutil
import statistics
import time
import unicodedata
from pathlib import Path

from app import config

EVAL = Path("eval")


def norm(text: str) -> str:
    """Lower-case, accent-free text so '8,1 %' matches '8,1%' and 'pénicilline' matches 'penicilline'."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c)).replace(" ", "")


QUESTION_FILES = {"dev": "questions.json", "test": "questions_test.json"}


def _term_found(term, answer: str) -> bool:
    """An expected term is a string, or a list of equivalent phrasings (any one is enough)."""
    options = term if isinstance(term, list) else [term]
    return any(norm(option) in norm(answer) for option in options)


def load_corpus(split: str = "dev") -> list[dict]:
    """Generate the corpus if needed, index every record, return the questions of one split."""
    if not (EVAL / QUESTION_FILES[split]).exists():
        from scripts import make_eval_corpus
        make_eval_corpus.main()

    from app.document_loader import load_pdf, split_text_with_metadata
    from app.vector_store import create_vector_store

    for pdf in sorted((EVAL / "corpus").glob("*.pdf")):
        chunks = split_text_with_metadata(load_pdf(str(pdf)))
        _, suspicious = create_vector_store(chunks, collection_name=pdf.stem)
        print(f"indexed {pdf.stem}: {len(chunks)} chunks, {suspicious} quarantined")
    return json.loads((EVAL / QUESTION_FILES[split]).read_text(encoding="utf-8"))


def _balanced(rows: list[dict], t: float) -> tuple[float, float, float]:
    """(in-scope kept, out-of-scope rejected, balanced accuracy) for one threshold."""
    ins = [r["top_score"] for r in rows if r["in_scope"]]
    outs = [r["top_score"] for r in rows if not r["in_scope"]]
    kept = sum(s >= t for s in ins) / len(ins)
    rejected = sum(s < t for s in outs) / len(outs)
    return kept, rejected, (kept + rejected) / 2


def _best_threshold(rows: list[dict]) -> float:
    """
    Threshold with the best balanced accuracy, searched on the OBSERVED scores:
    every midpoint between two consecutive distinct scores is a candidate, so no
    good cut is missed by a coarse 0.05 grid. Ties -> keep more answerable questions.
    """
    scores = sorted({r["top_score"] for r in rows})
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 1e-6]
    return max(candidates, key=lambda t: (_balanced(rows, t)[2], _balanced(rows, t)[0]))


def _recall_threshold(rows: list[dict], target_recall: float) -> float:
    """
    Highest threshold that still keeps at least `target_recall` of the answerable
    questions. The gate is the CHEAP first layer: it should remove obvious
    off-topic questions and almost never refuse a real one; the LLM's own
    "Information non trouvée" rule and the grounding check handle the hard cases.
    """
    scores = sorted({r["top_score"] for r in rows})
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    return max(t for t in candidates if _balanced(rows, t)[0] >= target_recall)


def _loo_rule(rows: list[dict], rule) -> tuple[float, float]:
    """Leave-one-out estimate of (answerable kept, unanswerable rejected) for a threshold rule."""
    kept, rejected = [], []
    for i, held in enumerate(rows):
        t = rule(rows[:i] + rows[i + 1:])
        (kept if held["in_scope"] else rejected).append(
            held["top_score"] >= t if held["in_scope"] else held["top_score"] < t)
    return sum(kept) / len(kept), sum(rejected) / len(rejected)


def _auc(rows: list[dict]) -> float:
    """
    ROC-AUC: probability that a random answerable question scores ABOVE a random
    unanswerable one (ties count half). Threshold-free, so it compares rerankers or
    chunk sizes fairly. 0.5 = chance, 1.0 = perfect separation.
    """
    ins = [r["top_score"] for r in rows if r["in_scope"]]
    outs = [r["top_score"] for r in rows if not r["in_scope"]]
    wins = sum((i > o) + 0.5 * (i == o) for i in ins for o in outs)
    return wins / (len(ins) * len(outs))


def _loo_balanced(rows: list[dict]) -> float:
    """
    Leave-one-out cross-validation: for each question, choose the threshold on the
    OTHER 31 questions, then judge the held-out one. This estimates how the gate
    does on a question it was not tuned on (no test-set leakage).
    """
    hits_in, hits_out = [], []
    for i, held in enumerate(rows):
        t = _best_threshold(rows[:i] + rows[i + 1:])
        if held["in_scope"]:
            hits_in.append(held["top_score"] >= t)
        else:
            hits_out.append(held["top_score"] < t)
    return (sum(hits_in) / len(hits_in) + sum(hits_out) / len(hits_out)) / 2


def calibrate(questions: list[dict], rewrite: bool = False, target_recall: float = 0.95) -> dict:
    """
    Best reranker score per question, then choose and validate the relevance threshold.

    rewrite=False scores the raw question (offline, no Groq call).
    rewrite=True scores the LLM-rewritten question, exactly as query_medrag() does at
    runtime, so the threshold is calibrated on the distribution it is used on.
    """
    from app.rag_pipeline import retrieve, rewrite_query

    config.RELEVANCE_THRESHOLD = 0.0  # keep everything: we only want the scores here
    rows = []
    for q in questions:
        query = rewrite_query(q["question"], strict=True) if rewrite else q["question"]  # fail loudly
        t0 = time.perf_counter()
        r = retrieve(q["question"], query, q["collection"])  # hybrid search + reranking (no LLM)
        rows.append({**q, "query_scored": query, "top_score": r.top_score or 0.0,
                     "retrieval_ms": round((time.perf_counter() - t0) * 1000)})

    best = _best_threshold(rows)
    kept, rejected, balanced = _balanced(rows, best)
    auc, loo = _auc(rows), _loo_balanced(rows)
    recall_t = _recall_threshold(rows, target_recall)  # the threshold MedRAG actually uses

    print(f"\nReranker: {config.RERANKER_MODEL} | chunk size {config.CHUNK_SIZE} | "
          f"queries: {'rewritten' if rewrite else 'raw'}")
    print(f"kept/refused below = with the recall-oriented threshold {recall_t:.4f}\n")
    for r in sorted(rows, key=lambda r: r["top_score"], reverse=True):  # every question, best score first
        mark = "IN " if r["in_scope"] else "OUT"
        side = "kept   " if r["top_score"] >= recall_t else "refused"
        print(f"  {r['top_score']:.4f}  {mark}  {side}  {r['collection'].replace('dossier_', ''):<17} {r['question']}")

    print(f"\nROC-AUC (threshold-free separation)      : {auc:.3f}")
    print(f"Best threshold (chosen on all 32)          : {best:.4f}  -> kept {kept:.0%} answerable, "
          f"rejected {rejected:.0%} unanswerable, balanced {balanced:.0%}  (optimistic: tuned on test)")
    print(f"Leave-one-out balanced accuracy (honest)   : {loo:.0%}")
    if loo < 0.6:
        print("WARNING: the gate is close to chance level; do NOT use this threshold.")

    # Latency of retrieval + reranking (both queries), the cost of a bigger reranker on this CPU
    lat = sorted(r["retrieval_ms"] for r in rows[1:])  # skip the first: it includes model loading
    print(f"Retrieval + reranking latency (CPU)      : p50 {statistics.median(lat):.0f} ms, "
          f"max {lat[-1]:.0f} ms")

    # The threshold MedRAG uses: recall-oriented (the gate is the first of two refusal layers)
    r_kept, r_rejected, _ = _balanced(rows, recall_t)
    loo_kept, loo_rejected = _loo_rule(rows, lambda rs: _recall_threshold(rs, target_recall))
    print(f"\nRecall-oriented threshold (keep >= {target_recall:.0%} answerable): {recall_t:.4f}")
    print(f"   on dev        : kept {r_kept:.0%} answerable, rejected {r_rejected:.0%} unanswerable")
    print(f"   leave-one-out : kept {loo_kept:.0%} answerable, rejected {loo_rejected:.0%} unanswerable")
    print(f"-> put RELEVANCE_THRESHOLD={recall_t:.4f} in .env; the LLM must refuse what the gate lets through.")

    return {"reranker": config.RERANKER_MODEL, "chunk_size": config.CHUNK_SIZE, "rewrite": rewrite,
            "scores": rows, "best": {"threshold": best, "in_scope_kept": kept, "out_of_scope_rejected": rejected,
                                     "balanced_accuracy": balanced},
            "roc_auc": auc, "loo_balanced_accuracy": loo,
            "retrieval_ms_p50": statistics.median(lat),
            "recall_threshold": {"target_recall": target_recall, "threshold": recall_t,
                                 "dev_kept": r_kept, "dev_rejected": r_rejected,
                                 "loo_kept": loo_kept, "loo_rejected": loo_rejected}}


def full_eval(questions: list[dict], sleep: float) -> dict:
    """Run every question through the complete pipeline with the real LLMs."""
    from app.rag_pipeline import query_medrag

    rows = []
    for n, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        r = query_medrag(q["question"], collection_name=q["collection"])
        latency = time.perf_counter() - t0
        pages = [s["page"] for s in r["sources"]]
        by_gate = bool(r["refused"] and r.get("retrieval") and r["retrieval"]["kept"] == 0)
        row = {**q, "refused": r["refused"], "refused_by": ("gate" if by_gate else "llm") if r["refused"] else None,
               "pages": pages, "latency_s": round(latency, 2), "answer": r["answer"],
               "grounding": r.get("grounding"), "top_score": (r.get("retrieval") or {}).get("top_score")}
        if q["in_scope"]:
            row["hit"] = q["expected_page"] in pages
            row["correct"] = (not r["refused"]) and all(_term_found(t, r["answer"]) for t in q["expected_terms"])
        rows.append(row)
        if q["in_scope"]:
            status = "ok" if row["correct"] else ("FALSE-REFUSAL" if r["refused"] else "WRONG")
        else:
            status = f"refused({row['refused_by']})" if r["refused"] else "ANSWERED!"  # must be refused
        print(f"[{n:02d}/{len(questions)}] {'IN ' if q['in_scope'] else 'OUT'} {status:14s} {latency:5.1f}s  "
              f"{q['question'][:55]}")
        time.sleep(sleep)  # stay under the Groq free-tier rate limit

    ins = [r for r in rows if r["in_scope"]]
    outs = [r for r in rows if not r["in_scope"]]
    answered = [r for r in rows if not r["refused"] and r["grounding"] and r["grounding"].get("checked")]
    lat = sorted(r["latency_s"] for r in rows)
    metrics = {
        "questions": len(rows), "in_scope": len(ins), "out_of_scope": len(outs),
        "retrieval_hit_at_5": round(sum(r["hit"] for r in ins) / len(ins), 3),
        "answer_accuracy": round(sum(r["correct"] for r in ins) / len(ins), 3),
        "false_refusal_rate": round(sum(r["refused"] for r in ins) / len(ins), 3),
        "out_of_scope_refusal_rate": round(sum(r["refused"] for r in outs) / len(outs), 3),
        "out_of_scope_refused_by_gate": round(sum(r["refused_by"] == "gate" for r in outs) / len(outs), 3),
        "out_of_scope_refused_by_llm": round(sum(r["refused_by"] == "llm" for r in outs) / len(outs), 3),
        "grounded_answer_rate": round(sum(bool(r["grounding"].get("grounded")) for r in answered) / len(answered), 3)
        if answered else None,
        "latency_p50_s": round(statistics.median(lat), 2),
        "latency_p95_s": round(lat[max(0, int(len(lat) * 0.95) - 1)], 2),
    }
    return {"settings": {"llm": config.MODEL_STRONG, "llm_fast": config.MODEL_FAST,
                         "reranker": config.RERANKER_MODEL, "threshold": config.RELEVANCE_THRESHOLD,
                         "embeddings": config.EMBEDDING_MODEL, "chunk_size": config.CHUNK_SIZE},
            "metrics": metrics, "rows": rows}


def write_markdown(result: dict, split: str) -> None:
    """A README-ready table of the metrics for one split."""
    m, s = result["metrics"], result["settings"]
    lines = [
        f"# MedRAG evaluation results ({split} set)", "",
        f"Corpus: 4 fictional multi-page records, {m['in_scope']} answerable questions (FR/EN/AR, with typos) "
        f"and {m['out_of_scope']} unanswerable ones"
        + (" — held-out questions, never used to tune the system." if split == "test" else
           " — the dev questions used for tuning (expect optimistic numbers).") , "",
        f"Settings: LLM `{s['llm']}`, rewriting/judge `{s['llm_fast']}`, reranker `{s['reranker']}`, "
        f"threshold {s['threshold']}, chunk size {s['chunk_size']}.", "",
        "| Metric | Value |", "| --- | --- |",
        f"| Retrieval hit@5 (right page among sources) | {m['retrieval_hit_at_5']:.0%} |",
        f"| Answer accuracy (expected facts in the answer) | {m['answer_accuracy']:.0%} |",
        f"| Refusal on unanswerable questions | {m['out_of_scope_refusal_rate']:.0%} "
        f"(gate {m['out_of_scope_refused_by_gate']:.0%}, LLM {m['out_of_scope_refused_by_llm']:.0%}) |",
        f"| False refusal on answerable questions | {m['false_refusal_rate']:.0%} |",
        (f"| Grounded answers (judge found no unsupported sentence) | {m['grounded_answer_rate']:.0%} |"
         if m["grounded_answer_rate"] is not None else "| Grounded answers | not measured |"),
        f"| Latency p50 / p95 | {m['latency_p50_s']} s / {m['latency_p95_s']} s |",
    ]
    (EVAL / f"RESULTS_{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibrate", action="store_true", help="only choose the relevance threshold")
    parser.add_argument("--rewrite", action="store_true", help="calibrate on LLM-rewritten queries (needs Groq)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="dev = tuning questions; test = held-out questions, run once at the end")
    parser.add_argument("--target-recall", type=float, default=0.95,
                        help="share of answerable dev questions the gate must keep (default 0.95)")
    parser.add_argument("--threshold", type=float, help="override RELEVANCE_THRESHOLD")
    parser.add_argument("--sleep", type=float, default=2.0, help="pause between questions (Groq rate limit)")
    args = parser.parse_args()

    # Absolute path: ChromaDB's engine on Windows failed to reopen its HNSW files
    # with the relative "eval/chroma_db" (bug hunt, step C.2b). Absolute = no ambiguity.
    config.CHROMA_DIR = str((EVAL / "chroma_db").resolve())  # keep evaluation data apart from real data
    # Start every run from an EMPTY index: reproducible, and no collection is ever
    # deleted/re-created inside one run (that triggers a ChromaDB 1.5.x bug, see vector_store.py).
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    if args.threshold is not None:
        config.RELEVANCE_THRESHOLD = args.threshold
    if args.calibrate and args.split == "test":
        parser.error("never calibrate on the test split: it would leak the test questions into tuning")
    questions = load_corpus(args.split)

    if args.calibrate:
        result = calibrate(questions, rewrite=args.rewrite, target_recall=args.target_recall)
        (EVAL / "calibration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    result = full_eval(questions, args.sleep)
    (EVAL / f"results_{args.split}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    write_markdown(result, args.split)
    print("\n" + json.dumps(result["metrics"], indent=2))
    print(f"\nSaved {EVAL}/results_{args.split}.json and {EVAL}/RESULTS_{args.split}.md")


if __name__ == "__main__":
    main()
