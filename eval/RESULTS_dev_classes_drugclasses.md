# MedRAG evaluation results (dev_classes set)

Corpus: 4 fictional multi-page records, 11 answerable questions (FR/EN/AR) and 6 unanswerable ones — NEW dev questions naming a drug class, written before the drug-class fix.

Settings: LLM `openai/gpt-oss-120b`, rewriting/judge `openai/gpt-oss-20b`, reranker `BAAI/bge-reranker-v2-m3`, threshold 0.0034, chunk size 600, drug-class expansion on.

Machine (local part: embeddings, BM25, reranker): 13th Gen Intel(R) Core(TM) i5-13450HX, 16 logical cores, 39.7 GB RAM, CPU only, Windows 11. LLM calls run on Groq's servers, so their time includes the network.

| Metric | Value |
| --- | --- |
| Retrieval hit@5 (right page among sources) | 100% |
| Answer accuracy (expected facts in the answer) | 91% |
| Refusal on unanswerable questions | 100% (gate 67%, LLM 33%) |
| False refusal on answerable questions | 9% |
| Grounded answers (judge found no unsupported sentence) | 100% |
| Latency p50 / p95, end to end | 6.54 s / 8.44 s |
| - of which search (hybrid retrieval + reranker), p50 / p95 | 3.03 s / 4.39 s |
| - of which LLM (rewrite + answer + judge), p50 / p95 | 2.72 s / 4.83 s |
