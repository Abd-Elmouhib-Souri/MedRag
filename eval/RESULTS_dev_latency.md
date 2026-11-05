# MedRAG evaluation results (dev set)

Corpus: 4 fictional multi-page records, 20 answerable questions (FR/EN/AR, with typos) and 12 unanswerable ones — the dev questions used for tuning (expect optimistic numbers).

Settings: LLM `openai/gpt-oss-120b`, rewriting/judge `openai/gpt-oss-20b`, reranker `BAAI/bge-reranker-v2-m3`, threshold 0.0034, chunk size 600, drug-class expansion off.

Machine (local part: embeddings, BM25, reranker): 13th Gen Intel(R) Core(TM) i5-13450HX, 16 logical cores, 39.7 GB RAM, CPU only, Windows 11. LLM calls run on Groq's servers, so their time includes the network.

| Metric | Value |
| --- | --- |
| Retrieval hit@5 (right page among sources) | 95% |
| Answer accuracy (expected facts in the answer) | 90% |
| Refusal on unanswerable questions | 100% (gate 100%, LLM 0%) |
| False refusal on answerable questions | 5% |
| Grounded answers (judge found no unsupported sentence) | 100% |
| Latency p50 / p95, end to end | 4.25 s / 9.13 s |
| - of which search (hybrid retrieval + reranker), p50 / p95 | 1.25 s / 2.62 s |
| - of which LLM (rewrite + answer + judge), p50 / p95 | 2.75 s / 7.75 s |
