# MedRAG evaluation results (test set)

Corpus: 4 fictional multi-page records, 11 answerable questions (FR/EN/AR, with typos) and 8 unanswerable ones — held-out questions, never used to tune the system.

Settings: LLM `openai/gpt-oss-120b`, rewriting/judge `openai/gpt-oss-20b`, reranker `BAAI/bge-reranker-v2-m3`, threshold 0.0034, chunk size 600.

| Metric | Value |
| --- | --- |
| Retrieval hit@5 (right page among sources) | 91% |
| Answer accuracy (expected facts in the answer) | 91% |
| Refusal on unanswerable questions | 100% (gate 100%, LLM 0%) |
| False refusal on answerable questions | 9% |
| Grounded answers (judge found no unsupported sentence) | 100% |
| Latency p50 / p95 | 5.41 s / 20.01 s |
