# MedRAG evaluation results (dev set)

Corpus: 4 fictional multi-page records, 20 answerable questions (FR/EN/AR, with typos) and 12 unanswerable ones — the dev questions used for tuning (expect optimistic numbers).

Settings: LLM `openai/gpt-oss-120b`, rewriting/judge `openai/gpt-oss-20b`, reranker `BAAI/bge-reranker-v2-m3`, threshold 0.0034, chunk size 600.

| Metric | Value |
| --- | --- |
| Retrieval hit@5 (right page among sources) | 95% |
| Answer accuracy (expected facts in the answer) | 95% |
| Refusal on unanswerable questions | 100% (gate 100%, LLM 0%) |
| False refusal on answerable questions | 5% |
| Grounded answers (judge found no unsupported sentence) | 100% |
| Latency p50 / p95 | 3.73 s / 10.36 s |
