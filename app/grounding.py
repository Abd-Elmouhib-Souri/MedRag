"""
app/grounding.py
────────────────
Post-generation groundedness check (hallucination mitigation, layer 2).

Layer 1 is the relevance gate: the LLM only sees chunks that pass the
reranker threshold. Layer 2, here, verifies the produced answer: a small LLM
judge lists every sentence of the answer that the sources do NOT support.
The answer is returned with a grounding score and the unsupported sentences,
so the UI can warn the clinician instead of silently trusting the model.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from app import config
from app.llm_chain import NOT_FOUND_ANSWER, get_llm

log = logging.getLogger(__name__)

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict fact-checker for a medical assistant. You receive SOURCES and an ANSWER.\n"
        "Split the ANSWER into factual sentences. A sentence is supported only if the SOURCES state it "
        "explicitly (same drug, dose, value, date). Ignore source tags like [Page 2].\n"
        'Return ONLY JSON: {{"total": <number of factual sentences>, "unsupported": ["<sentence>", ...]}}'
    )),
    ("user", "SOURCES:\n{context}\n\nANSWER:\n{answer}"),
])


@dataclass
class GroundingResult:
    checked: bool                       # False if the check was skipped or failed
    grounded: bool | None = None        # True when no unsupported sentence was found
    score: float | None = None          # share of supported sentences, 0..1
    unsupported: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_json(raw: str) -> dict:
    """Extract the first JSON object from the judge output (models add text sometimes)."""
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("no JSON object in judge output")
    return json.loads(match.group())


def check_grounding(answer: str, context: str) -> GroundingResult:
    """Ask the fast LLM which sentences of the answer are not backed by the context."""
    if not config.GROUNDING_CHECK:
        return GroundingResult(checked=False, note="disabled")
    if answer.strip().startswith(NOT_FOUND_ANSWER):
        # A refusal makes no factual claim, so it is grounded by definition.
        return GroundingResult(checked=True, grounded=True, score=1.0, note="refusal")

    try:
        raw = (JUDGE_PROMPT | get_llm("fast")).invoke({"context": context, "answer": answer})
        data = _parse_json(getattr(raw, "content", str(raw)))
        unsupported = [s for s in data.get("unsupported", []) if isinstance(s, str) and s.strip()]
        total = max(int(data.get("total", 0) or 0), len(unsupported), 1)
        score = round(1 - len(unsupported) / total, 3)
        return GroundingResult(checked=True, grounded=not unsupported, score=score, unsupported=unsupported)
    except Exception as exc:  # never block the answer because the judge failed
        log.warning("Grounding check failed: %s", exc)
        return GroundingResult(checked=False, note=f"judge error: {exc}"[:200])
