from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ComposedAnswer:
    text: str
    confidence: str
    citations: list[str]
    intent: str


def _intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ("how many", "how much", "total", "number", "percent", "percentage", "votes")):
        return "numeric"
    if any(k in q for k in ("compare", "difference", "higher", "lower", "versus", "vs")):
        return "comparison"
    if any(k in q for k in ("what is", "define", "meaning", "who is")):
        return "definition"
    if any(k in q for k in ("policy", "budget says", "impact", "strategy", "objective")):
        return "policy_summary"
    return "general"


def _confidence(chunks: list[dict]) -> str:
    if not chunks:
        return "low"
    scores = [float(c.get("final_score", 0.0)) for c in chunks[:3]]
    top = max(scores)
    spread = top - (scores[1] if len(scores) > 1 else 0.0)
    if top >= 0.75 and spread >= 0.2:
        return "high"
    if top >= 0.45:
        return "medium"
    return "low"


def _first_clean_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    cleaned = re.sub(r"\|.*?\|", " ", cleaned)
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    for part in parts:
        if len(part.strip()) >= 15:
            return part.strip()
    return cleaned[:220].strip()


def compose_structured_answer(query: str, chunks: list[dict], *, mode: str = "detailed") -> ComposedAnswer:
    intent = _intent(query)
    top_chunks = chunks[:3]
    citations = [str(c.get("chunk_id")) for c in top_chunks if c.get("chunk_id")]
    confidence = _confidence(top_chunks)

    evidence_sentences: list[str] = []
    for c in top_chunks:
        sent = _first_clean_sentence(str(c.get("text", "")))
        if sent:
            cid = c.get("chunk_id", "chunk")
            evidence_sentences.append(f"{sent} [{cid}]")

    if not evidence_sentences:
        return ComposedAnswer(
            text="I do not have enough information from the provided documents.",
            confidence="low",
            citations=[],
            intent=intent,
        )

    direct = evidence_sentences[0]
    if intent == "numeric":
        direct_prefix = "Direct answer:"
    elif intent == "comparison":
        direct_prefix = "Comparison result:"
    elif intent == "definition":
        direct_prefix = "Definition:"
    elif intent == "policy_summary":
        direct_prefix = "Policy summary:"
    else:
        direct_prefix = "Answer:"

    if mode == "concise":
        text = f"{direct_prefix} {direct}\nConfidence: {confidence}."
    elif mode == "examiner":
        evidence = " ".join(evidence_sentences[:2])
        text = (
            f"{direct_prefix} {direct}\n"
            f"Evidence: {evidence}\n"
            f"Confidence note: {confidence} confidence based on retrieval score concentration."
        )
    else:
        evidence = " ".join(evidence_sentences[:2])
        text = (
            f"{direct_prefix} {direct}\n"
            f"Evidence: {evidence}\n"
            f"Confidence: {confidence}."
        )

    return ComposedAnswer(text=text.strip(), confidence=confidence, citations=citations, intent=intent)
