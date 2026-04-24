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


def _confidence_with_penalty(base_confidence: str, contradiction_count: int) -> str:
    if contradiction_count <= 0:
        return base_confidence
    if base_confidence == "high":
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


def _parse_election_fact(text: str) -> dict[str, str | int | None] | None:
    if "Election record." not in text:
        return None
    m_year = re.search(r"Year:\s*(\d+)", text)
    m_region = re.search(r"New Region:\s*([^.]+)\.\s*Code:", text)
    m_candidate = re.search(r"Candidate:\s*([^.]+)\.\s*Party:", text)
    m_party = re.search(r"Party:\s*([^.]+)\.\s*Votes:", text)
    m_votes = re.search(r"Votes:\s*([\d,]+)", text)
    m_pct = re.search(r"Vote Percentage:\s*([0-9.]+%)", text)
    if not (m_year and m_region and m_candidate and m_party and m_votes):
        return None
    try:
        votes = int(m_votes.group(1).replace(",", ""))
    except ValueError:
        return None
    return {
        "year": m_year.group(1).strip(),
        "region": m_region.group(1).strip().replace("\ufffd", " "),
        "candidate": m_candidate.group(1).strip(),
        "party": m_party.group(1).strip(),
        "votes": votes,
        "pct": (m_pct.group(1).strip() if m_pct else None),
    }


def _regions_from_query(query: str) -> list[str]:
    q = query.lower()
    region_pattern = re.compile(
        r"(north east|northeast|savannah|northern|upper east|upper west|greater accra|central|volta|ashanti)\s+region"
    )
    out: list[str] = []
    for m in region_pattern.findall(q):
        canonical = m.replace("northeast", "north east").title() + " Region"
        if canonical not in out:
            out.append(canonical)
    return out


def _format_fact_sentence(fact: dict[str, str | int | None], cid: str) -> str:
    votes = int(fact["votes"])  # type: ignore[arg-type]
    pct = f" ({fact['pct']})" if fact.get("pct") else ""
    return (
        f"{fact['candidate']} ({fact['party']}) in {fact['region']} recorded "
        f"{votes:,} votes{pct} in {fact['year']}. [{cid}]"
    )


def _comparison_direct_answer(query: str, chunks: list[dict]) -> str | None:
    facts: list[tuple[dict[str, str | int | None], str]] = []
    for c in chunks:
        fact = _parse_election_fact(str(c.get("text", "")))
        if fact:
            facts.append((fact, str(c.get("chunk_id", "chunk"))))
    if len(facts) < 2:
        return None

    regions = _regions_from_query(query)
    if len(regions) >= 2:
        selected: list[tuple[dict[str, str | int | None], str]] = []
        for region in regions[:2]:
            matches = [f for f in facts if str(f[0].get("region", "")).lower() == region.lower()]
            if matches:
                best = max(matches, key=lambda x: int(x[0]["votes"]))  # type: ignore[arg-type]
                selected.append(best)
        if len(selected) == 2:
            a, b = selected[0], selected[1]
            av = int(a[0]["votes"])  # type: ignore[arg-type]
            bv = int(b[0]["votes"])  # type: ignore[arg-type]
            if av >= bv:
                lead, trail = a, b
            else:
                lead, trail = b, a
            diff = abs(av - bv)
            return (
                f"{lead[0]['region']} has the higher top vote count: {int(lead[0]['votes']):,} "
                f"for {lead[0]['candidate']} ({lead[0]['party']}), versus {int(trail[0]['votes']):,} "
                f"in {trail[0]['region']} for {trail[0]['candidate']} ({trail[0]['party']}). "
                f"Difference: {diff:,}. [{lead[1]}] [{trail[1]}]"
            )

    facts.sort(key=lambda x: int(x[0]["votes"]), reverse=True)  # type: ignore[arg-type]
    top, nxt = facts[0], facts[1]
    return (
        f"Top vote count in retrieved comparison records is {int(top[0]['votes']):,} in {top[0]['region']} "
        f"({top[0]['candidate']}, {top[0]['party']}), followed by {int(nxt[0]['votes']):,} in {nxt[0]['region']} "
        f"({nxt[0]['candidate']}, {nxt[0]['party']}). [{top[1]}] [{nxt[1]}]"
    )


def _numeric_answer_from_weighted_votes(query: str, chunks: list[dict]) -> tuple[str | None, int]:
    q = query.lower()
    facts: list[tuple[dict[str, str | int | None], str, float]] = []
    for c in chunks:
        fact = _parse_election_fact(str(c.get("text", "")))
        if not fact:
            continue
        score = float(c.get("final_score", 0.0))
        facts.append((fact, str(c.get("chunk_id", "chunk")), score))
    if not facts:
        return None, 0

    key_votes: dict[tuple[str, str, str, str], list[tuple[int, str, float]]] = {}
    for fact, cid, score in facts:
        key = (
            str(fact.get("party", "")),
            str(fact.get("region", "")),
            str(fact.get("candidate", "")),
            str(fact.get("year", "")),
        )
        key_votes.setdefault(key, []).append((int(fact.get("votes", 0) or 0), cid, score))

    weighted: list[tuple[tuple[str, str, str, str], float, int, str]] = []
    contradictions = 0
    for key, rows in key_votes.items():
        votes_only = {v for v, _, _ in rows}
        if len(votes_only) > 1:
            contradictions += 1
        total_w = sum(max(0.05, s) for _, _, s in rows)
        weighted_vote = sum(v * max(0.05, s) for v, _, s in rows) / total_w
        best_row = max(rows, key=lambda r: r[2])
        weighted.append((key, weighted_vote, best_row[0], best_row[1]))

    weighted.sort(key=lambda x: x[1], reverse=True)
    if not weighted:
        return None, contradictions

    top = weighted[0]
    party, region, candidate, year = top[0]
    direct = (
        f"{party} in {region} has about {int(round(top[1])):,} weighted votes "
        f"(best supporting row: {top[2]:,} for {candidate} in {year}). [{top[3]}]"
    )

    if any(k in q for k in ("compare", "difference", "higher", "lower", "versus", "vs")) and len(weighted) >= 2:
        nxt = weighted[1]
        nparty, nregion, ncandidate, nyear = nxt[0]
        diff = abs(int(round(top[1])) - int(round(nxt[1])))
        direct = (
            f"{region} ({party}) is higher at about {int(round(top[1])):,} weighted votes "
            f"vs {nregion} ({nparty}) at about {int(round(nxt[1])):,}. "
            f"Difference: {diff:,}. [{top[3]}] [{nxt[3]}]"
        )
        _ = (candidate, year, ncandidate, nyear)

    return direct, contradictions


def compose_structured_answer(query: str, chunks: list[dict], *, mode: str = "detailed") -> ComposedAnswer:
    intent = _intent(query)
    top_chunks = chunks[:8]
    citations = [str(c.get("chunk_id")) for c in top_chunks if c.get("chunk_id")]
    confidence = _confidence(top_chunks)

    evidence_sentences: list[str] = []
    for c in top_chunks:
        cid = c.get("chunk_id", "chunk")
        fact = _parse_election_fact(str(c.get("text", "")))
        if fact:
            evidence_sentences.append(_format_fact_sentence(fact, str(cid)))
            continue
        sent = _first_clean_sentence(str(c.get("text", "")))
        if sent:
            evidence_sentences.append(f"{sent} [{cid}]")

    if not evidence_sentences:
        return ComposedAnswer(
            text="I do not have enough information from the provided documents.",
            confidence="low",
            citations=[],
            intent=intent,
        )

    direct = evidence_sentences[0]
    contradiction_count = 0
    if intent == "comparison":
        weighted, contradiction_count = _numeric_answer_from_weighted_votes(query, top_chunks)
        comparison_answer = weighted or _comparison_direct_answer(query, top_chunks)
        if comparison_answer:
            direct = comparison_answer
    elif intent == "numeric":
        weighted, contradiction_count = _numeric_answer_from_weighted_votes(query, top_chunks)
        if weighted:
            direct = weighted
        else:
            fact = _parse_election_fact(str(top_chunks[0].get("text", ""))) if top_chunks else None
            if fact and top_chunks:
                direct = _format_fact_sentence(fact, str(top_chunks[0].get("chunk_id", "chunk")))
    elif intent in {"general", "definition"}:
        fact = _parse_election_fact(str(top_chunks[0].get("text", ""))) if top_chunks else None
        if fact and top_chunks:
            direct = _format_fact_sentence(fact, str(top_chunks[0].get("chunk_id", "chunk")))
    confidence = _confidence_with_penalty(confidence, contradiction_count)
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
