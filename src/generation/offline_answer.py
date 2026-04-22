"""
Extractive, sentence-style answers for offline mode (no LLM API).
Uses structured text from election_csv chunks (see chunking.row_to_text).
"""

from __future__ import annotations

import re
from typing import Any


_PARTY_ALIASES: dict[str, tuple[str, ...]] = {
    "NPP": ("npp", "new patriotic party"),
    "NDC": ("ndc", "national democratic congress"),
    "GUM": ("gum", "ghana union movement"),
    "CPP": ("cpp", "convention people's party", "convention peoples party"),
    "PNC": ("pnc", "people's national convention", "peoples national convention"),
}

_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "Northern Region": ("northern region", "northern", "nothen region", "nothen"),
    "Upper East Region": ("upper east region", "upper east"),
    "Upper West Region": ("upper west region", "upper west"),
    "Greater Accra Region": ("greater accra region", "accra"),
}


def _parse_election_record(text: str) -> dict[str, Any] | None:
    if "Election record." not in text:
        return None
    m_year = re.search(r"Year:\s*(\d+)", text)
    m_or = re.search(r"Old Region:\s*([^.]+)\.\s*New Region:", text)
    m_nr = re.search(r"New Region:\s*([^.]+)\.\s*Code:", text)
    m_cand = re.search(r"Candidate:\s*([^.]+)\.\s*Party:", text)
    m_party = re.search(r"Party:\s*([^.]+)\.\s*Votes:", text)
    m_votes = re.search(r"Votes:\s*([\d,]+)", text)
    m_pct = re.search(r"Vote Percentage:\s*([0-9.]+%)", text)
    if not (m_year and m_nr and m_cand and m_party and m_votes):
        return None
    raw_votes = m_votes.group(1).replace(",", "")
    try:
        votes_int = int(raw_votes)
    except ValueError:
        return None
    def _clean(s: str) -> str:
        return " ".join(s.replace("\xa0", " ").replace("\ufffd", " ").split())

    return {
        "year": m_year.group(1),
        "old_region": (_clean(m_or.group(1)) if m_or else None),
        "region": _clean(m_nr.group(1)),
        "candidate": _clean(m_cand.group(1)),
        "party": _clean(m_party.group(1)),
        "votes": votes_int,
        "pct": (m_pct.group(1).strip() if m_pct else None),
    }


def _quantity_intent(query: str) -> bool:
    q = query.lower()
    return any(
        p in q
        for p in (
            "how many",
            "how much",
            "number of",
            "total vote",
            "vote total",
            "people voted",
            "person voted",
            "turnout",
            "received",
            "got ",
            "votes ",
            "vote ",
        )
    )


def _national_aggregate_intent(query: str) -> bool:
    q = query.lower()
    return ("total" in q or "overall" in q or "national" in q or "ghana" in q or "country" in q) and (
        "vote" in q or "people" in q or "turnout" in q or "ballot" in q
    )


def _format_bullet(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for r in records:
        pct = f" ({r['pct']} of votes in that row)" if r.get("pct") else ""
        lines.append(
            f"- **{r['candidate']}** ({r['party']}) in **{r['region']}** ({r['year']}): "
            f"**{r['votes']:,}** votes{pct}."
        )
    return "\n".join(lines)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    text = re.sub(r"\|.*?\|", " ", text)
    return text.strip()


def extract_key_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        s = sentence.strip()
        if len(s) >= 20:
            return s
    return (text or "").strip()


def _extract_numeric_facts_from_chunk(text: str) -> dict[str, Any] | None:
    cleaned = clean_text(text)
    if not cleaned:
        return None

    vote_match = re.search(r"\bVotes:\s*([\d,]+)\b", cleaned, flags=re.IGNORECASE)
    pct_match = re.search(r"\bVote Percentage:\s*([0-9]+(?:\.[0-9]+)?%)\b", cleaned, flags=re.IGNORECASE)

    candidate_match = re.search(r"\bCandidate:\s*([^.]+)", cleaned, flags=re.IGNORECASE)
    region_match = re.search(r"\bNew Region:\s*([^.]+)", cleaned, flags=re.IGNORECASE)
    year_match = re.search(r"\bYear:\s*(\d{4})\b", cleaned, flags=re.IGNORECASE)

    if not vote_match and not pct_match:
        # Generic number fallback for budget chunks
        generic = re.findall(r"\b\d[\d,]*(?:\.\d+)?%?\b", cleaned)
        if not generic:
            return None
        first = generic[0]
        return {
            "candidate": None,
            "region": None,
            "year": None,
            "votes": None,
            "pct": first if first.endswith("%") else None,
            "raw_number": first,
            "sentence": extract_key_sentence(cleaned),
        }

    votes_int: int | None = None
    if vote_match:
        try:
            votes_int = int(vote_match.group(1).replace(",", ""))
        except ValueError:
            votes_int = None

    return {
        "candidate": candidate_match.group(1).strip() if candidate_match else None,
        "region": region_match.group(1).strip() if region_match else None,
        "year": year_match.group(1).strip() if year_match else None,
        "votes": votes_int,
        "pct": pct_match.group(1).strip() if pct_match else None,
        "raw_number": vote_match.group(1) if vote_match else None,
        "sentence": extract_key_sentence(cleaned),
    }


def _comparative_numeric_summary(query: str, chunks: list[dict]) -> str | None:
    numeric_facts: list[dict[str, Any]] = []
    for chunk in chunks[:5]:
        fact = _extract_numeric_facts_from_chunk(chunk.get("text", ""))
        if fact:
            numeric_facts.append(fact)

    if not numeric_facts:
        return None

    vote_facts = [f for f in numeric_facts if isinstance(f.get("votes"), int)]
    vote_facts.sort(key=lambda x: int(x["votes"]), reverse=True)

    if len(vote_facts) >= 2:
        top = vote_facts[0]
        nxt = vote_facts[1]

        top_subject = "the highest vote count"
        if top.get("candidate") and top.get("region"):
            top_subject = f"{top['candidate']} in {top['region']}"
        elif top.get("region"):
            top_subject = f"the {top['region']} figure"

        next_subject = "the next reported figure"
        if nxt.get("candidate") and nxt.get("region"):
            next_subject = f"{nxt['candidate']} in {nxt['region']}"
        elif nxt.get("region"):
            next_subject = f"the {nxt['region']} figure"

        tail = ""
        if any(word in query.lower() for word in ["total", "overall", "national"]):
            tail = " The retrieved rows do not provide one single consolidated nationwide total."

        return (
            f"The retrieved records show that {top_subject} has {top['votes']:,} votes, followed by "
            f"{next_subject} with {nxt['votes']:,}.{tail}"
        ).strip()

    if len(vote_facts) == 1:
        only = vote_facts[0]
        who = "the matched entry"
        if only.get("candidate") and only.get("region"):
            who = f"{only['candidate']} in {only['region']}"
        return f"The retrieved records show {who} with {only['votes']:,} votes."

    pct_facts = [f for f in numeric_facts if f.get("pct")]
    if pct_facts:
        top_pct = pct_facts[0]
        subj = "the retrieved passage"
        if top_pct.get("candidate"):
            subj = top_pct["candidate"]
        return f"The retrieved records report {subj} at {top_pct['pct']}."

    return None


def _detect_party_and_region(query: str) -> tuple[str | None, str | None]:
    q = query.lower()
    party: str | None = None
    region: str | None = None

    for canonical, aliases in _PARTY_ALIASES.items():
        if any(alias in q for alias in aliases):
            party = canonical
            break

    for canonical, aliases in _REGION_ALIASES.items():
        if any(alias in q for alias in aliases):
            region = canonical
            break

    return party, region


def _direct_party_region_vote_answer(query: str, chunks: list[dict]) -> str | None:
    party, region = _detect_party_and_region(query)
    if not party and not region:
        return None

    records: list[dict[str, Any]] = []
    for chunk in chunks:
        parsed = _parse_election_record(chunk.get("text", ""))
        if parsed:
            records.append(parsed)
    if not records:
        return None

    filtered = records
    if party:
        filtered = [r for r in filtered if str(r.get("party", "")).strip().upper() == party]
    if region:
        region_l = region.lower()
        filtered = [
            r
            for r in filtered
            if region_l
            in {
                str(r.get("region", "")).strip().lower(),
                str(r.get("old_region", "")).strip().lower(),
            }
        ]
    if not filtered:
        return None

    best = max(filtered, key=lambda r: int(r.get("votes", 0)))
    scope_bits: list[str] = []
    if party:
        scope_bits.append(party)
    if region:
        scope_bits.append(region)
    scope = " in ".join(scope_bits) if scope_bits else "the matched records"

    response = f"The retrieved records show {scope} with {best['votes']:,} votes"
    if best.get("candidate"):
        response += f" for {best['candidate']}"
    if best.get("year"):
        response += f" in {best['year']}"
    if best.get("pct"):
        response += f" ({best['pct']})"
    return response + "."


def generate_offline_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I do not have enough information from the provided documents."

    top_chunks = chunks[:3]
    q = query.lower()
    key_points: list[str] = []

    for chunk in top_chunks:
        text = clean_text(chunk.get("text", ""))
        if not text:
            continue
        sentence = extract_key_sentence(text)
        if len(sentence) > 300:
            sentence = sentence[:300].rsplit(" ", 1)[0] + "..."
        key_points.append(sentence)

    if not key_points:
        return "I do not have enough information from the provided documents."

    direct_party_region = _direct_party_region_vote_answer(query, top_chunks)
    if direct_party_region:
        return direct_party_region

    numeric_words = ["how many", "how much", "total", "number", "amount", "votes", "percentage", "percent"]
    if any(word in q for word in numeric_words):
        comparative = _comparative_numeric_summary(query, top_chunks)
        if comparative:
            return comparative

    answer = " ".join(key_points)
    answer = answer[0].upper() + answer[1:] if answer else answer

    if any(word in q for word in ["how many", "how much", "total", "number", "amount"]):
        return (
            "The retrieved data indicates "
            + answer
            + " The records do not provide a single consolidated nationwide total."
        )
    if any(word in q for word in ["who", "which", "winner", "party"]):
        return "The retrieved records indicate that " + answer
    return "The retrieved evidence indicates that " + answer


def try_sentence_answer(query: str, chunks: list[dict]) -> str | None:
    """
    If we can derive a direct, conversational answer from election rows, return markdown.
    Otherwise return None and the caller should fall back to raw excerpts.
    """
    election_chunks = [c for c in chunks if c.get("source") == "election_csv"]
    if not election_chunks:
        return None

    records: list[dict[str, Any]] = []
    for c in election_chunks:
        parsed = _parse_election_record(c.get("text") or "")
        if parsed:
            records.append(parsed)

    if not records:
        return None

    if not _quantity_intent(query):
        return None

    if _national_aggregate_intent(query):
        return (
            "National turnout is not stored as a single row in the retrieved election records. "
            "The dataset reports votes per candidate and per region, so summing rows would over-count. "
            + generate_offline_answer(query, chunks)
        )

    if len(records) == 1:
        r = records[0]
        pct = f" Their share in that row is {r['pct']}." if r.get("pct") else ""
        return (
            f"In the {r['year']} election, {r['candidate']} ({r['party']}) in {r['region']} "
            f"received {r['votes']:,} votes.{pct}"
        )

    return generate_offline_answer(query, chunks)


def compose_offline_fallback(query: str, chunks: list[dict]) -> str | None:
    """
    When try_sentence_answer returns None, still return a real QA-style answer:
    synthesized bullets from election rows or short budget excerpts — not raw chunk dumps.
    """
    _ = query
    election_chunks = [c for c in chunks if c.get("source") == "election_csv"]
    pdf_chunks = [
        c
        for c in chunks
        if "pdf" in str(c.get("source", "")).lower() or str(c.get("source", "")).lower() == "budget_pdf"
    ]

    if election_chunks:
        return generate_offline_answer(query, election_chunks)

    if pdf_chunks:
        return generate_offline_answer(query, pdf_chunks)

    return None
