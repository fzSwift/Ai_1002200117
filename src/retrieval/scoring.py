from __future__ import annotations

import re
from typing import Iterable


ELECTION_TERMS = {
    "election", "votes", "vote", "party", "candidate", "region", "constituency",
    "npp", "ndc", "cpp", "pnc", "ind"
}
BUDGET_TERMS = {
    "budget", "allocation", "revenue", "expenditure", "ministry", "debt",
    "inflation", "gdp", "fiscal", "agriculture", "education", "health"
}

PARTY_TERMS = {"npp", "ndc", "cpp", "pnc", "gum", "ind"}
REGION_TERMS = {
    "northern region",
    "savannah region",
    "north east region",
    "upper east region",
    "upper west region",
    "greater accra region",
    "volta region",
    "ashanti region",
}


def classify_query(query: str) -> str:
    q = query.lower()
    election_hits = sum(1 for term in ELECTION_TERMS if term in q)
    budget_hits = sum(1 for term in BUDGET_TERMS if term in q)

    if election_hits > budget_hits:
        return "election"
    if budget_hits > election_hits:
        return "budget"
    return "mixed"


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def keyword_overlap(query: str, chunk_keywords: Iterable[str]) -> float:
    query_terms = set(re.findall(r"[A-Za-z][A-Za-z\-']{2,}", query.lower()))
    chunk_terms = set(word.lower() for word in chunk_keywords)
    if not query_terms:
        return 0.0
    return len(query_terms & chunk_terms) / len(query_terms)


def compute_domain_bonus(query: str, query_type: str, chunk: dict) -> float:
    bonus = 0.0
    source = chunk.get("source", "")
    text = chunk.get("text", "")
    if query_type == "election" and source == "election_csv":
        bonus += 0.5
    elif query_type == "budget" and source == "budget_pdf":
        bonus += 0.5
    elif query_type == "mixed":
        bonus += 0.2

    bonus += 0.3 * keyword_overlap(query, chunk.get("keywords", []))

    query_nums = numeric_tokens(query)
    text_nums = numeric_tokens(text)
    if query_nums and query_nums & text_nums:
        bonus += 0.2

    if "2025" in query and chunk.get("year") == "2025":
        bonus += 0.1
    return min(bonus, 1.0)


def extract_metadata_constraints(query: str) -> dict[str, str]:
    q = query.lower()
    constraints: dict[str, str] = {}

    if "election" in q:
        constraints["source"] = "election_csv"
    elif "budget" in q:
        constraints["source"] = "budget_pdf"

    y = re.search(r"\b(19|20)\d{2}\b", q)
    if y:
        constraints["year"] = y.group(0)

    for party in PARTY_TERMS:
        if party in q:
            constraints["party"] = party.upper()
            break

    for region in REGION_TERMS:
        if region in q:
            constraints["region"] = region.title()
            break
    if "nothen region" in q:
        constraints["region"] = "Northern Region"

    return constraints


def metadata_match_score(chunk: dict, constraints: dict[str, str]) -> float:
    if not constraints:
        return 0.0
    score = 0.0
    meta = chunk.get("metadata", {}) or {}
    text = str(chunk.get("text", "")).lower()

    if constraints.get("source") and chunk.get("source") == constraints["source"]:
        score += 0.4
    if constraints.get("year") and str(chunk.get("year", "")) == constraints["year"]:
        score += 0.2
    if constraints.get("party"):
        p = constraints["party"].lower()
        if str(meta.get("party", "")).lower() == p or f"party: {p}" in text:
            score += 0.25
    if constraints.get("region"):
        r = constraints["region"].lower()
        if r in str(meta.get("new_region", "")).lower() or r in text:
            score += 0.25
    return min(score, 1.0)
