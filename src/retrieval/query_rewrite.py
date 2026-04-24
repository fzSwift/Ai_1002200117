"""
Lightweight query expansion before embedding/BM25 retrieval.
Does not replace the user's question for the LLM prompt — only strengthens the search query.
"""

from __future__ import annotations

import re


def rewrite_query(raw: str) -> str:
    q = " ".join(raw.strip().split())
    if not q:
        return q

    ql = q.lower()
    tails: list[str] = []

    if re.search(r"\bnpp\b", ql) and "new patriotic" not in ql and "patriotic" not in ql:
        tails.append("New Patriotic Party")
    if re.search(r"\bndc\b", ql) and "national democratic" not in ql and "democratic congress" not in ql:
        tails.append("National Democratic Congress")
    if re.search(r"\bcpp\b", ql) and "convention people's party" not in ql:
        tails.append("CPP Convention People's Party")
    if re.search(r"\bpnc\b", ql) and "people's national convention" not in ql:
        tails.append("PNC People's National Convention")
    if "nothen" in ql:
        tails.append("Northern Region")
    if "accraa" in ql:
        tails.append("Greater Accra Region")

    if any(
        w in ql
        for w in (
            "election",
            "vote",
            "voted",
            "candidate",
            "region",
            "party",
            "constituency",
            "ballot",
        )
    ):
        if "2020" not in ql and "2016" not in ql:
            tails.append("2020 Ghana presidential election")

    if any(w in ql for w in ("budget", "fiscal", "revenue", "expenditure", "allocation")):
        if "2025" not in ql:
            tails.append("2025 Ghana budget")

    region_aliases = {
        "northern": "Northern Region",
        "savannah": "Savannah Region",
        "north east": "North East Region",
        "upper east": "Upper East Region",
        "upper west": "Upper West Region",
        "greater accra": "Greater Accra Region",
    }
    for alias, canonical in region_aliases.items():
        if alias in ql and canonical.lower() not in ql:
            tails.append(canonical)

    if not tails:
        return q

    return f"{q} {' '.join(tails)}"

