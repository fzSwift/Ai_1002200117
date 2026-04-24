from __future__ import annotations

import re


def is_structured_numeric_query(query: str) -> bool:
    q = query.lower()
    numeric_signal = any(w in q for w in ("vote", "votes", "how many", "number", "total", "percent", "percentage"))
    entity_signal = any(w in q for w in ("npp", "ndc", "party", "region", "candidate", "year"))
    return numeric_signal and entity_signal


def parse_structured_constraints(query: str) -> dict[str, str]:
    q = query.lower()
    constraints: dict[str, str] = {}

    party_alias = {
        "npp": "NPP",
        "ndc": "NDC",
        "cpp": "CPP",
        "pnc": "PNC",
        "gum": "GUM",
    }
    for token, canonical in party_alias.items():
        if token in q:
            constraints["party"] = canonical
            break

    if "nothen region" in q:
        constraints["region"] = "Northern Region"
    else:
        region_match = re.search(r"(northern|savannah|north east|upper east|upper west|volta|ashanti|central|oti)\s+region", q)
        if region_match:
            constraints["region"] = f"{region_match.group(1).title()} Region"

    year_match = re.search(r"\b(19|20)\d{2}\b", q)
    if year_match:
        constraints["year"] = year_match.group(0)
    return constraints
