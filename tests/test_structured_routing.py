from __future__ import annotations

import pandas as pd

from src.data.structured_store import StructuredElectionStore
from src.routing.query_router import is_structured_numeric_query, parse_structured_constraints


def test_query_router_parses_party_region_typo() -> None:
    q = "What is the NPP vote in the nothen region in 2020?"
    assert is_structured_numeric_query(q)
    c = parse_structured_constraints(q)
    assert c["party"] == "NPP"
    assert c["region"] == "Northern Region"
    assert c["year"] == "2020"


def test_structured_store_query_votes(tmp_path) -> None:
    df = pd.DataFrame(
        [
            {"Year": 2020, "Old Region": "Northern Region", "New Region": "North East Region", "Candidate": "A", "Party": "NPP", "Votes": 12345, "Votes(%)": "55%"},
            {"Year": 2020, "Old Region": "Northern Region", "New Region": "Savannah Region", "Candidate": "B", "Party": "NPP", "Votes": 10000, "Votes(%)": "45%"},
        ]
    )
    store = StructuredElectionStore(tmp_path / "election.sqlite3")
    store.build_from_df(df)
    rows = store.query_votes(party="NPP", region="Northern Region", year="2020")
    assert rows
    assert int(rows[0]["Votes"]) == 12345
