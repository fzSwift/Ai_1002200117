from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


class StructuredElectionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def build_from_df(self, df: pd.DataFrame) -> None:
        with sqlite3.connect(self.db_path) as conn:
            rows = df.copy()
            rows["row_index"] = range(len(rows))
            rows.to_sql("election_rows", conn, if_exists="replace", index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_party ON election_rows(Party)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_new_region ON election_rows([New Region])")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_old_region ON election_rows([Old Region])")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON election_rows(Year)")
            conn.commit()

    def query_votes(
        self,
        *,
        party: str | None = None,
        region: str | None = None,
        year: str | None = None,
        limit: int = 3,
    ) -> list[dict]:
        sql = (
            "SELECT row_index, Year, [Old Region] as old_region, [New Region] as new_region, "
            "Candidate, Party, Votes, [Votes(%)] as votes_pct FROM election_rows WHERE 1=1"
        )
        params: list[object] = []
        if party:
            sql += " AND UPPER(Party)=UPPER(?)"
            params.append(party)
        if region:
            sql += " AND (UPPER([New Region])=UPPER(?) OR UPPER([Old Region])=UPPER(?))"
            params.extend([region, region])
        if year:
            sql += " AND CAST(Year AS TEXT)=?"
            params.append(year)
        sql += " ORDER BY CAST(Votes AS INTEGER) DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
