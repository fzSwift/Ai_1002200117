from __future__ import annotations

import re

import pandas as pd


def clean_election_df(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in cleaned.columns]

    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.replace("\u00a0", " ", regex=False)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    if "Votes" in cleaned.columns:
        cleaned["Votes"] = (
            cleaned["Votes"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("nan", None)
        )
        cleaned["Votes"] = pd.to_numeric(cleaned["Votes"], errors="coerce")

    return cleaned.dropna(how="all")
