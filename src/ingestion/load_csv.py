from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_election_csv(path: str | Path) -> pd.DataFrame:
    """Load the Ghana election CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)
