from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def append_json_log(path: str | Path, payload: dict) -> str:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(payload, ensure_ascii=False)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return str(log_path)
