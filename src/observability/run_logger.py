from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_default_log_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "logs" / "runs.jsonl"


class RunLogger:
    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = (log_path or build_default_log_path()).resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def new_run_id(self) -> str:
        return str(uuid.uuid4())

    def append(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=True, default=str)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_run(
        self,
        *,
        run_id: str,
        payload: dict[str, Any],
        status: str,
        error: str | None = None,
    ) -> None:
        event = {
            "event_type": "run_summary",
            "run_id": run_id,
            "timestamp": utc_now_iso(),
            "status": status,
            "error": error,
            "payload": payload,
        }
        self.append(event)
