from __future__ import annotations

import json

ML_TO_PRIORITY = {
    "high": "P1",
    "medium": "P2",
    "low": "P3",
}


def map_ml_label_to_priority(label: object) -> str:
    key = str(label or "").strip().lower()
    return ML_TO_PRIORITY.get(key, "P3")


def normalize_priority(value: object) -> str:
    val = str(value or "").strip().upper()
    return val if val in {"P1", "P2", "P3", "P4"} else "P3"


def clamp_confidence(value: object, default: float = 0.5) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    score = max(0.0, min(1.0, score))
    return round(score, 2)


def extract_json_object(text: str) -> dict | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
