from __future__ import annotations

import re
from typing import Any

import pandas as pd

HIGH_PRIORITY_RE = r"\b(?:urgent|asap|immediately|now|help|fix|down|can't|cannot|failed|error|issue)\b"
MEDIUM_PRIORITY_RE = r"\b(?:please|problem|question|support|waiting|slow|refund|delay)\b"
NEGATIVE_WORDS = {"down", "error", "failed", "broken", "cannot", "can't", "issue", "problem"}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def has_all_caps_word(text: str) -> bool:
    for token in re.findall(r"\b[A-Za-z]{2,}\b", text):
        if token.isupper():
            return True
    return False


def build_feature_row(ticket_text: str, *, author_id: str, inbound: bool = True) -> pd.DataFrame:
    normalized = normalize_text(ticket_text)
    tokens = re.findall(r"\b\w+\b", ticket_text)
    lower_tokens = [t.lower() for t in tokens]

    has_urgent_keyword = bool(re.search(HIGH_PRIORITY_RE, normalized))
    has_negative_word = any(token in NEGATIVE_WORDS for token in lower_tokens)
    rule_high_keyword = has_urgent_keyword
    rule_medium_keyword = bool(re.search(MEDIUM_PRIORITY_RE, normalized))
    question_count = ticket_text.count("?")
    exclamation_count = ticket_text.count("!")
    rule_multi_signal = question_count >= 2 or exclamation_count >= 2 or has_all_caps_word(ticket_text)
    priority_score = (
        int(bool(inbound)) * 1
        + int(has_urgent_keyword) * 3
        + int(rule_high_keyword) * 3
        + int(has_negative_word) * 2
        + int(rule_multi_signal) * 1
        + int(rule_medium_keyword) * 1
    )

    return pd.DataFrame(
        [
            {
                "normalized_text": normalized,
                "word_count": len(tokens),
                "text_length": len(ticket_text),
                "question_count": question_count,
                "exclamation_count": exclamation_count,
                "priority_score": priority_score,
                "author_id": author_id,
                "inbound": bool(inbound),
            }
        ]
    )


def ensure_required_columns(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    preprocessor = model.named_steps.get("preprocess")
    if preprocessor is None:
        return frame

    required: list[str] = []
    for _, _, cols in preprocessor.transformers:
        if isinstance(cols, str) and cols in {"drop", "passthrough"}:
            continue
        if isinstance(cols, str):
            required.append(cols)
        else:
            required.extend(cols)

    for col in required:
        if col in frame.columns:
            continue
        if col == "author_id":
            frame[col] = "unknown_customer"
        elif col == "inbound":
            frame[col] = True
        elif col == "normalized_text":
            frame[col] = ""
        else:
            frame[col] = 0
    return frame
