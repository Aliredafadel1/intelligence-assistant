from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .preprocess_common import project_root
except ImportError:
    from preprocess_common import project_root


HIGH_PRIORITY_RE = r"\b(?:urgent|asap|immediately|now|help|fix|down|can't|cannot|failed|error|issue)\b"
MEDIUM_PRIORITY_RE = r"\b(?:please|problem|question|support|waiting|slow|refund|delay)\b"


def _safe_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("Int64")


def _safe_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype("boolean")


def compute_priority_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = out.get("normalized_text", pd.Series("", index=out.index, dtype="string")).fillna("")

    out["is_inbound"] = _safe_bool(out.get("inbound", pd.Series(True, index=out.index)))
    out["question_count"] = _safe_int(out.get("question_count", pd.Series(0, index=out.index)))
    out["exclamation_count"] = _safe_int(out.get("exclamation_count", pd.Series(0, index=out.index)))
    out["has_all_caps_word"] = _safe_bool(out.get("has_all_caps_word", pd.Series(False, index=out.index)))
    out["has_urgent_keyword"] = _safe_bool(out.get("has_urgent_keyword", pd.Series(False, index=out.index)))
    out["has_negative_word"] = _safe_bool(out.get("has_negative_word", pd.Series(False, index=out.index)))

    out["rule_high_keyword"] = text.str.contains(HIGH_PRIORITY_RE, regex=True).astype("boolean")
    out["rule_medium_keyword"] = text.str.contains(MEDIUM_PRIORITY_RE, regex=True).astype("boolean")
    out["rule_multi_signal"] = (
        (out["question_count"] >= 2)
        | (out["exclamation_count"] >= 2)
        | out["has_all_caps_word"]
    ).astype("boolean")

    # Weighted weak supervision score (tweakable).
    score = (
        out["is_inbound"].astype("Int64") * 1
        + out["has_urgent_keyword"].astype("Int64") * 3
        + out["rule_high_keyword"].astype("Int64") * 3
        + out["has_negative_word"].astype("Int64") * 2
        + out["rule_multi_signal"].astype("Int64") * 1
        + out["rule_medium_keyword"].astype("Int64") * 1
    )
    out["priority_score"] = score.astype("Int64")

    out["priority_label"] = "low"
    out.loc[out["priority_score"] >= 6, "priority_label"] = "high"
    out.loc[(out["priority_score"] >= 3) & (out["priority_score"] < 6), "priority_label"] = "medium"

    # Prefer non-customer messages as low unless score is clearly high.
    out.loc[(out["is_inbound"] == False) & (out["priority_score"] < 6), "priority_label"] = "low"

    return out


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Create weak supervision priority labels for ML."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "processed" / "full_tweets_for_prediction_clean.csv",
        help="Input cleaned ML CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "full_tweets_with_priority_labels.csv",
        help="Output CSV path with weak priority labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    labeled = compute_priority_scores(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)

    print(f"Input rows                : {len(df)}")
    print(f"Saved labeled dataset     : {output_path}")
    print("Priority distribution:")
    print(labeled["priority_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
