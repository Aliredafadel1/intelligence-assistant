from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s!?]")
MULTI_PUNCT_RE = re.compile(r"([!?])\1+")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_text(value: object) -> str:
    """
    Light cleaning:
    - keep natural text mostly intact
    - remove URLs
    - remove @mentions
    - normalize whitespace

    Good for EDA / RAG / human-readable analysis.
    """
    text = "" if pd.isna(value) else str(value)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def normalize_text(value: object) -> str:
    """
    Stronger normalization for ML / keyword-based features:
    - lowercase
    - remove URLs and @mentions
    - keep letters, numbers, spaces, !, ?
    - collapse repeated whitespace
    - reduce repeated punctuation like !!!!! -> !
    """
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = MULTI_PUNCT_RE.sub(r"\1", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def to_nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def to_nullable_bool(series: pd.Series) -> pd.Series:
    """
    Convert common boolean-like values safely to pandas nullable boolean.
    """
    lowered = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    return lowered.map(mapping).astype("boolean")


def clean_dataframe(
    df: pd.DataFrame,
    *,
    inbound_only: bool = False,
    drop_duplicate_clean_text: bool = False,
    min_clean_text_length: int = 1,
) -> pd.DataFrame:
    out = df.copy()

    # Normalize expected columns when present
    if "tweet_id" in out.columns:
        out["tweet_id"] = to_nullable_int(out["tweet_id"])

    if "response_tweet_id" in out.columns:
        out["response_tweet_id"] = to_nullable_int(out["response_tweet_id"])

    if "in_response_to_tweet_id" in out.columns:
        out["in_response_to_tweet_id"] = to_nullable_int(out["in_response_to_tweet_id"])

    if "inbound" in out.columns:
        out["inbound"] = to_nullable_bool(out["inbound"])

    if "created_at" in out.columns:
        out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)

    if "author_id" in out.columns:
        out["author_id"] = out["author_id"].astype("string").str.strip()

    if "text" in out.columns:
        out["text"] = out["text"].astype("string").str.strip()

        # Light-cleaned text for EDA / RAG
        out["clean_text"] = out["text"].map(clean_text).astype("string")

        # Stronger normalized text for ML / keyword logic
        out["normalized_text"] = out["text"].map(normalize_text).astype("string")

        # Length features
        out["text_length"] = out["clean_text"].str.len().astype("Int64")
        out["normalized_text_length"] = out["normalized_text"].str.len().astype("Int64")
        out["word_count"] = out["normalized_text"].str.split().str.len().astype("Int64")

        # Simple useful feature flags for later modeling
        raw_text = out["text"].fillna("")
        out["exclamation_count"] = raw_text.str.count(r"!").astype("Int64")
        out["question_count"] = raw_text.str.count(r"\?").astype("Int64")
        out["has_all_caps_word"] = raw_text.str.contains(r"\b[A-Z]{3,}\b", regex=True).astype("boolean")
        out["has_url"] = raw_text.str.contains(URL_RE).astype("boolean")
        out["has_mention"] = raw_text.str.contains(MENTION_RE).astype("boolean")
        out["has_urgent_keyword"] = out["normalized_text"].str.contains(
            r"\b(?:urgent|asap|immediately|now|help|please|fix)\b",
            regex=True,
        ).astype("boolean")
        out["has_negative_word"] = out["normalized_text"].str.contains(
            r"\b(?:bad|worst|terrible|angry|disappointed|frustrated)\b",
            regex=True,
        ).astype("boolean")
        out["starts_with_question"] = out["normalized_text"].str.startswith(
            ("why", "how", "what", "when", "where", "is", "are", "can", "do")
        ).astype("boolean")

       

    # Optional: keep only customer messages
    if inbound_only and "inbound" in out.columns:
        out = out[out["inbound"] == True]

    # Drop empty content after cleaning
    if "clean_text" in out.columns:
        out = out[out["clean_text"].fillna("").str.len() >= min_clean_text_length]

    # First, drop duplicate tweet IDs if present
    if "tweet_id" in out.columns:
        out = out.drop_duplicates(subset=["tweet_id"], keep="first")
    else:
        out = out.drop_duplicates(keep="first")

    # Optional: also drop duplicate clean_text rows
    if drop_duplicate_clean_text and "clean_text" in out.columns:
        out = out.drop_duplicates(subset=["clean_text"], keep="first")

    return out.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Clean tweet dataset for EDA / ML / RAG.")

    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "processed" / "full_tweets_for_prediction.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "full_tweets_for_prediction_clean.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--inbound-only",
        action="store_true",
        help="Keep only inbound tweets if the column exists.",
    )
    parser.add_argument(
        "--drop-duplicate-clean-text",
        action="store_true",
        help="Drop rows with duplicate clean_text after standard duplicate removal.",
    )
    parser.add_argument(
        "--min-clean-text-length",
        type=int,
        default=1,
        help="Minimum length required for clean_text to keep a row.",
    )

    return parser.parse_args()


def print_summary(before: pd.DataFrame, after: pd.DataFrame) -> None:
    print(f"Input rows                : {len(before)}")
    print(f"Clean rows                : {len(after)}")
    print(f"Removed rows              : {len(before) - len(after)}")

    if "text" in before.columns:
        empty_raw = before["text"].astype("string").fillna("").str.strip().eq("").sum()
        print(f"Empty raw text rows       : {empty_raw}")

    if "clean_text" in after.columns:
        empty_clean = after["clean_text"].fillna("").str.strip().eq("").sum()
        print(f"Empty clean text rows     : {empty_clean}")

        duplicated_clean = after["clean_text"].duplicated().sum()
        print(f"Remaining duplicate texts : {duplicated_clean}")

        print(f"Avg clean text length     : {after['clean_text'].str.len().mean():.2f}")

    if "word_count" in after.columns:
        print(f"Avg word count            : {after['word_count'].mean():.2f}")

    if "inbound" in after.columns:
        inbound_counts = after["inbound"].value_counts(dropna=False).to_dict()
        print(f"Inbound distribution      : {inbound_counts}")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    cleaned = clean_dataframe(
        df,
        inbound_only=args.inbound_only,
        drop_duplicate_clean_text=args.drop_duplicate_clean_text,
        min_clean_text_length=args.min_clean_text_length,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print_summary(df, cleaned)
    print(f"Saved file                : {output_path}")


if __name__ == "__main__":
    main()