from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .clean_data import clean_dataframe
except ImportError:
    from clean_data import clean_dataframe


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def first_response_tweet_id(raw: object) -> int | None:
    """
    TWCS can contain multiple comma-separated IDs in response_tweet_id.
    Keep only the first valid numeric ID for pairing.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None

    first = text.split(",")[0].strip()
    try:
        return int(float(first))
    except ValueError:
        return None


def clean_full_dataset(
    raw: pd.DataFrame,
    *,
    inbound_only: bool = False,
    drop_duplicate_clean_text: bool = False,
    min_clean_text_length: int = 1,
) -> pd.DataFrame:
    return clean_dataframe(
        raw,
        inbound_only=inbound_only,
        drop_duplicate_clean_text=drop_duplicate_clean_text,
        min_clean_text_length=min_clean_text_length,
    )


def build_rag_qa_pairs(df: pd.DataFrame) -> pd.DataFrame:
    required = {"tweet_id", "inbound", "text", "response_tweet_id", "author_id", "created_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    customer = df.loc[df["inbound"] == True].copy()
    customer["_answer_tweet_id"] = customer["response_tweet_id"].map(first_response_tweet_id)
    customer = customer.loc[customer["_answer_tweet_id"].notna()]

    brand = df.loc[df["inbound"] == False, ["tweet_id", "text", "author_id", "created_at"]].copy()
    brand = brand.rename(
        columns={
            "tweet_id": "answer_tweet_id",
            "text": "response_text",
            "author_id": "brand_author_id",
            "created_at": "response_created_at",
        }
    )

    pairs = customer.merge(
        brand,
        left_on="_answer_tweet_id",
        right_on="answer_tweet_id",
        how="inner",
    )
    pairs = pairs.rename(
        columns={
            "tweet_id": "question_tweet_id",
            "text": "question_text",
            "author_id": "customer_author_id",
            "created_at": "question_created_at",
        }
    )

    selected_columns = [
        "question_tweet_id",
        "answer_tweet_id",
        "question_text",
        "response_text",
        "clean_text",
        "normalized_text",
        "word_count",
        "customer_author_id",
        "brand_author_id",
        "question_created_at",
        "response_created_at",
    ]
    out = pairs.reindex(columns=[col for col in selected_columns if col in pairs.columns])
    out = out.drop_duplicates(subset=["question_tweet_id", "answer_tweet_id"], keep="first")
    return out.reset_index(drop=True)
