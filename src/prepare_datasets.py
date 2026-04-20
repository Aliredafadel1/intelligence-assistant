from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .clean_data import clean_twcs
    from .load_data import load_twcs
except ImportError:
    from clean_data import clean_twcs
    from load_data import load_twcs


def default_processed_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "processed"


def first_response_tweet_id(raw: object) -> int | None:
    """
    TWCS sometimes lists multiple IDs in ``response_tweet_id`` (comma-separated).
    Use the first ID as the primary company reply for Q→A pairing.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    first = s.split(",")[0].strip()
    try:
        return int(float(first))
    except ValueError:
        return None


def build_rag_qa_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows suitable for RAG: **inbound (customer) tweets** that link to an **outbound
    (brand) tweet** present in the same table, via ``response_tweet_id``.
    """
    required = {"tweet_id", "inbound", "text", "response_tweet_id", "author_id", "created_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    cust = df.loc[df["inbound"] == True].copy()
    cust["_answer_tweet_id"] = cust["response_tweet_id"].map(first_response_tweet_id)
    cust = cust.loc[cust["_answer_tweet_id"].notna()]

    agent = df.loc[df["inbound"] == False, ["tweet_id", "text", "author_id", "created_at"]].copy()
    agent = agent.rename(
        columns={
            "tweet_id": "answer_tweet_id",
            "text": "response_text",
            "author_id": "brand_author_id",
            "created_at": "response_created_at",
        }
    )

    pairs = cust.merge(
        agent,
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

    cols = [
        "question_tweet_id",
        "answer_tweet_id",
        "question_text",
        "response_text",
        "customer_author_id",
        "brand_author_id",
        "question_created_at",
        "response_created_at",
    ]
    out = pairs.reindex(columns=[c for c in cols if c in pairs.columns])
    out = out.drop_duplicates(subset=["question_tweet_id", "answer_tweet_id"])
    return out.reset_index(drop=True)


def full_data_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """All tweets after cleaning — use for labels + ML / LLM priority training."""
    return df.copy()


def split_rag_and_prediction(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_rag_qa_pairs(df), full_data_for_prediction(df)


def save_split_outputs(
    rag_df: pd.DataFrame,
    full_df: pd.DataFrame,
    out_dir: str | Path | None = None,
    *,
    rag_name: str = "rag_qa_pairs.csv",
    full_name: str = "full_tweets_for_prediction.csv",
) -> tuple[Path, Path]:
    out = Path(out_dir) if out_dir is not None else default_processed_dir()
    out.mkdir(parents=True, exist_ok=True)
    rag_path = out / rag_name
    full_path = out / full_name
    rag_df.to_csv(rag_path, index=False)
    full_df.to_csv(full_path, index=False)
    return rag_path, full_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split TWCS into RAG Q&A pairs and full tweet table for prediction."
    )
    parser.add_argument("--sample", action="store_true", help="Load data/sample/sample.csv")
    parser.add_argument("--nrows", type=int, default=None, help="Read only first N rows")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/processed)",
    )
    args = parser.parse_args()

    read_kw: dict = {}
    if args.nrows is not None:
        read_kw["nrows"] = args.nrows

    raw = load_twcs(use_sample=args.sample, **read_kw)
    cleaned = clean_twcs(raw)
    rag_df, full_df = split_rag_and_prediction(cleaned)
    rag_path, full_path = save_split_outputs(rag_df, full_df, args.out_dir)

    print(f"RAG Q&A pairs: {len(rag_df):,} rows -> {rag_path}")
    print(f"Full prediction set: {len(full_df):,} rows -> {full_path}")


if __name__ == "__main__":
    main()
