from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .data.clean_data import clean_dataframe
    from .data.load_data import load_twcs
except ImportError:
    from data.clean_data import clean_dataframe
    from data.load_data import load_twcs


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


def build_retrieval_corpus(rag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieval-ready corpus with explicit query/document columns.
    """
    expected = {"question_tweet_id", "answer_tweet_id", "question_text", "response_text"}
    missing = expected - set(rag_df.columns)
    if missing:
        raise ValueError(f"RAG pairs missing columns: {sorted(missing)}")

    out = rag_df.copy()
    out["query_text"] = out["question_text"]
    out["document_text"] = out["response_text"]
    out = out.dropna(subset=["query_text", "document_text"])
    out["query_text"] = out["query_text"].astype("string").str.strip()
    out["document_text"] = out["document_text"].astype("string").str.strip()
    out = out[(out["query_text"].str.len() > 0) & (out["document_text"].str.len() > 0)]
    return out.reset_index(drop=True)


def _random_split(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_size)
    n_val = int(n * val_size)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = max(0, n - n_train)

    train_df = shuffled.iloc[:n_train].reset_index(drop=True)
    val_df = shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = shuffled.iloc[n_train + n_val : n_train + n_val + n_test].reset_index(drop=True)
    return train_df, val_df, test_df


def build_ml_splits(
    full_df: pd.DataFrame,
    *,
    label_col: str = "priority_label",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build ML-ready train/val/test splits.
    - Uses simple stratification when ``label_col`` exists and has >=2 classes.
    - Falls back to random split otherwise.
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    if full_df.empty:
        return full_df.copy(), full_df.copy(), full_df.copy()

    if label_col not in full_df.columns:
        return _random_split(full_df, train_size, val_size, test_size, random_state)

    labeled = full_df.copy()
    labeled[label_col] = labeled[label_col].astype("string")
    valid_label_mask = labeled[label_col].notna() & (labeled[label_col].str.len() > 0)
    labeled = labeled.loc[valid_label_mask].reset_index(drop=True)
    if labeled.empty:
        return _random_split(full_df, train_size, val_size, test_size, random_state)

    class_counts = labeled[label_col].value_counts(dropna=True)
    if len(class_counts) < 2:
        return _random_split(labeled, train_size, val_size, test_size, random_state)

    grouped_parts: list[pd.DataFrame] = []
    for idx, (_, grp) in enumerate(labeled.groupby(label_col, sort=False)):
        part = grp.sample(frac=1.0, random_state=random_state + idx).reset_index(drop=True)
        grouped_parts.append(part)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for grp in grouped_parts:
        n = len(grp)
        n_train = int(n * train_size)
        n_val = int(n * val_size)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = max(0, n - n_train)

        train_parts.append(grp.iloc[:n_train])
        val_parts.append(grp.iloc[n_train : n_train + n_val])
        test_parts.append(grp.iloc[n_train + n_val : n_train + n_val + n_test])

    train_df = pd.concat(train_parts, ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    return train_df, val_df, test_df


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


def save_pipeline_outputs(
    retrieval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path | None = None,
    *,
    retrieval_name: str = "retrieval_corpus.csv",
    train_name: str = "ml_train.csv",
    val_name: str = "ml_val.csv",
    test_name: str = "ml_test.csv",
) -> tuple[Path, Path, Path, Path]:
    out = Path(out_dir) if out_dir is not None else default_processed_dir()
    out.mkdir(parents=True, exist_ok=True)

    retrieval_path = out / retrieval_name
    train_path = out / train_name
    val_path = out / val_name
    test_path = out / test_name

    retrieval_df.to_csv(retrieval_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    return retrieval_path, train_path, val_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare retrieval-ready corpus and ML-ready splits from TWCS."
    )
    parser.add_argument("--sample", action="store_true", help="Load data/sample/sample.csv")
    parser.add_argument("--nrows", type=int, default=None, help="Read only first N rows")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="priority_label",
        help="Label column for ML stratification (fallbacks to random if absent).",
    )
    parser.add_argument("--train-size", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    read_kw: dict = {}
    if args.nrows is not None:
        read_kw["nrows"] = args.nrows

    raw = load_twcs(use_sample=args.sample, **read_kw)
    cleaned = clean_dataframe(raw)
    rag_df, full_df = split_rag_and_prediction(cleaned)
    rag_path, full_path = save_split_outputs(rag_df, full_df, args.out_dir)
    retrieval_df = build_retrieval_corpus(rag_df)
    train_df, val_df, test_df = build_ml_splits(
        full_df,
        label_col=args.label_col,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )
    retrieval_path, train_path, val_path, test_path = save_pipeline_outputs(
        retrieval_df,
        train_df,
        val_df,
        test_df,
        args.out_dir,
    )

    print(f"RAG Q&A pairs: {len(rag_df):,} rows -> {rag_path}")
    print(f"Full prediction set: {len(full_df):,} rows -> {full_path}")
    print(f"Retrieval corpus: {len(retrieval_df):,} rows -> {retrieval_path}")
    print(f"ML train split: {len(train_df):,} rows -> {train_path}")
    print(f"ML val split: {len(val_df):,} rows -> {val_path}")
    print(f"ML test split: {len(test_df):,} rows -> {test_path}")


if __name__ == "__main__":
    main()
