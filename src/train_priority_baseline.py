from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .preprocess_common import project_root
except ImportError:
    from preprocess_common import project_root


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Train a baseline ML model on weak priority labels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "processed" / "full_tweets_with_priority_labels.csv",
        help="Input labeled CSV path.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from sklearn.compose import ColumnTransformer  # pyright: ignore[reportMissingImports]
        from sklearn.feature_extraction.text import TfidfVectorizer  # pyright: ignore[reportMissingImports]
        from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingImports]
        from sklearn.metrics import classification_report  # pyright: ignore[reportMissingImports]
        from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingImports]
        from sklearn.pipeline import Pipeline  # pyright: ignore[reportMissingImports]
        from sklearn.preprocessing import OneHotEncoder, StandardScaler  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for training. Install with: pip install scikit-learn"
        ) from exc

    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "priority_label" not in df.columns:
        raise ValueError("Missing 'priority_label'. Run src/label_priority.py first.")

    df = df.dropna(subset=["priority_label"]).copy()
    if df.empty:
        raise ValueError("No rows available after dropping missing priority labels.")

    if "normalized_text" not in df.columns:
        fallback_text = (
            df["text"] if "text" in df.columns else pd.Series("", index=df.index, dtype="string")
        )
        df["normalized_text"] = fallback_text.astype("string").fillna("")

    numeric_features = [
        c
        for c in ["word_count", "text_length", "question_count", "exclamation_count", "priority_score"]
        if c in df.columns
    ]
    categorical_features = [c for c in ["author_id", "inbound"] if c in df.columns]

    X = df
    y = df["priority_label"].astype("string")
    if y.nunique() < 2:
        raise ValueError(
            "Need at least two priority classes to train. "
            "Adjust weak labeling rules or verify input data."
        )

    transformers: list[tuple[str, object, str | list[str]]] = [
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "normalized_text")
    ]
    if numeric_features:
        transformers.append(("num", StandardScaler(with_mean=False), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    class_counts = y.value_counts()
    stratify_target = y if (class_counts.min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_target,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(classification_report(y_test, preds, digits=4))


if __name__ == "__main__":
    main()
