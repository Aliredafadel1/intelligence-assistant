from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ML baseline performance: text-only vs text+engineered features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/full_tweets_with_priority_labels.csv"),
        help="Input labeled CSV.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/artifacts/feature_ablation_report.json"),
        help="Output report JSON path.",
    )
    return parser.parse_args()


def _metric_value(report: dict, key: str, metric: str) -> float:
    value = report.get(key, {}).get(metric, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _evaluate_model(model: object, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, classification_report  # pyright: ignore[reportMissingImports]

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    report_dict = classification_report(y_test, preds, digits=4, output_dict=True)
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "macro_f1": round(_metric_value(report_dict, "macro avg", "f1-score"), 4),
        "weighted_f1": round(_metric_value(report_dict, "weighted avg", "f1-score"), 4),
    }


def main() -> None:
    from sklearn.compose import ColumnTransformer  # pyright: ignore[reportMissingImports]
    from sklearn.feature_extraction.text import TfidfVectorizer  # pyright: ignore[reportMissingImports]
    from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingImports]
    from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingImports]
    from sklearn.pipeline import Pipeline  # pyright: ignore[reportMissingImports]
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # pyright: ignore[reportMissingImports]

    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "priority_label" not in df.columns:
        raise ValueError("Missing 'priority_label'. Run labeling first.")
    if "normalized_text" not in df.columns:
        text_col = df["text"] if "text" in df.columns else pd.Series("", index=df.index, dtype="string")
        df["normalized_text"] = text_col.astype("string").fillna("")
    df = df.dropna(subset=["priority_label"]).copy()
    if df.empty:
        raise ValueError("No rows left after dropping null priority labels.")

    y = df["priority_label"].astype("string")
    x_train, x_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    text_only = ColumnTransformer(
        transformers=[("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "normalized_text")],
        remainder="drop",
    )
    engineered_cols = [c for c in ["word_count", "text_length", "question_count", "exclamation_count"] if c in df.columns]
    categorical_cols = [c for c in ["author_id", "inbound"] if c in df.columns]
    text_plus_engineered_steps: list[tuple[str, object, str | list[str]]] = [
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "normalized_text")
    ]
    if engineered_cols:
        text_plus_engineered_steps.append(("num", StandardScaler(with_mean=False), engineered_cols))
    if categorical_cols:
        text_plus_engineered_steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    text_plus_engineered = ColumnTransformer(transformers=text_plus_engineered_steps, remainder="drop")

    text_only_model = Pipeline(
        steps=[
            ("preprocess", text_only),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    engineered_model = Pipeline(
        steps=[
            ("preprocess", text_plus_engineered),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    text_only_metrics = _evaluate_model(text_only_model, x_train, y_train, x_test, y_test)
    engineered_metrics = _evaluate_model(engineered_model, x_train, y_train, x_test, y_test)

    report = {
        "input_rows": int(len(df)),
        "split": {"train_rows": int(len(x_train)), "test_rows": int(len(x_test))},
        "setups": {
            "text_only": {
                "features": ["normalized_text (TF-IDF)"],
                "metrics": text_only_metrics,
            },
            "text_plus_engineered": {
                "features": ["normalized_text (TF-IDF)", *engineered_cols, *categorical_cols],
                "metrics": engineered_metrics,
            },
        },
        "delta_engineered_minus_text_only": {
            "accuracy": round(engineered_metrics["accuracy"] - text_only_metrics["accuracy"], 4),
            "macro_f1": round(engineered_metrics["macro_f1"] - text_only_metrics["macro_f1"], 4),
            "weighted_f1": round(engineered_metrics["weighted_f1"] - text_only_metrics["weighted_f1"], 4),
        },
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved feature ablation report: {output_path}")


if __name__ == "__main__":
    main()
