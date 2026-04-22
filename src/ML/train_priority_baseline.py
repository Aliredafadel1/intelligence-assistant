from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

try:
    from ..preprocess_common import project_root
except ImportError:
    from preprocess_common import project_root


def parse_args() -> argparse.Namespace:
    root = project_root()
    artifacts_dir = root / "data" / "artifacts"
    parser = argparse.ArgumentParser(
        description="Train and compare multiple ML classifiers on weak priority labels."
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
    parser.add_argument(
        "--model-out",
        type=Path,
        default=artifacts_dir / "priority_baseline.joblib",
        help="Output path for the selected best model.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=artifacts_dir / "priority_model_comparison_report.json",
        help="Output path for model comparison report JSON.",
    )
    parser.add_argument(
        "--include-priority-score",
        action="store_true",
        help="Include priority_score as a feature (disabled by default to reduce weak-label leakage).",
    )
    return parser.parse_args()


def _metric_value(report: dict, key: str, metric: str) -> float:
    value = report.get(key, {}).get(metric, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _evaluate_model(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    from sklearn.metrics import accuracy_score, classification_report  # pyright: ignore[reportMissingImports]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report_dict = classification_report(y_test, preds, digits=4, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_precision": _metric_value(report_dict, "macro avg", "precision"),
        "macro_recall": _metric_value(report_dict, "macro avg", "recall"),
        "macro_f1": _metric_value(report_dict, "macro avg", "f1-score"),
        "weighted_f1": _metric_value(report_dict, "weighted avg", "f1-score"),
        "classification_report": report_dict,
    }


def main() -> None:
    try:
        from sklearn.compose import ColumnTransformer  # pyright: ignore[reportMissingImports]
        from sklearn.ensemble import RandomForestClassifier  # pyright: ignore[reportMissingImports]
        from sklearn.feature_extraction.text import TfidfVectorizer  # pyright: ignore[reportMissingImports]
        from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingImports]
        from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingImports]
        from sklearn.pipeline import Pipeline  # pyright: ignore[reportMissingImports]
        from sklearn.preprocessing import OneHotEncoder, StandardScaler  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for training. Install with: pip install scikit-learn"
        ) from exc
    try:
        from xgboost import XGBClassifier  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for model comparison. Install with: pip install xgboost"
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

    numeric_candidates = ["word_count", "text_length", "question_count", "exclamation_count"]
    if args.include_priority_score:
        numeric_candidates.append("priority_score")
    numeric_features = [c for c in numeric_candidates if c in df.columns]
    categorical_features = [c for c in ["author_id", "inbound"] if c in df.columns]

    X = df
    y = df["priority_label"].astype("string")
    if y.nunique() < 2:
        raise ValueError(
            "Need at least two priority classes to train. "
            "Adjust weak labeling rules or verify input data."
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

    transformers: list[tuple[str, object, str | list[str]]] = [
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "normalized_text")
    ]
    if numeric_features:
        transformers.append(("num", StandardScaler(with_mean=False), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    label_values = sorted(y.astype("string").dropna().unique().tolist())
    label_to_int = {label: idx for idx, label in enumerate(label_values)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    y_train_num = y_train.astype("string").map(label_to_int).astype(int)
    y_test_num = y_test.astype("string").map(label_to_int).astype(int)

    models: dict[str, object] = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=args.random_state,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        num_class=len(label_values),
                        random_state=args.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    results: dict[str, dict] = {}
    trained_models: dict[str, object] = {}
    for model_name, model in models.items():
        if model_name == "xgboost":
            metrics = _evaluate_model(model, X_train, y_train_num, X_test, y_test_num)
            report = metrics.get("classification_report", {})
            mapped_report: dict[str, object] = {}
            for key, value in report.items():
                mapped_key = str(key)
                if mapped_key.isdigit():
                    label_idx = int(mapped_key)
                    mapped_report[int_to_label.get(label_idx, mapped_key)] = value
                    continue
                mapped_report[mapped_key] = value
            metrics["classification_report"] = mapped_report
        else:
            metrics = _evaluate_model(model, X_train, y_train, X_test, y_test)
        results[model_name] = metrics
        trained_models[model_name] = model

    best_model_name = max(results.items(), key=lambda item: item[1]["macro_f1"])[0]
    best_model = trained_models[best_model_name]

    model_out = args.model_out.resolve()
    report_out = args.report_out.resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_out)
    per_model_paths: dict[str, str] = {}
    for model_name, model in trained_models.items():
        setattr(model, "priority_label_order", label_values)
        model_path = model_out.parent / f"priority_{model_name}.joblib"
        joblib.dump(model, model_path)
        per_model_paths[model_name] = str(model_path)

    leaderboard = [
        {
            "model": model_name,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "macro_precision": round(metrics["macro_precision"], 4),
            "macro_recall": round(metrics["macro_recall"], 4),
        }
        for model_name, metrics in sorted(
            results.items(),
            key=lambda item: item[1]["macro_f1"],
            reverse=True,
        )
    ]

    report_payload = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "labels": label_values,
        "feature_config": {
            "text_feature": "normalized_text",
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "include_priority_score": bool(args.include_priority_score),
        },
        "best_model": best_model_name,
        "best_model_path": str(model_out),
        "per_model_paths": per_model_paths,
        "leaderboard": leaderboard,
        "model_metrics": {
            model_name: {
                "accuracy": round(metrics["accuracy"], 4),
                "macro_precision": round(metrics["macro_precision"], 4),
                "macro_recall": round(metrics["macro_recall"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "weighted_f1": round(metrics["weighted_f1"], 4),
                "classification_report": metrics["classification_report"],
            }
            for model_name, metrics in results.items()
        },
    }
    report_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("=== Model Comparison (sorted by macro_f1) ===")
    for row in leaderboard:
        print(
            f"{row['model']:>20s} | accuracy={row['accuracy']:.4f} | "
            f"macro_f1={row['macro_f1']:.4f} | weighted_f1={row['weighted_f1']:.4f}"
        )
    print(f"Selected best model: {best_model_name}")
    print(f"Saved best model: {model_out}")
    print(f"Saved report: {report_out}")


if __name__ == "__main__":
    main()
