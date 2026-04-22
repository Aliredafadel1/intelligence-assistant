from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from ..preprocess_common import project_root
except ImportError:
    from preprocess_common import project_root


def parse_args() -> argparse.Namespace:
    root = project_root()
    artifacts_dir = root / "data" / "artifacts"
    parser = argparse.ArgumentParser(
        description="Compare MAE/RMSE/R2 across trained priority classifiers."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "processed" / "full_tweets_with_priority_labels.csv",
        help="Input labeled CSV path used for evaluation split.",
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
        "--report-out",
        type=Path,
        default=artifacts_dir / "priority_model_regression_metrics.json",
        help="Output path for MAE/RMSE/R2 comparison JSON.",
    )
    parser.add_argument("--json", action="store_true", help="Print report JSON to stdout.")
    return parser.parse_args()


def _to_label_string(value: object, int_to_label: dict[int, str]) -> str:
    text = str(value)
    if text.isdigit():
        idx = int(text)
        if idx in int_to_label:
            return int_to_label[idx]
    return text


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # pyright: ignore[reportMissingImports]
        from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for metrics comparison. Install with: pip install scikit-learn"
        ) from exc

    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "priority_label" not in df.columns:
        raise ValueError("Missing 'priority_label' in input dataset.")
    df = df.dropna(subset=["priority_label"]).copy()
    if df.empty:
        raise ValueError("No rows available after dropping missing priority labels.")

    y = df["priority_label"].astype("string")
    if y.nunique() < 2:
        raise ValueError("Need at least two priority classes for comparison.")

    class_counts = y.value_counts()
    stratify_target = y if (class_counts.min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_target,
    )
    _ = X_train, y_train  # split parity with training; models are pre-trained

    labels_sorted = sorted(y.astype("string").dropna().unique().tolist())
    label_to_int = {label: idx for idx, label in enumerate(labels_sorted)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    y_true_num = np.array([label_to_int[str(v)] for v in y_test.astype("string")], dtype=float)

    root = project_root()
    artifacts_dir = root / "data" / "artifacts"
    model_paths = {
        "logistic_regression": artifacts_dir / "priority_logistic_regression.joblib",
        "random_forest": artifacts_dir / "priority_random_forest.joblib",
        "xgboost": artifacts_dir / "priority_xgboost.joblib",
    }

    metrics_rows: list[dict[str, object]] = []
    for model_name, path in model_paths.items():
        model_path = path.resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for {model_name}: {model_path}")
        model = joblib.load(model_path)
        preds_raw = model.predict(X_test)
        preds_label = [_to_label_string(v, int_to_label) for v in preds_raw]
        y_pred_num = np.array([label_to_int.get(lbl, 0) for lbl in preds_label], dtype=float)

        mae = _safe_float(mean_absolute_error(y_true_num, y_pred_num))
        rmse = _safe_float(np.sqrt(mean_squared_error(y_true_num, y_pred_num)))
        r2 = _safe_float(r2_score(y_true_num, y_pred_num))

        metrics_rows.append(
            {
                "model": model_name,
                "mae": round(mae, 6),
                "rmse": round(rmse, 6),
                "r2": round(r2, 6),
                "model_path": str(model_path),
            }
        )

    leaderboard = sorted(metrics_rows, key=lambda row: (row["rmse"], row["mae"]))
    payload = {
        "note": "MAE/RMSE/R2 are computed on ordinal-encoded class labels for comparative analysis.",
        "encoding": label_to_int,
        "test_rows": int(len(y_test)),
        "leaderboard_by_rmse": leaderboard,
    }

    report_path = args.report_out.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved regression-style metrics report: {report_path}")
    if args.json:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
