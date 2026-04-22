from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

try:
    from ..features.ml_features import build_feature_row, ensure_required_columns
    from ..priority_schema import ML_TO_PRIORITY, map_ml_label_to_priority
    from ..preprocess_common import project_root
except ImportError:
    from features.ml_features import build_feature_row, ensure_required_columns
    from priority_schema import ML_TO_PRIORITY, map_ml_label_to_priority
    from preprocess_common import project_root


def _decode_model_label(pred: object, model: object) -> str:
    raw = str(pred)
    if not raw.isdigit():
        return raw

    label_order = getattr(model, "priority_label_order", None)
    if isinstance(label_order, list) and label_order:
        idx = int(raw)
        if 0 <= idx < len(label_order):
            return str(label_order[idx])

    default_order = sorted(ML_TO_PRIORITY.keys())
    idx = int(raw)
    if 0 <= idx < len(default_order):
        return default_order[idx]
    return raw


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Predict priority label for a new support ticket.")
    parser.add_argument("--ticket", type=str, required=True, help="Ticket text to score.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=root / "data" / "artifacts" / "priority_baseline.joblib",
        help="Path to trained sklearn pipeline.",
    )
    parser.add_argument(
        "--author-id",
        type=str,
        default="unknown_customer",
        help="Optional author ID feature value.",
    )
    parser.add_argument(
        "--outbound",
        action="store_true",
        help="Mark message as outbound/non-customer (default is inbound customer).",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    features = build_feature_row(
        args.ticket,
        author_id=args.author_id,
        inbound=not args.outbound,
    )
    features = ensure_required_columns(features, model)

    pred = model.predict(features)[0]
    pred_label = _decode_model_label(pred, model)
    proba_map: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        classes = model.classes_.tolist()
        proba_map = {
            _decode_model_label(c, model): round(float(p), 4) for c, p in zip(classes, probs)
        }

    payload = {
        "ticket": args.ticket,
        "predicted_priority_label": pred_label,
        "predicted_priority": map_ml_label_to_priority(pred_label),
        "probabilities": proba_map,
        "model_path": str(model_path),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Ticket            : {args.ticket}")
        print(f"Predicted label   : {payload['predicted_priority_label']}")
        print(f"Predicted priority: {payload['predicted_priority']}")
        if proba_map:
            print("Class probabilities:")
            for label, score in sorted(proba_map.items(), key=lambda x: x[1], reverse=True):
                print(f"  {label}: {score:.4f}")
        print(f"Model             : {model_path}")


if __name__ == "__main__":
    main()
