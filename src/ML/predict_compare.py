from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

try:
    from ..LLM.llm_client import default_model as default_llm_model
    from ..LLM.llm_client import generate_text
    from ..features.ml_features import build_feature_row, ensure_required_columns
    from .predict_zero_shot import build_zero_shot_prompt, normalize_zero_shot_output
    from ..preprocess_common import project_root
    from ..priority_schema import ML_TO_PRIORITY, map_ml_label_to_priority
except ImportError:
    from LLM.llm_client import default_model as default_llm_model
    from LLM.llm_client import generate_text
    from features.ml_features import build_feature_row, ensure_required_columns
    from predict_zero_shot import build_zero_shot_prompt, normalize_zero_shot_output
    from preprocess_common import project_root
    from priority_schema import ML_TO_PRIORITY, map_ml_label_to_priority


def _decode_model_label(pred: object, model: object) -> str:
    raw = str(pred)
    if not raw.isdigit():
        return raw

    label_order = getattr(model, "priority_label_order", None)
    if isinstance(label_order, list) and label_order:
        idx = int(raw)
        if 0 <= idx < len(label_order):
            return str(label_order[idx])

    # Fallback for older saved artifacts without explicit label order.
    default_order = sorted(ML_TO_PRIORITY.keys())
    idx = int(raw)
    if 0 <= idx < len(default_order):
        return default_order[idx]
    return raw


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Compare ML priority prediction vs LLM zero-shot prediction.")
    parser.add_argument("--ticket", type=str, required=True, help="Incoming ticket text.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=root / "data" / "artifacts" / "priority_baseline.joblib",
        help="Path to trained ML model artifact.",
    )
    parser.add_argument(
        "--author-id",
        type=str,
        default="unknown_customer",
        help="Optional author ID for ML features.",
    )
    parser.add_argument(
        "--outbound",
        action="store_true",
        help="Mark message as outbound/non-customer (default inbound).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=default_llm_model(),
        help="LLM model used for zero-shot prediction.",
    )
    parser.add_argument(
        "--no-llm-fallback",
        action="store_true",
        help="Fail instead of using local fallback when hosted LLM key is missing.",
    )
    parser.add_argument("--json", action="store_true", help="Return output as JSON.")
    return parser.parse_args()


def run_ml_prediction(ticket: str, *, model_path: Path, author_id: str, outbound: bool) -> dict:
    resolved_model_path = model_path.resolve()
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")
    model = joblib.load(resolved_model_path)

    features = build_feature_row(ticket, author_id=author_id, inbound=not outbound)
    features = ensure_required_columns(features, model)

    pred = model.predict(features)[0]
    pred_label = _decode_model_label(pred, model)
    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        classes = model.classes_.tolist()
        probabilities = {
            _decode_model_label(c, model): round(float(p), 4) for c, p in zip(classes, probs)
        }

    return {
        "predicted_priority_label": pred_label,
        "predicted_priority": map_ml_label_to_priority(pred_label),
        "probabilities": probabilities,
        "model_path": str(resolved_model_path),
    }


def run_zero_shot_prediction(ticket: str, *, llm_model: str, allow_fallback: bool) -> dict:
    prompt = build_zero_shot_prompt(ticket)
    raw = generate_text(
        prompt,
        model=llm_model,
        temperature=0.1,
        max_tokens=300,
        allow_fallback=allow_fallback,
    )
    return {
        "llm_model": llm_model,
        "prediction": normalize_zero_shot_output(raw),
    }


def main() -> None:
    args = parse_args()
    ml_output = run_ml_prediction(
        args.ticket,
        model_path=args.model_path,
        author_id=args.author_id,
        outbound=args.outbound,
    )
    llm_output = run_zero_shot_prediction(
        args.ticket,
        llm_model=args.llm_model,
        allow_fallback=not args.no_llm_fallback,
    )

    payload = {
        "ticket": args.ticket,
        "ml_prediction": ml_output,
        "llm_zero_shot_prediction": llm_output,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
