from __future__ import annotations

import argparse
import json

try:
    from ..LLM.llm_client import default_model as default_llm_model
    from ..LLM.llm_client import generate_text
    from ..priority_schema import clamp_confidence, extract_json_object, normalize_priority
except ImportError:
    from LLM.llm_client import default_model as default_llm_model
    from LLM.llm_client import generate_text
    from priority_schema import clamp_confidence, extract_json_object, normalize_priority


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot LLM priority prediction from ticket text.")
    parser.add_argument("--ticket", type=str, required=True, help="Incoming ticket text.")
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


def build_zero_shot_prompt(ticket_text: str) -> str:
    return (
        "You are a support triage assistant.\n"
        "Predict ticket priority from text only (zero-shot, no retrieval context).\n"
        "Return ONLY JSON with keys: priority, confidence, rationale, next_action.\n"
        "priority must be one of P1, P2, P3, P4.\n"
        "confidence must be a number between 0 and 1.\n\n"
        f"Ticket:\n{ticket_text.strip()}"
    )


def normalize_zero_shot_output(raw: str) -> dict:
    parsed = extract_json_object(raw)
    if parsed is None:
        return {
            "priority": "P3",
            "confidence": 0.5,
            "rationale": "Model output was not valid JSON; using normalized fallback.",
            "next_action": "Collect more details and route for manual review.",
        }
    return {
        "priority": normalize_priority(parsed.get("priority")),
        "confidence": clamp_confidence(parsed.get("confidence")),
        "rationale": str(parsed.get("rationale", "")).strip() or "No rationale provided.",
        "next_action": str(parsed.get("next_action", "")).strip() or "No next action provided.",
    }


def main() -> None:
    args = parse_args()
    prompt = build_zero_shot_prompt(args.ticket)
    raw = generate_text(
        prompt,
        model=args.llm_model,
        temperature=0.1,
        max_tokens=300,
        allow_fallback=not args.no_llm_fallback,
    )
    output = normalize_zero_shot_output(raw)
    payload = {
        "ticket": args.ticket,
        "llm_model": args.llm_model,
        "zero_shot_prediction": output,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Ticket: {args.ticket}")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
