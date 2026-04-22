from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

try:
    from ..LLM.llm_client import default_model as default_llm_model
    from ..LLM.llm_client import generate_text
    from ..rag.triage_with_rag import (
        build_non_rag_prompt,
        build_triage_prompt,
        normalize_rag_answer,
        run_retrieval,
        top_answer_tweet_id,
    )
    from .predict_compare import run_ml_prediction, run_zero_shot_prediction
except ImportError:
    from LLM.llm_client import default_model as default_llm_model
    from rag.triage_with_rag import (
        build_non_rag_prompt,
        build_triage_prompt,
        normalize_rag_answer,
        run_retrieval,
        top_answer_tweet_id,
    )
    from predict_compare import run_ml_prediction, run_zero_shot_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified comparison: RAG, non-RAG, ML, and LLM zero-shot."
    )
    parser.add_argument("--ticket", type=str, required=True, help="Incoming ticket text.")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval results for RAG path.")
    parser.add_argument(
        "--retrieval-backend",
        choices=("local", "chroma"),
        default="local",
        help="Where retrieval is executed for RAG path.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Local index directory for retrieval-backend=local.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Chroma path for retrieval-backend=chroma.",
    )
    parser.add_argument(
        "--chroma-collection",
        type=str,
        default="rag_tickets",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for Chroma query path.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/artifacts/priority_baseline.joblib"),
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
        help="LLM model for generation and zero-shot paths.",
    )
    parser.add_argument(
        "--no-llm-fallback",
        action="store_true",
        help="Fail instead of using local fallback when hosted LLM key is missing.",
    )
    parser.add_argument("--json", action="store_true", help="Print output as JSON.")
    return parser.parse_args()


def _timed_call(name: str, fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()
    try:
        result = fn()
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return result, {"ok": True, "latency_ms": elapsed_ms, "error": None, "stage": name}
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return None, {"ok": False, "latency_ms": elapsed_ms, "error": str(exc), "stage": name}


def main() -> None:
    args = parse_args()
    allow_fallback = not args.no_llm_fallback

    retrieved, retrieval_meta = _timed_call("retrieval", lambda: run_retrieval(args))
    retrieved_records: list[dict[str, Any]] = []
    top_answer_id: int | None = None
    if retrieval_meta["ok"] and retrieved is not None:
        retrieved_records = retrieved.to_dict(orient="records")
        top_answer_id = top_answer_tweet_id(retrieved)

    rag_answer: str | None = None
    rag_prompt: str | None = None
    rag_meta: dict[str, Any]
    if retrieved is not None:
        rag_prompt = build_triage_prompt(args.ticket, retrieved)
        rag_raw, rag_meta = _timed_call(
            "rag_generation",
            lambda: generate_text(
                rag_prompt or "",
                model=args.llm_model,
                temperature=0.1,
                max_tokens=400,
                allow_fallback=allow_fallback,
            ),
        )
        if rag_meta["ok"] and rag_raw is not None:
            rag_answer = normalize_rag_answer(rag_raw)
    else:
        rag_meta = {
            "ok": False,
            "latency_ms": 0.0,
            "error": "Skipped because retrieval failed.",
            "stage": "rag_generation",
        }

    non_rag_prompt = build_non_rag_prompt(args.ticket)
    non_rag_answer, non_rag_meta = _timed_call(
        "non_rag_generation",
        lambda: generate_text(
            non_rag_prompt,
            model=args.llm_model,
            temperature=0.1,
            max_tokens=400,
            allow_fallback=allow_fallback,
        ),
    )

    ml_output, ml_meta = _timed_call(
        "ml_prediction",
        lambda: run_ml_prediction(
            args.ticket,
            model_path=args.model_path,
            author_id=args.author_id,
            outbound=args.outbound,
        ),
    )

    zero_shot_output, zero_shot_meta = _timed_call(
        "zero_shot_prediction",
        lambda: run_zero_shot_prediction(
            args.ticket,
            llm_model=args.llm_model,
            allow_fallback=allow_fallback,
        ),
    )

    total_latency_ms = round(
        retrieval_meta["latency_ms"]
        + rag_meta["latency_ms"]
        + non_rag_meta["latency_ms"]
        + ml_meta["latency_ms"]
        + zero_shot_meta["latency_ms"],
        2,
    )

    payload = {
        "ticket": args.ticket,
        "config": {
            "llm_model": args.llm_model,
            "retrieval_backend": args.retrieval_backend,
            "embed_model": args.embed_model,
            "top_k": args.k,
        },
        "outputs": {
            "rag_answer": rag_answer,
            "non_rag_answer": non_rag_answer,
            "ml_prediction": ml_output,
            "llm_zero_shot_prediction": zero_shot_output,
            "retrieved": retrieved_records,
            "top_answer_tweet_id": top_answer_id,
        },
        "metrics": {
            "latency_ms": {
                "retrieval": retrieval_meta["latency_ms"],
                "rag_generation": rag_meta["latency_ms"],
                "non_rag_generation": non_rag_meta["latency_ms"],
                "ml_prediction": ml_meta["latency_ms"],
                "zero_shot_prediction": zero_shot_meta["latency_ms"],
                "total": total_latency_ms,
            },
            "cost": {
                "status": "not_tracked_yet",
                "message": "Token/cost logging is not yet implemented in llm_client.",
            },
        },
        "status": {
            "retrieval": retrieval_meta,
            "rag_generation": rag_meta,
            "non_rag_generation": non_rag_meta,
            "ml_prediction": ml_meta,
            "zero_shot_prediction": zero_shot_meta,
        },
    }

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
