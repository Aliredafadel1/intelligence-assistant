from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

from .schemas import CompareRequest

try:
    from ..LLM.llm_client import default_model as default_llm_model
    from ..LLM.llm_client import generate_text
    from ..observability.run_logger import RunLogger, utc_now_iso
    from ..rag.triage_with_rag import (
        build_non_rag_prompt,
        build_triage_prompt,
        normalize_rag_answer,
        run_retrieval,
        top_answer_tweet_id,
    )
except ImportError:
    from LLM.llm_client import default_model as default_llm_model
    from observability.run_logger import RunLogger, utc_now_iso
    from rag.triage_with_rag import (
        build_non_rag_prompt,
        build_triage_prompt,
        normalize_rag_answer,
        run_retrieval,
        top_answer_tweet_id,
    )


def _load_ml_predictors() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from ..ML.predict_compare import run_ml_prediction, run_zero_shot_prediction
    except ImportError:
        try:
            from ML.predict_compare import run_ml_prediction, run_zero_shot_prediction
        except ImportError as exc:
            raise RuntimeError(
                "ML endpoints are unavailable in serve mode. "
                "Install training dependencies or run local training mode."
            ) from exc
    return run_ml_prediction, run_zero_shot_prediction


def _timed_call(name: str, fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()
    try:
        result = fn()
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return result, {"ok": True, "latency_ms": elapsed_ms, "error": None, "stage": name}
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return None, {"ok": False, "latency_ms": elapsed_ms, "error": str(exc), "stage": name}


def _truncate_text(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str):
        return value
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + "...(truncated)"


def _redact_for_log(payload: dict[str, Any], req: CompareRequest) -> dict[str, Any]:
    safe = json.loads(json.dumps(payload, default=str))

    if req.hash_ticket_in_log:
        ticket_text = str(safe.get("ticket", ""))
        safe["ticket_hash_sha256"] = hashlib.sha256(ticket_text.encode("utf-8")).hexdigest()
        safe["ticket"] = None
    else:
        safe["ticket"] = _truncate_text(safe.get("ticket"), req.log_max_text_chars)

    outputs = safe.get("outputs", {})
    outputs["rag_answer"] = _truncate_text(outputs.get("rag_answer"), req.log_max_text_chars)
    outputs["non_rag_answer"] = _truncate_text(outputs.get("non_rag_answer"), req.log_max_text_chars)

    retrieved = outputs.get("retrieved", [])
    if isinstance(retrieved, list):
        if req.log_top_k > 0:
            retrieved = retrieved[: req.log_top_k]
        compact_rows: list[dict[str, Any]] = []
        for row in retrieved:
            if not isinstance(row, dict):
                continue
            compact_rows.append(
                {
                    "rank": row.get("rank"),
                    "similarity_score": row.get("similarity_score"),
                    "question_tweet_id": row.get("question_tweet_id"),
                    "answer_tweet_id": row.get("answer_tweet_id"),
                    "query_text": _truncate_text(row.get("query_text"), req.log_max_text_chars),
                    "document_text": _truncate_text(row.get("document_text"), req.log_max_text_chars),
                    "brand_author_id": row.get("brand_author_id"),
                }
            )
        outputs["retrieved"] = compact_rows
    safe["outputs"] = outputs

    return safe


def _build_retrieval_args(req: CompareRequest, llm_model: str) -> argparse.Namespace:
    return argparse.Namespace(
        ticket=req.ticket,
        k=req.k,
        retrieval_backend=req.retrieval_backend,
        index_dir=Path(req.index_dir).resolve() if req.index_dir else None,
        chroma_path=Path(req.chroma_path).resolve() if req.chroma_path else None,
        chroma_collection=req.chroma_collection,
        embed_model=req.embed_model,
        llm_model=llm_model,
        no_llm_fallback=not req.allow_llm_fallback,
        json=True,
    )


def run_compare_pipeline(req: CompareRequest) -> dict[str, Any]:
    run_ml_prediction, run_zero_shot_prediction = _load_ml_predictors()
    llm_model = req.llm_model or default_llm_model()
    started_at = utc_now_iso()
    run_logger = RunLogger(Path(req.log_file))
    run_id = run_logger.new_run_id()
    allow_fallback = req.allow_llm_fallback

    retrieval_args = _build_retrieval_args(req, llm_model)
    retrieved, retrieval_meta = _timed_call("retrieval", lambda: run_retrieval(retrieval_args))

    retrieved_records: list[dict[str, Any]] = []
    top_answer_id: int | None = None
    if retrieval_meta["ok"] and retrieved is not None:
        retrieved_records = retrieved.to_dict(orient="records")
        top_answer_id = top_answer_tweet_id(retrieved)

    rag_answer: str | None = None
    if retrieved is not None:
        rag_prompt = build_triage_prompt(req.ticket, retrieved)
        rag_raw, rag_meta = _timed_call(
            "rag_generation",
            lambda: generate_text(
                rag_prompt,
                model=llm_model,
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

    non_rag_prompt = build_non_rag_prompt(req.ticket)
    non_rag_answer, non_rag_meta = _timed_call(
        "non_rag_generation",
        lambda: generate_text(
            non_rag_prompt,
            model=llm_model,
            temperature=0.1,
            max_tokens=400,
            allow_fallback=allow_fallback,
        ),
    )

    ml_output, ml_meta = _timed_call(
        "ml_prediction",
        lambda: run_ml_prediction(
            req.ticket,
            model_path=Path(req.model_path),
            author_id=req.author_id,
            outbound=req.outbound,
        ),
    )

    zero_shot_output, zero_shot_meta = _timed_call(
        "zero_shot_prediction",
        lambda: run_zero_shot_prediction(
            req.ticket,
            llm_model=llm_model,
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
        "run_id": run_id,
        "timestamp": started_at,
        "ticket": req.ticket,
        "config": {
            "llm_model": llm_model,
            "retrieval_backend": req.retrieval_backend,
            "embed_model": req.embed_model,
            "top_k": req.k,
            "log_file": str(Path(req.log_file).resolve()),
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

    if not req.no_log:
        failed_stages = [name for name, meta in payload["status"].items() if not bool(meta.get("ok", False))]
        log_status = "failed" if failed_stages else "ok"
        log_error = None if not failed_stages else f"Failed stages: {', '.join(failed_stages)}"
        run_logger.log_run(
            run_id=run_id,
            payload=_redact_for_log(payload, req),
            status=log_status,
            error=log_error,
        )
        payload["logging"] = {
            "enabled": True,
            "log_path": str(run_logger.log_path),
            "status": log_status,
            "policy": {
                "log_top_k": req.log_top_k,
                "log_max_text_chars": req.log_max_text_chars,
                "hash_ticket_in_log": req.hash_ticket_in_log,
            },
        }
    else:
        payload["logging"] = {"enabled": False}

    return payload


def get_system_health() -> dict[str, Any]:
    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    llm_env_ok = bool(groq_key and "YOUR_GROQ_KEY" not in groq_key and "REPLACE_WITH_REAL_GROQ_KEY" not in groq_key)

    vector_db_ok = (Path("data/processed/chroma_db")).resolve().exists()
    ml_model_ok = (Path("data/artifacts/priority_baseline.joblib")).resolve().exists()

    components = {
        "api": {"status": "ok"},
        "llm_env": {"status": "ok" if llm_env_ok else "degraded"},
        "vector_db": {"status": "ok" if vector_db_ok else "degraded"},
        "ml_model": {"status": "ok" if ml_model_ok else "degraded"},
    }
    overall = "ok" if all(c["status"] == "ok" for c in components.values()) else "degraded"
    return {"status": overall, "components": components}


def run_rag_ask(
    *,
    question: str,
    k: int = 5,
    retrieval_backend: str = "local",
    index_dir: str | None = None,
    chroma_path: str | None = None,
    chroma_collection: str = "rag_tickets",
    embed_model: str = "text-embedding-3-small",
) -> dict[str, Any]:
    args = argparse.Namespace(
        ticket=question,
        k=k,
        retrieval_backend=retrieval_backend,
        index_dir=Path(index_dir).resolve() if index_dir else None,
        chroma_path=Path(chroma_path).resolve() if chroma_path else None,
        chroma_collection=chroma_collection,
        embed_model=embed_model,
        llm_model=default_llm_model(),
        no_llm_fallback=False,
        json=True,
    )
    retrieved = run_retrieval(args)
    return {
        "question": question,
        "retrieval_backend": retrieval_backend,
        "top_k": k,
        "retrieved": retrieved.to_dict(orient="records"),
        "top_answer_tweet_id": top_answer_tweet_id(retrieved),
    }


def run_ml_predict(*, ticket: str, model_path: str, author_id: str = "unknown_customer", outbound: bool = False) -> dict[str, Any]:
    run_ml_prediction, _ = _load_ml_predictors()
    return run_ml_prediction(
        ticket,
        model_path=Path(model_path),
        author_id=author_id,
        outbound=outbound,
    )


def run_llm_ask(*, prompt: str, llm_model: str | None = None, allow_fallback: bool = True) -> dict[str, Any]:
    text = generate_text(
        prompt,
        model=llm_model or default_llm_model(),
        temperature=0.1,
        max_tokens=400,
        allow_fallback=allow_fallback,
    )
    return {"model": llm_model or default_llm_model(), "text": text}


def get_debug_snapshot() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "default_model_path": str((Path("data/artifacts/priority_baseline.joblib")).resolve()),
        "default_chroma_path": str((Path("data/processed/chroma_db")).resolve()),
        "has_groq_key": bool((os.environ.get("GROQ_API_KEY") or "").strip()),
    }
