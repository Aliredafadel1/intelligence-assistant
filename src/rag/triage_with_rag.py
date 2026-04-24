from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..LLM.llm_client import default_model as default_llm_model
from ..LLM.llm_client import generate_text
from .retrieve_rag import (
    default_chroma_dir,
    default_index_dir,
    load_rag_index,
    retrieve_top_k,
    retrieve_top_k_chroma,
    select_output_columns,
)


def build_context_block(results: pd.DataFrame, max_items: int = 5) -> str:
    if results.empty:
        return "No retrieved context."
    lines: list[str] = []
    for i, (_, row) in enumerate(results.head(max_items).iterrows(), start=1):
        q = str(row.get("query_text", "")).strip()
        d = str(row.get("document_text", row.get("index_text", ""))).strip()
        s = float(row.get("similarity_score", 0.0))
        lines.append(f"[{i}] score={s:.4f}")
        if q:
            lines.append(f"customer: {q}")
        if d:
            lines.append(f"agent: {d}")
        lines.append("")
    return "\n".join(lines).strip()


def build_triage_prompt(ticket_text: str, results: pd.DataFrame) -> str:
    context = build_context_block(results)
    return (
        "You are a support triage assistant.\n"
        "Given a new ticket and retrieved historical context, provide:\n"
        "1) suggested priority (P1/P2/P3/P4),\n"
        "2) confidence (0-1) as triage decision confidence,\n"
        "3) short rationale,\n"
        "4) immediate next action.\n\n"
        "Important: confidence is NOT the raw retrieval similarity score.\n"
        "Use retrieval matches as evidence, then estimate confidence in the final priority decision.\n"
        "Return confidence rounded to 2 decimals.\n\n"
        f"New ticket:\n{ticket_text.strip()}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Output ONLY compact JSON with keys: priority, confidence, rationale, next_action."
    )


def build_non_rag_prompt(ticket_text: str) -> str:
    return (
        "You are a customer support assistant.\n"
        "Generate a concise response to the customer ticket with no retrieved context.\n"
        "Keep it practical and action-oriented in 2-4 sentences.\n"
        "Return ONLY compact JSON with keys: answer, confidence.\n"
        "confidence must be a number between 0 and 1 and represent your answer confidence.\n\n"
        f"Customer ticket:\n{ticket_text.strip()}\n"
    )


def _extract_json_object(text: str) -> dict | None:
    raw = (text or "").strip()
    if not raw:
        return None
    # Handle fenced Markdown JSON responses.
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalize_priority(value: object) -> str:
    val = str(value or "").strip().upper()
    return val if val in {"P1", "P2", "P3", "P4"} else "P3"


def _normalize_confidence(value: object) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    return round(conf, 2)


def normalize_rag_answer(raw_answer: str) -> str:
    parsed = _extract_json_object(raw_answer)
    if parsed is None:
        fallback = {
            "priority": "P3",
            "confidence": 0.5,
            "rationale": "Model output was not valid JSON; using safe normalized fallback.",
            "next_action": "Collect missing details and route ticket for manual review.",
        }
        return json.dumps(fallback)

    normalized = {
        "priority": _normalize_priority(parsed.get("priority")),
        "confidence": _normalize_confidence(parsed.get("confidence")),
        "rationale": str(parsed.get("rationale", "")).strip() or "No rationale provided.",
        "next_action": str(parsed.get("next_action", "")).strip() or "No next action provided.",
    }
    return json.dumps(normalized)


def normalize_non_rag_answer(raw_answer: str) -> dict[str, object]:
    parsed = _extract_json_object(raw_answer)
    if parsed is None:
        return {
            "answer": (raw_answer or "").strip() or "No non-RAG answer returned.",
            "confidence": 0.5,
        }
    return {
        "answer": str(parsed.get("answer", "")).strip() or "No non-RAG answer returned.",
        "confidence": _normalize_confidence(parsed.get("confidence")),
    }


def top_answer_tweet_id(results: pd.DataFrame) -> int | None:
    if results.empty or "answer_tweet_id" not in results.columns:
        return None
    value = results.iloc[0].get("answer_tweet_id")
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end triage: retrieve with RAG then generate decision with Llama 3.1."
    )
    parser.add_argument("--ticket", type=str, required=True, help="Incoming ticket text.")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval results.")
    parser.add_argument(
        "--retrieval-backend",
        choices=("local", "chroma"),
        default="local",
        help="Where retrieval is executed.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Local index directory for --retrieval-backend local.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Chroma path for --retrieval-backend chroma.",
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
        default="text-embedding-3-small",
        help="Embedding model for Chroma query path.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=default_llm_model(),
        help="Generation model for final triage response.",
    )
    parser.add_argument(
        "--no-llm-fallback",
        action="store_true",
        help="Fail instead of using local fallback when hosted LLM key is missing.",
    )
    parser.add_argument("--json", action="store_true", help="Return combined output as JSON.")
    return parser.parse_args()


def run_retrieval(args: argparse.Namespace) -> pd.DataFrame:
    if args.retrieval_backend == "chroma":
        cpath = args.chroma_path.resolve() if args.chroma_path is not None else default_chroma_dir()
        df = retrieve_top_k_chroma(
            args.ticket,
            k=args.k,
            collection_name=args.chroma_collection,
            chroma_path=cpath,
            model=args.embed_model,
        )
        return select_output_columns(df)

    index_dir = args.index_dir.resolve() if args.index_dir is not None else default_index_dir()
    backend, manifest, payload = load_rag_index(index_dir)
    df = retrieve_top_k(args.ticket, backend, manifest, payload, k=args.k)
    return select_output_columns(df)


def main() -> None:
    args = parse_args()
    retrieved = run_retrieval(args)
    rag_prompt = build_triage_prompt(args.ticket, retrieved)
    non_rag_prompt = build_non_rag_prompt(args.ticket)

    rag_answer = generate_text(
        rag_prompt,
        model=args.llm_model,
        temperature=0.1,
        max_tokens=400,
        allow_fallback=not args.no_llm_fallback,
    )
    rag_answer = normalize_rag_answer(rag_answer)
    non_rag_answer = generate_text(
        non_rag_prompt,
        model=args.llm_model,
        temperature=0.1,
        max_tokens=400,
        allow_fallback=not args.no_llm_fallback,
    )
    answer_id = top_answer_tweet_id(retrieved)

    if args.json:
        payload = {
            "ticket": args.ticket,
            "llm_model": args.llm_model,
            "retrieval_backend": args.retrieval_backend,
            "retrieved": retrieved.to_dict(orient="records"),
            "rag_answer": rag_answer,
            "non_rag_answer": non_rag_answer,
            "top_answer_tweet_id": answer_id,
        }
        print(json.dumps(payload, indent=2, default=str))
        return

    print("=== Retrieved Context ===")
    if retrieved.empty:
        print("No retrieval results.")
    else:
        print(retrieved.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))
    print(f"\nTop answer_tweet_id: {answer_id}")
    print("\n=== RAG Answer ===")
    print(rag_answer.encode("ascii", errors="replace").decode("ascii"))
    print("\n=== Non-RAG Answer ===")
    print(non_rag_answer.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
