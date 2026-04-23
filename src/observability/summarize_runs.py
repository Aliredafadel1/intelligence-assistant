from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .run_logger import build_default_log_path
except ImportError:
    from run_logger import build_default_log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize run logs (counts, success rate, p50/p95 latencies)."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=build_default_log_path(),
        help="Path to JSONL run logs.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="If >0, summarize only the most recent N log lines.",
    )
    parser.add_argument("--json", action="store_true", help="Print output as JSON.")
    return parser.parse_args()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(vals) - 1)
    frac = pos - lower
    return float(vals[lower] + (vals[upper] - vals[lower]) * frac)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    args = parse_args()
    log_path = args.log_file.resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    raw_lines = log_path.read_text(encoding="utf-8").splitlines()
    if args.head > 0:
        raw_lines = raw_lines[-args.head :]

    records: list[dict[str, Any]] = []
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("event_type") != "run_summary":
            continue
        records.append(obj)

    total_runs = len(records)
    ok_runs = sum(1 for r in records if r.get("status") == "ok")
    failed_runs = total_runs - ok_runs
    success_rate = (ok_runs / total_runs) if total_runs else 0.0

    stages = [
        "retrieval",
        "rag_generation",
        "non_rag_generation",
        "ml_prediction",
        "zero_shot_prediction",
        "total",
    ]
    stage_latencies: dict[str, list[float]] = {stage: [] for stage in stages}
    by_backend: dict[str, int] = {}
    by_model: dict[str, int] = {}

    for rec in records:
        payload = rec.get("payload", {})
        if not isinstance(payload, dict):
            continue
        config = payload.get("config", {})
        metrics = payload.get("metrics", {})
        latency = metrics.get("latency_ms", {}) if isinstance(metrics, dict) else {}

        if isinstance(config, dict):
            backend = str(config.get("retrieval_backend", "unknown"))
            by_backend[backend] = by_backend.get(backend, 0) + 1

            llm_model = str(config.get("llm_model", "unknown"))
            by_model[llm_model] = by_model.get(llm_model, 0) + 1

        if isinstance(latency, dict):
            for stage in stages:
                if stage in latency:
                    stage_latencies[stage].append(_safe_float(latency.get(stage)))

    latency_summary = {
        stage: {
            "count": len(vals),
            "p50_ms": round(_percentile(vals, 0.5), 2),
            "p95_ms": round(_percentile(vals, 0.95), 2),
            "avg_ms": round((sum(vals) / len(vals)), 2) if vals else 0.0,
        }
        for stage, vals in stage_latencies.items()
    }

    summary = {
        "log_file": str(log_path),
        "total_runs": total_runs,
        "ok_runs": ok_runs,
        "failed_runs": failed_runs,
        "success_rate": round(success_rate, 4),
        "by_retrieval_backend": by_backend,
        "by_llm_model": by_model,
        "latency_summary_ms": latency_summary,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
