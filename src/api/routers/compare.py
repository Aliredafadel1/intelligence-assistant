from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import CompareRequest, CompareResponse
from ..services import run_compare_pipeline

router = APIRouter(tags=["comparison"])


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Run unified 4-way comparison",
    description=(
        "Runs retrieval + RAG generation, non-RAG generation, ML prediction, "
        "and LLM zero-shot prediction in one request with latency breakdown."
    ),
)
def compare(request: CompareRequest) -> CompareResponse:
    try:
        payload = run_compare_pipeline(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
    return CompareResponse(**payload)
