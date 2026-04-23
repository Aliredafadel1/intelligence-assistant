from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..services import run_rag_ask

router = APIRouter(tags=["rag"])


@router.get("/ask")
def ask_rag(
    q: str = Query(..., min_length=1, description="Question / ticket text."),
    k: int = Query(5, ge=1, le=20),
    backend: str = Query("local", pattern="^(local|chroma)$"),
) -> dict:
    try:
        return run_rag_ask(question=q, k=k, retrieval_backend=backend)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
