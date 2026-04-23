from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from ..services import run_llm_ask

router = APIRouter(tags=["llm"])


class LLMAskRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    llm_model: str | None = None
    allow_fallback: bool = True


@router.post("/ask")
def ask_llm(request: LLMAskRequest) -> dict:
    try:
        return run_llm_ask(
            prompt=request.prompt,
            llm_model=request.llm_model,
            allow_fallback=request.allow_fallback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
