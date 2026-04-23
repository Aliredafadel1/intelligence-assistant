from __future__ import annotations

from fastapi import APIRouter

from ..services import get_system_health

router = APIRouter(tags=["system"])


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "Decision Intelligence Assistant API is running."}


@router.get("/health")
def health() -> dict:
    return get_system_health()
