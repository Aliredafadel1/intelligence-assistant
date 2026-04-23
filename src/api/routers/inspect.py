from __future__ import annotations

from fastapi import APIRouter

from ..services import get_debug_snapshot

router = APIRouter(tags=["inspect"])


@router.get("/debug")
def debug_info() -> dict:
    return get_debug_snapshot()
