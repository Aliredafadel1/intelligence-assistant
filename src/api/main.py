from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def _load_local_env_file() -> None:
    """Load key=value pairs from repo .env for local development."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value


_load_local_env_file()

from .routers import (
    compare_router,
    inspect_router,
    llm_router,
    ml_router,
    rag_router,
    system_router,
)

app = FastAPI(
    title="Decision Intelligence Assistant API",
    version="0.1.0",
    description="API for unified comparison across RAG, non-RAG, ML, and LLM zero-shot methods.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router)
app.include_router(rag_router, prefix="/rag")
app.include_router(ml_router, prefix="/ml")
app.include_router(llm_router, prefix="/llm")
app.include_router(compare_router, prefix="/comparison")
app.include_router(inspect_router, prefix="/inspect")
