from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
