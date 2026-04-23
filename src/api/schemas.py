from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    ticket: str = Field(..., min_length=1, description="Incoming ticket text.")
    k: int = Field(5, ge=1, le=20, description="Top-k retrieval results.")
    retrieval_backend: str = Field("local", pattern="^(local|chroma)$")
    index_dir: str | None = None
    chroma_path: str | None = None
    chroma_collection: str = "rag_tickets"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_path: str = "data/artifacts/priority_baseline.joblib"
    author_id: str = "unknown_customer"
    outbound: bool = False
    llm_model: str | None = None
    allow_llm_fallback: bool = True
    log_file: str = "logs/runs.jsonl"
    no_log: bool = False
    log_top_k: int = Field(3, ge=0, le=20)
    log_max_text_chars: int = Field(240, ge=0, le=2000)
    hash_ticket_in_log: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticket": "urgent payment failed and internet down",
                    "retrieval_backend": "chroma",
                    "k": 5,
                    "chroma_collection": "rag_tickets",
                    "allow_llm_fallback": True,
                    "no_log": False,
                    "log_top_k": 2,
                    "log_max_text_chars": 120,
                    "hash_ticket_in_log": True,
                },
                {
                    "ticket": "app is slow but usable",
                    "retrieval_backend": "local",
                    "k": 3,
                    "model_path": "data/artifacts/priority_baseline.joblib",
                    "allow_llm_fallback": True,
                    "no_log": True,
                },
            ]
        }
    }


class CompareResponse(BaseModel):
    run_id: str
    timestamp: str
    ticket: str
    config: dict[str, Any]
    outputs: dict[str, Any]
    metrics: dict[str, Any]
    status: dict[str, Any]
    logging: dict[str, Any]
