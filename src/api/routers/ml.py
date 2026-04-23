from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from ..services import run_ml_predict

router = APIRouter(tags=["ml"])


class MLPredictRequest(BaseModel):
    ticket: str = Field(..., min_length=1)
    model_path: str = "data/artifacts/priority_baseline.joblib"
    author_id: str = "unknown_customer"
    outbound: bool = False


@router.post("/predict")
def predict_ml(request: MLPredictRequest) -> dict:
    try:
        return run_ml_predict(
            ticket=request.ticket,
            model_path=request.model_path,
            author_id=request.author_id,
            outbound=request.outbound,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
