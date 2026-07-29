"""FastAPI router — GET /api/v1/health"""

import time
from fastapi import APIRouter
from app.schemas.response import HealthResponse
from app.services import predictor

router = APIRouter()
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check — returns model status, version, and uptime."""
    model_loaded = predictor.is_model_loaded()
    metadata = predictor.get_metadata() if model_loaded else None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_type=metadata.get("model_type", "Unknown") if metadata else "Not loaded",
        accuracy=metadata.get("accuracy") if metadata else None,
        auc_roc=metadata.get("auc_roc") if metadata else None,
        version="1.0.0",
    )
