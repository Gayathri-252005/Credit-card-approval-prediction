"""FastAPI router — POST /api/v1/predict-single"""

from fastapi import APIRouter, HTTPException
from app.schemas.request import SingleApplicantRequest
from app.schemas.response import SinglePredictionResponse, FeatureImportanceItem
from app.services import predictor

router = APIRouter()


@router.post("/predict-single", response_model=SinglePredictionResponse)
async def predict_single(payload: SingleApplicantRequest):
    """
    Analyst / Customer single-applicant prediction endpoint.
    Returns decision, probability, risk tier, feature importances, and tips.
    """
    if not predictor.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running ml/train_model.py",
        )

    try:
        result = predictor.predict_single(payload.model_dump())
        result["feature_importance"] = [
            FeatureImportanceItem(**fi) for fi in result["feature_importance"]
        ]
        return SinglePredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
