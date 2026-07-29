"""FastAPI router — POST /api/v1/batch-screening"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.response import BatchScreeningResponse, BatchApplicantResult
from app.services import compliance as compliance_service
from app.services import predictor

router = APIRouter()


@router.post("/batch-screening", response_model=BatchScreeningResponse)
async def batch_screening(file: UploadFile = File(...)):
    """
    Compliance batch screening endpoint.
    Accepts a CSV file upload and returns per-applicant risk decisions.
    """
    if not predictor.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted.",
        )

    try:
        csv_bytes = await file.read()
        result = compliance_service.process_batch_csv(csv_bytes)
        result["results"] = [BatchApplicantResult(**r) for r in result["results"]]
        return BatchScreeningResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
