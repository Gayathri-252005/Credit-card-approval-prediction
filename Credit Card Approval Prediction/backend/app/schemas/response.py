from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    direction: str  # "positive" | "negative" | "neutral"


class SinglePredictionResponse(BaseModel):
    applicant_id: Optional[str] = None
    decision: str                     # "APPROVED" | "REJECTED" | "COMPLIANCE_REJECTED"
    probability: float                # Approval probability 0–1
    risk_score: float                 # Derived risk score 0–100
    risk_tier: str                    # "Low" | "Medium" | "High" | "Very High"
    compliance_status: str            # "Compliant" | "High-Risk Delinquent"
    is_compliance_rejected: bool
    feature_importance: List[FeatureImportanceItem]
    improvement_tips: List[str]
    model_version: str


class BatchApplicantResult(BaseModel):
    row_index: int
    applicant_id: Optional[str] = None
    credit_score: Optional[int] = None
    annual_income: Optional[float] = None
    payment_status: Optional[int] = None
    compliance_status: str
    decision: str
    probability: float
    risk_score: float
    risk_tier: str
    is_compliance_rejected: bool


class BatchScreeningResponse(BaseModel):
    total_applicants: int
    approved_count: int
    rejected_count: int
    compliance_rejected_count: int
    high_risk_count: int
    approval_rate: float
    results: List[BatchApplicantResult]


class CustomerEligibilityResponse(BaseModel):
    pre_qualification_score: float     # 0–100
    likelihood_label: str              # "Very Likely" | "Likely" | "Possible" | "Unlikely"
    estimated_tier: str                # "Platinum" | "Premium" | "Standard" | "Secured"
    card_recommendations: List[Dict[str, Any]]
    improvement_tips: List[str]
    soft_pull_note: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    accuracy: Optional[float] = None
    auc_roc: Optional[float] = None
    version: str
