from pydantic import BaseModel, Field
from typing import Optional


class SingleApplicantRequest(BaseModel):
    """Input schema for a single credit card applicant (Analyst + Customer workflows)."""
    credit_score: int = Field(..., ge=300, le=850, description="FICO credit score (300-850)")
    annual_income: float = Field(..., gt=0, description="Annual income in USD")
    debt_to_income_ratio: float = Field(..., ge=0.0, le=1.0, description="Debt-to-income ratio (0–1)")
    employment_months: int = Field(..., ge=0, description="Months of employment history")
    credit_history_months: int = Field(..., ge=0, description="Months of credit history")
    income_type: str = Field(..., description="Income source: Salaried | Self-Employed | Unemployed")
    payment_status: int = Field(
        ..., ge=0, le=4,
        description="Payment status: 0=On-time, 1=30-59d, 2=60-89d, 3=90+d, 4=Default"
    )
    num_open_accounts: int = Field(default=3, ge=0, le=30)
    monthly_expenses: Optional[float] = Field(default=None, description="Monthly expenses in USD")

    model_config = {"json_schema_extra": {
        "example": {
            "credit_score": 720,
            "annual_income": 75000,
            "debt_to_income_ratio": 0.28,
            "employment_months": 48,
            "credit_history_months": 84,
            "income_type": "Salaried",
            "payment_status": 0,
            "num_open_accounts": 4,
            "monthly_expenses": 1750.0,
        }
    }}


class CustomerSelfServiceRequest(BaseModel):
    """Simplified input schema for the Customer Self-Service Wizard."""
    annual_income: float = Field(..., gt=0)
    income_type: str = Field(..., description="Salaried | Self-Employed | Unemployed")
    employment_months: int = Field(..., ge=0)
    credit_history_months: int = Field(..., ge=0)
    monthly_expenses: float = Field(..., gt=0)

    model_config = {"json_schema_extra": {
        "example": {
            "annual_income": 60000,
            "income_type": "Salaried",
            "employment_months": 36,
            "credit_history_months": 60,
            "monthly_expenses": 1500,
        }
    }}
