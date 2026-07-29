"""
FastAPI Customer self-service eligibility endpoint.
POST /api/v1/eligibility-check
"""

from fastapi import APIRouter, HTTPException
from app.schemas.request import CustomerSelfServiceRequest
from app.schemas.response import CustomerEligibilityResponse

router = APIRouter()

CARD_TIERS = {
    "Platinum": {
        "name": "Platinum Elite Card",
        "limit": "$25,000+",
        "apr": "12.99%",
        "rewards": "5x points on travel, 3x dining",
        "color": "#E5E4E2",
    },
    "Premium": {
        "name": "Premium Rewards Card",
        "limit": "$10,000–$25,000",
        "apr": "16.99%",
        "rewards": "3x points on all purchases",
        "color": "#FFD700",
    },
    "Standard": {
        "name": "Standard Cash Back Card",
        "limit": "$3,000–$10,000",
        "apr": "21.99%",
        "rewards": "1.5% cash back on all purchases",
        "color": "#C0C0C0",
    },
    "Secured": {
        "name": "Secured Builder Card",
        "limit": "$200–$3,000",
        "apr": "24.99%",
        "rewards": "1% cash back, no annual fee",
        "color": "#CD7F32",
    },
}


def _compute_eligibility(data: dict) -> dict:
    """Lightweight rule-based scoring for customer self-service (no model needed)."""
    income = data["annual_income"]
    emp_months = data["employment_months"]
    history_months = data["credit_history_months"]
    expenses = data["monthly_expenses"]
    income_type = data["income_type"]

    # Estimate DTI from expenses
    monthly_income = income / 12
    dti = min(expenses / monthly_income, 0.95) if monthly_income > 0 else 0.95

    # Weighted scoring (no credit score required — soft pull friendly)
    income_score = min(income / 100_000, 1.0) * 25
    emp_score = min(emp_months / 60, 1.0) * 20
    history_score = min(history_months / 84, 1.0) * 25
    dti_score = max(0, (1 - dti / 0.50)) * 20
    type_score = {"Salaried": 10, "Self-Employed": 7, "Unemployed": 0}.get(income_type, 5)

    total = income_score + emp_score + history_score + dti_score + type_score

    if total >= 80:
        tier = "Platinum"
        likelihood = "Very Likely"
    elif total >= 60:
        tier = "Premium"
        likelihood = "Likely"
    elif total >= 40:
        tier = "Standard"
        likelihood = "Possible"
    else:
        tier = "Secured"
        likelihood = "Unlikely"

    # Build card recommendations based on tier hierarchy
    tier_order = ["Secured", "Standard", "Premium", "Platinum"]
    tier_idx = tier_order.index(tier)
    recommendations = [CARD_TIERS[t] for t in tier_order[: tier_idx + 1]]
    recommendations.reverse()  # best match first

    # Generate personalized tips
    tips = []
    if history_months < 24:
        tips.append("Build 2+ years of credit history with a secured card or as an authorized user.")
    if emp_months < 12:
        tips.append("6+ months of stable employment significantly improves your application.")
    if dti > 0.40:
        tips.append(f"Reduce monthly expenses or increase income to lower your DTI ratio.")
    if income_type == "Unemployed":
        tips.append("Demonstrate a steady income source before applying for unsecured credit.")
    if not tips:
        tips.append("You have an excellent financial profile! Apply with confidence.")

    return {
        "pre_qualification_score": round(total, 1),
        "likelihood_label": likelihood,
        "estimated_tier": tier,
        "card_recommendations": recommendations,
        "improvement_tips": tips,
        "soft_pull_note": (
            "This pre-qualification check does NOT affect your credit score. "
            "A hard credit inquiry is only performed upon formal application."
        ),
    }


@router.post("/eligibility-check", response_model=CustomerEligibilityResponse)
async def eligibility_check(payload: CustomerSelfServiceRequest):
    """
    Customer self-service pre-qualification endpoint.
    Returns eligibility score, tier, card recommendations, and improvement tips.
    """
    try:
        result = _compute_eligibility(payload.model_dump())
        return CustomerEligibilityResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
