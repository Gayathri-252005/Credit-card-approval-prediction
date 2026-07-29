"""
predictor.py
------------
Model inference service. Loads the trained pipeline and provides
prediction utilities for both single and batch workflows.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# Add ml directory to path for ComplianceTransformer
ML_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ml")
)
sys.path.insert(0, ML_DIR)

from compliance_transformer import ComplianceTransformer, PAYMENT_STATUS_DESCRIPTIONS

MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

MODEL_VERSION = "1.0.0"

# Lazy-loaded singletons
_pipeline = None
_metadata = None
_compliance_transformer = ComplianceTransformer(threshold=2)


def _load_artifacts():
    global _pipeline, _metadata
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    if _metadata is None and os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _metadata = json.load(f)
    return _pipeline, _metadata


def is_model_loaded() -> bool:
    return os.path.exists(MODEL_PATH)


def get_metadata() -> Optional[Dict]:
    _, meta = _load_artifacts()
    return meta


def _build_input_df(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert raw input dict to model-ready DataFrame."""
    # Compute monthly_expenses if not provided
    if data.get("monthly_expenses") is None:
        data["monthly_expenses"] = (
            data["annual_income"] / 12 * data["debt_to_income_ratio"]
        )
    df = pd.DataFrame([data])
    df["compliance_flag"] = _compliance_transformer.transform(df)
    return df


def _get_risk_tier(probability: float) -> str:
    if probability >= 0.80:
        return "Low"
    elif probability >= 0.60:
        return "Medium"
    elif probability >= 0.40:
        return "High"
    return "Very High"


def _compute_feature_importance(pipeline, input_df: pd.DataFrame) -> List[Dict]:
    """Compute normalized feature importances for the prediction."""
    try:
        meta = _metadata
        if meta is None:
            return []
        importances = meta.get("feature_importances", {})
        numeric_feats = meta.get("numeric_features", [])
        cat_feats = meta.get("categorical_features", [])
        all_names = meta.get("all_feature_names", list(importances.keys()))

        # Normalize
        total = sum(importances.values()) or 1
        result = []
        for feat in all_names[:8]:  # Top 8
            imp = importances.get(feat, 0)
            direction = "positive" if feat in ["credit_score", "annual_income",
                                                "employment_months", "credit_history_months"] else \
                        "negative" if feat in ["debt_to_income_ratio", "compliance_flag"] else "neutral"
            result.append({
                "feature": feat,
                "importance": round(imp / total, 4),
                "direction": direction,
            })
        result.sort(key=lambda x: x["importance"], reverse=True)
        return result[:6]
    except Exception:
        return []


def _generate_tips(data: Dict, probability: float, is_compliance: bool) -> List[str]:
    tips = []
    if is_compliance:
        tips.append("Resolve any outstanding past-due accounts immediately.")
        tips.append("Request a payment plan with your creditors to clear delinquencies.")
        tips.append("Wait 12-24 months after resolving delinquencies before reapplying.")
        return tips
    if data.get("credit_score", 850) < 670:
        tips.append("Improve your credit score by paying bills on time consistently.")
    if data.get("debt_to_income_ratio", 0) > 0.40:
        tips.append("Reduce existing debt to lower your debt-to-income ratio below 36%.")
    if data.get("annual_income", 0) < 30000:
        tips.append("Consider a secured card or becoming an authorized user to build credit.")
    if data.get("employment_months", 12) < 12:
        tips.append("Longer employment history (12+ months) significantly improves approval odds.")
    if data.get("credit_history_months", 12) < 24:
        tips.append("Building 2+ years of credit history increases approval likelihood.")
    if not tips:
        tips.append("Maintain your current credit profile — you have a strong application!")
    return tips[:4]


def predict_single(data: Dict[str, Any]) -> Dict[str, Any]:
    """Run single applicant prediction and return full response payload."""
    pipeline, metadata = _load_artifacts()

    compliance_status_code = int(data.get("payment_status", 0))
    is_compliance_rejected = _compliance_transformer.is_disqualified(compliance_status_code)
    compliance_label = _compliance_transformer.get_compliance_label(compliance_status_code)

    input_df = _build_input_df(dict(data))

    if is_compliance_rejected:
        probability = 0.0
        decision = "COMPLIANCE_REJECTED"
    else:
        numeric_feats = metadata["numeric_features"]
        cat_feats = metadata["categorical_features"]
        cols = numeric_feats + cat_feats
        X = input_df[cols]
        prob_arr = pipeline.predict_proba(X)
        probability = float(prob_arr[0][1])
        decision = "APPROVED" if probability >= 0.50 else "REJECTED"

    risk_score = round((1 - probability) * 100, 1)
    risk_tier = _get_risk_tier(probability)
    feature_importance = _compute_feature_importance(pipeline, input_df)
    tips = _generate_tips(data, probability, is_compliance_rejected)

    return {
        "decision": decision,
        "probability": round(probability, 4),
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "compliance_status": compliance_label,
        "is_compliance_rejected": is_compliance_rejected,
        "feature_importance": feature_importance,
        "improvement_tips": tips,
        "model_version": MODEL_VERSION,
    }
