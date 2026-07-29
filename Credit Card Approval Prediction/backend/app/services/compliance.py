"""
compliance.py
-------------
Batch compliance screening service.
Processes CSV uploads, applies ComplianceTransformer, and runs model predictions.
"""

import os
import sys
import io
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any

ML_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ml")
)
sys.path.insert(0, ML_DIR)

from compliance_transformer import ComplianceTransformer

MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

_pipeline = None
_metadata = None
_ct = ComplianceTransformer(threshold=2)


def _load_artifacts():
    global _pipeline, _metadata
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    if _metadata is None and os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _metadata = json.load(f)
    return _pipeline, _metadata


def _get_risk_tier(probability: float) -> str:
    if probability >= 0.80:
        return "Low"
    elif probability >= 0.60:
        return "Medium"
    elif probability >= 0.40:
        return "High"
    return "Very High"


# Expected columns mapping — flexible CSV column names
COLUMN_ALIASES = {
    "credit_score":           ["credit_score", "creditscore", "fico_score", "fico"],
    "annual_income":          ["annual_income", "annualincome", "income", "yearly_income"],
    "debt_to_income_ratio":   ["debt_to_income_ratio", "dti", "debt_ratio", "dti_ratio"],
    "employment_months":      ["employment_months", "emp_months", "employment_length"],
    "credit_history_months":  ["credit_history_months", "credit_history", "history_months"],
    "income_type":            ["income_type", "income_source", "employment_type"],
    "payment_status":         ["payment_status", "payment_history", "pay_status"],
    "num_open_accounts":      ["num_open_accounts", "open_accounts", "accounts"],
    "monthly_expenses":       ["monthly_expenses", "expenses", "monthly_expense"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remap flexible CSV column names to standard feature names."""
    col_map = {}
    df_cols_lower = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df_cols_lower:
                col_map[df_cols_lower[alias]] = standard
                break
    return df.rename(columns=col_map)


def process_batch_csv(csv_bytes: bytes) -> Dict[str, Any]:
    """
    Process a CSV file upload for batch compliance screening.

    Returns a dict with per-applicant results and aggregate statistics.
    """
    pipeline, metadata = _load_artifacts()

    # Parse CSV
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df = _normalize_columns(df)

    required_cols = ["payment_status"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: '{col}'")

    # Fill defaults for optional columns
    defaults = {
        "credit_score": 650,
        "annual_income": 50000,
        "debt_to_income_ratio": 0.35,
        "employment_months": 24,
        "credit_history_months": 48,
        "income_type": "Salaried",
        "num_open_accounts": 3,
        "monthly_expenses": 1200,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Compliance transformation
    df["compliance_flag"] = _ct.transform(df)
    df["compliance_status"] = df["payment_status"].apply(_ct.get_compliance_label)
    df["is_compliance_rejected"] = df["payment_status"].apply(_ct.is_disqualified)

    numeric_feats = metadata["numeric_features"]
    cat_feats = metadata["categorical_features"]
    cols = numeric_feats + cat_feats

    # Ensure income_type is string
    df["income_type"] = df["income_type"].fillna("Salaried").astype(str)
    df["monthly_expenses"] = df.get("monthly_expenses", df["annual_income"] / 12 * df["debt_to_income_ratio"])

    X = df[cols].copy()

    # Predict probabilities
    proba = pipeline.predict_proba(X)[:, 1]

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        ps = int(row.get("payment_status", 0))
        is_rejected_compliance = bool(row["is_compliance_rejected"])
        prob = 0.0 if is_rejected_compliance else float(proba[i])
        decision = "COMPLIANCE_REJECTED" if is_rejected_compliance else ("APPROVED" if prob >= 0.50 else "REJECTED")

        results.append({
            "row_index": i,
            "applicant_id": str(row.get("applicant_id", f"APP-{i+1:04d}")),
            "credit_score": int(row.get("credit_score", 0)) if "credit_score" in df.columns else None,
            "annual_income": float(row.get("annual_income", 0)),
            "payment_status": ps,
            "compliance_status": str(row["compliance_status"]),
            "decision": decision,
            "probability": round(prob, 4),
            "risk_score": round((1 - prob) * 100, 1),
            "risk_tier": _get_risk_tier(prob),
            "is_compliance_rejected": is_rejected_compliance,
        })

    approved = sum(1 for r in results if r["decision"] == "APPROVED")
    compliance_rejected = sum(1 for r in results if r["decision"] == "COMPLIANCE_REJECTED")
    rejected = len(results) - approved - compliance_rejected
    high_risk = sum(1 for r in results if r["risk_tier"] in ["High", "Very High"])

    return {
        "total_applicants": len(results),
        "approved_count": approved,
        "rejected_count": rejected,
        "compliance_rejected_count": compliance_rejected,
        "high_risk_count": high_risk,
        "approval_rate": round(approved / len(results), 4) if results else 0.0,
        "results": results,
    }
