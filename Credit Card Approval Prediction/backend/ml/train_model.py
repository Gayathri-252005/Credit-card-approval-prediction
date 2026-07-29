"""
train_model.py
--------------
ML pipeline: preprocessing + XGBoost classifier.
Saves artifacts to backend/models/:
  - xgb_model.joblib   : trained XGBoost classifier
  - scaler.joblib      : StandardScaler for numeric features
  - encoder.joblib     : OneHotEncoder for categorical features
  - feature_names.joblib : ordered feature name list
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Add parent dir to path so we can import compliance_transformer
sys.path.insert(0, os.path.dirname(__file__))

try:
    from xgboost import XGBClassifier
    MODEL_TYPE = "XGBoost"
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    MODEL_TYPE = "RandomForest"

from compliance_transformer import ComplianceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature config ────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "credit_score",
    "annual_income",
    "debt_to_income_ratio",
    "employment_months",
    "credit_history_months",
    "num_open_accounts",
    "monthly_expenses",
    "compliance_flag",           # Engineered: binary from payment_status
]

CATEGORICAL_FEATURES = ["income_type"]

TARGET = "approved"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to raw dataframe."""
    ct = ComplianceTransformer(threshold=2, column="payment_status")
    df = df.copy()
    df["compliance_flag"] = ct.transform(df)
    return df


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} records from {DATA_PATH}")
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def build_model():
    if MODEL_TYPE == "XGBoost":
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
    return classifier


def train():
    print(f"\n{'='*60}")
    print(f"  Credit Risk ML Pipeline — {MODEL_TYPE}")
    print(f"{'='*60}\n")

    X, y = load_data()
    print(f"Class distribution:\n{y.value_counts().to_string()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    classifier = build_model()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test AUC-ROC  : {auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # ── Feature importance ─────────────────────────────────────────────────
    # Get feature names after one-hot encoding
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    if MODEL_TYPE == "XGBoost":
        importances = pipeline.named_steps["classifier"].feature_importances_
    else:
        importances = pipeline.named_steps["classifier"].feature_importances_

    feature_importance_dict = dict(zip(all_feature_names, importances.tolist()))
    feature_importance_sorted = dict(
        sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    print(f"\nTop Feature Importances:")
    for feat, imp in list(feature_importance_sorted.items())[:10]:
        print(f"  {feat:35s}: {imp:.4f}")

    # ── Save artifacts ──────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "xgb_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nModel pipeline saved -> {model_path}")

    # Save metadata
    metadata = {
        "model_type": MODEL_TYPE,
        "accuracy": round(acc, 4),
        "auc_roc": round(auc, 4),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_feature_names": all_feature_names,
        "feature_importances": feature_importance_sorted,
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test),
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved    -> {meta_path}")

    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"{'='*60}\n")
    return pipeline, metadata


if __name__ == "__main__":
    train()
