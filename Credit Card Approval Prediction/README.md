# CreditIQ — Credit Risk & Eligibility Decisioning System

An AI-powered credit card application screening platform supporting three operational workflows.

---

## Quick Start

### 1. Backend (FastAPI + XGBoost)

```powershell
cd backend

# Install dependencies (first time only)
py -m pip install -r requirements.txt

# Generate dataset + train model (first time only)
py ml/generate_dataset.py
py ml/train_model.py

# Start the API server
py -m uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 2. Frontend (Vanilla HTML/JS)

Simply open `frontend/index.html` in your browser, OR serve it with Python:

```powershell
cd frontend
py -m http.server 3000
# Then open http://localhost:3000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/v1/health` | Model status & metrics |
| `POST` | `/api/v1/predict-single` | Analyst single-applicant screening |
| `POST` | `/api/v1/batch-screening` | Compliance CSV batch upload |
| `POST` | `/api/v1/eligibility-check` | Customer self-service pre-qualification |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict-single \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 720,
    "annual_income": 75000,
    "debt_to_income_ratio": 0.28,
    "employment_months": 48,
    "credit_history_months": 84,
    "income_type": "Salaried",
    "payment_status": 0,
    "num_open_accounts": 4
  }'
```

---

## Compliance Logic

Payment status codes map to binary compliance labels:

| Code | Description | Compliance |
|------|-------------|------------|
| 0 | On-Time | ✅ Compliant |
| 1 | 30-59 Days Late | ✅ Compliant (minor) |
| 2 | 60-89 Days Late | ❌ **High-Risk** |
| 3 | 90+ Days Late | ❌ **High-Risk** |
| 4 | Prior Default | ❌ **High-Risk** (auto-disqualify) |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Classifier |
| Training Samples | 4,000 |
| Test Accuracy | **85.4%** |
| AUC-ROC | **0.9317** |
| F1 (Approved) | 0.90 |
| F1 (Rejected) | 0.72 |

Top feature: `compliance_flag` (70.2% importance) — the binary compliance transformer.

---

## Running Tests

```powershell
cd backend
py -m pytest tests/ -v
```

- `test_feature_engineering.py` — 18 unit tests for ComplianceTransformer
- `test_api.py` — 16 integration tests across all endpoints

---

## Project Structure

```
Credit Card Approval Prediction/
├── backend/
│   ├── app/            # FastAPI application
│   │   ├── main.py
│   │   ├── routers/    # predict, batch, health, eligibility
│   │   ├── schemas/    # Pydantic request/response models
│   │   └── services/   # predictor, compliance logic
│   ├── ml/             # ML pipeline
│   │   ├── generate_dataset.py
│   │   ├── train_model.py
│   │   └── compliance_transformer.py
│   ├── models/         # Trained model artifacts (.joblib, .json)
│   ├── data/           # Synthetic dataset (credit_data.csv)
│   └── tests/          # pytest test suites
└── frontend/
    ├── index.html      # Single-page application
    ├── style.css       # Dark theme design system
    └── app.js          # Frontend logic
```
