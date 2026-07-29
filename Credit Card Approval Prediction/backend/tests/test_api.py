"""
test_api.py
-----------
Integration tests for the FastAPI endpoints.
Uses httpx TestClient — does not require the server to be running.

NOTE: Requires the model to be trained first:
  cd backend && python ml/train_model.py
"""

import sys
import os
import io
import csv
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_has_status_field(self):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestPredictSingleEndpoint:
    VALID_PAYLOAD = {
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

    def test_predict_single_valid_payload(self):
        response = client.post("/api/v1/predict-single", json=self.VALID_PAYLOAD)
        # Either 200 (model loaded) or 503 (model not yet trained)
        assert response.status_code in [200, 503]

    def test_predict_single_response_schema(self):
        response = client.post("/api/v1/predict-single", json=self.VALID_PAYLOAD)
        if response.status_code == 200:
            data = response.json()
            assert "decision" in data
            assert "probability" in data
            assert "risk_score" in data
            assert "risk_tier" in data
            assert "compliance_status" in data
            assert data["decision"] in ["APPROVED", "REJECTED", "COMPLIANCE_REJECTED"]
            assert 0.0 <= data["probability"] <= 1.0
            assert 0.0 <= data["risk_score"] <= 100.0

    def test_compliance_rejection_for_high_risk_status(self):
        payload = {**self.VALID_PAYLOAD, "payment_status": 3}
        response = client.post("/api/v1/predict-single", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert data["decision"] == "COMPLIANCE_REJECTED"
            assert data["is_compliance_rejected"] is True
            assert data["probability"] == 0.0

    def test_compliance_rejection_for_default(self):
        payload = {**self.VALID_PAYLOAD, "payment_status": 4}
        response = client.post("/api/v1/predict-single", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert data["is_compliance_rejected"] is True

    def test_invalid_credit_score_range(self):
        payload = {**self.VALID_PAYLOAD, "credit_score": 999}  # out of range
        response = client.post("/api/v1/predict-single", json=payload)
        assert response.status_code == 422

    def test_invalid_payment_status(self):
        payload = {**self.VALID_PAYLOAD, "payment_status": 9}  # out of range
        response = client.post("/api/v1/predict-single", json=payload)
        assert response.status_code == 422


class TestBatchScreeningEndpoint:
    def _make_csv_bytes(self, rows: list) -> bytes:
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return output.getvalue().encode("utf-8")

    def test_batch_csv_upload(self):
        rows = [
            {
                "payment_status": 0, "credit_score": 720, "annual_income": 75000,
                "debt_to_income_ratio": 0.28, "employment_months": 48,
                "credit_history_months": 84, "income_type": "Salaried",
                "num_open_accounts": 4, "monthly_expenses": 1750
            },
            {
                "payment_status": 3, "credit_score": 520, "annual_income": 30000,
                "debt_to_income_ratio": 0.65, "employment_months": 6,
                "credit_history_months": 12, "income_type": "Self-Employed",
                "num_open_accounts": 2, "monthly_expenses": 1800
            },
        ]
        csv_bytes = self._make_csv_bytes(rows)
        response = client.post(
            "/api/v1/batch-screening",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code in [200, 503]

    def test_batch_response_schema(self):
        rows = [
            {
                "payment_status": 0, "credit_score": 720, "annual_income": 75000,
                "debt_to_income_ratio": 0.28, "employment_months": 48,
                "credit_history_months": 84, "income_type": "Salaried",
                "num_open_accounts": 4, "monthly_expenses": 1750
            },
        ]
        csv_bytes = self._make_csv_bytes(rows)
        response = client.post(
            "/api/v1/batch-screening",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        if response.status_code == 200:
            data = response.json()
            assert "total_applicants" in data
            assert "approved_count" in data
            assert "results" in data
            assert data["total_applicants"] == 1

    def test_batch_wrong_file_type(self):
        response = client.post(
            "/api/v1/batch-screening",
            files={"file": ("test.txt", b"not a csv", "text/plain")},
        )
        assert response.status_code == 400

    def test_batch_compliance_rejected_in_results(self):
        rows = [{"payment_status": 4, "annual_income": 50000}]
        csv_bytes = self._make_csv_bytes(rows)
        response = client.post(
            "/api/v1/batch-screening",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        if response.status_code == 200:
            data = response.json()
            assert data["results"][0]["is_compliance_rejected"] is True


class TestEligibilityEndpoint:
    VALID_PAYLOAD = {
        "annual_income": 60000,
        "income_type": "Salaried",
        "employment_months": 36,
        "credit_history_months": 60,
        "monthly_expenses": 1500,
    }

    def test_eligibility_returns_200(self):
        response = client.post("/api/v1/eligibility-check", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_eligibility_response_schema(self):
        response = client.post("/api/v1/eligibility-check", json=self.VALID_PAYLOAD)
        data = response.json()
        assert "pre_qualification_score" in data
        assert "likelihood_label" in data
        assert "estimated_tier" in data
        assert "card_recommendations" in data
        assert "improvement_tips" in data
        assert 0 <= data["pre_qualification_score"] <= 100

    def test_high_income_gets_premium_tier(self):
        payload = {**self.VALID_PAYLOAD, "annual_income": 200000, "employment_months": 120, "credit_history_months": 180}
        response = client.post("/api/v1/eligibility-check", json=payload)
        data = response.json()
        assert data["estimated_tier"] in ["Platinum", "Premium"]

    def test_low_income_unemployed_gets_secured(self):
        payload = {
            "annual_income": 15000,
            "income_type": "Unemployed",
            "employment_months": 0,
            "credit_history_months": 6,
            "monthly_expenses": 1200,
        }
        response = client.post("/api/v1/eligibility-check", json=payload)
        data = response.json()
        assert data["estimated_tier"] == "Secured"
