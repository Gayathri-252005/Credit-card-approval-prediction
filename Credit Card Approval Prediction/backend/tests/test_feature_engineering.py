"""
test_feature_engineering.py
----------------------------
Unit tests for ComplianceTransformer feature engineering.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add ml directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from compliance_transformer import ComplianceTransformer


@pytest.fixture
def transformer():
    return ComplianceTransformer(threshold=2)


class TestComplianceTransformerBasic:
    def test_status_0_is_compliant(self, transformer):
        result = transformer.transform(np.array([0]))
        assert result[0] == 0, "Status 0 (on-time) must be Compliant (0)"

    def test_status_1_is_compliant(self, transformer):
        result = transformer.transform(np.array([1]))
        assert result[0] == 0, "Status 1 (30-59 days) must be Compliant (0)"

    def test_status_2_is_high_risk(self, transformer):
        result = transformer.transform(np.array([2]))
        assert result[0] == 1, "Status 2 (60-89 days) must be High-Risk (1)"

    def test_status_3_is_high_risk(self, transformer):
        result = transformer.transform(np.array([3]))
        assert result[0] == 1, "Status 3 (90+ days) must be High-Risk (1)"

    def test_status_4_is_high_risk(self, transformer):
        result = transformer.transform(np.array([4]))
        assert result[0] == 1, "Status 4 (default) must be High-Risk (1)"


class TestComplianceTransformerBatch:
    def test_array_batch(self, transformer):
        statuses = [0, 1, 2, 3, 4]
        result = transformer.transform(np.array(statuses))
        expected = [0, 0, 1, 1, 1]
        assert list(result) == expected

    def test_dataframe_input(self, transformer):
        df = pd.DataFrame({"payment_status": [0, 1, 2, 3]})
        result = transformer.transform(df)
        assert list(result) == [0, 0, 1, 1]

    def test_series_input(self, transformer):
        s = pd.Series([0, 1, 2, 4])
        result = transformer.transform(s)
        assert list(result) == [0, 0, 1, 1]


class TestComplianceLabels:
    def test_compliant_label(self, transformer):
        assert transformer.get_compliance_label(0) == "Compliant"
        assert transformer.get_compliance_label(1) == "Compliant"

    def test_high_risk_label(self, transformer):
        assert transformer.get_compliance_label(2) == "High-Risk Delinquent"
        assert transformer.get_compliance_label(3) == "High-Risk Delinquent"
        assert transformer.get_compliance_label(4) == "High-Risk Delinquent"


class TestDisqualification:
    def test_status_0_not_disqualified(self, transformer):
        assert transformer.is_disqualified(0) is False

    def test_status_1_not_disqualified(self, transformer):
        assert transformer.is_disqualified(1) is False

    def test_status_2_disqualified(self, transformer):
        assert transformer.is_disqualified(2) is True

    def test_status_4_disqualified(self, transformer):
        assert transformer.is_disqualified(4) is True


class TestCustomThreshold:
    def test_threshold_3(self):
        ct = ComplianceTransformer(threshold=3)
        result = ct.transform(np.array([0, 1, 2, 3, 4]))
        assert list(result) == [0, 0, 0, 1, 1]

    def test_threshold_1(self):
        ct = ComplianceTransformer(threshold=1)
        result = ct.transform(np.array([0, 1, 2, 3, 4]))
        assert list(result) == [0, 1, 1, 1, 1]


class TestFitTransform:
    def test_fit_returns_self(self, transformer):
        result = transformer.fit(None)
        assert result is transformer

    def test_fit_transform_consistent(self, transformer):
        X = np.array([0, 1, 2, 3, 4])
        t1 = transformer.fit(X).transform(X)
        t2 = transformer.transform(X)
        assert list(t1) == list(t2)
