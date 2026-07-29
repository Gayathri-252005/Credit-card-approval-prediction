"""
compliance_transformer.py
--------------------------
Custom scikit-learn transformer that maps multi-class payment status codes
to a binary compliance label for use in the feature engineering pipeline.

Payment Status Mapping:
  0  → Compliant          (on-time payments)
  1  → Compliant          (30-59 days late — minor, not disqualifying)
  2  → High-Risk          (60-89 days late — severe)
  3  → High-Risk          (90+ days late — critical)
  4  → High-Risk          (prior default — auto-disqualify)

Compliance Output:
  0  → Compliant
  1  → High-Risk Delinquent (auto-disqualified from credit products)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Threshold above which an applicant is flagged High-Risk
HIGH_RISK_THRESHOLD = 2  # payment_status >= 2 → High-Risk


class ComplianceTransformer(BaseEstimator, TransformerMixin):
    """
    Maps multi-class payment status codes to binary compliance labels.

    Parameters
    ----------
    threshold : int
        Payment status value at or above which an applicant is flagged
        as High-Risk Delinquent. Default is 2 (60+ days late).
    column : str
        Name of the payment status column in the DataFrame.
        Only used when input is a DataFrame.
    """

    def __init__(self, threshold: int = HIGH_RISK_THRESHOLD, column: str = "payment_status"):
        self.threshold = threshold
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform payment status to binary compliance label.

        Parameters
        ----------
        X : array-like, pd.Series, or pd.DataFrame
            Input data containing payment status codes.

        Returns
        -------
        np.ndarray of shape (n_samples,) with values {0, 1}.
        """
        if isinstance(X, pd.DataFrame):
            values = X[self.column].values
        elif isinstance(X, pd.Series):
            values = X.values
        else:
            values = np.asarray(X).ravel()

        compliance_flag = (values >= self.threshold).astype(int)
        return compliance_flag

    def get_compliance_label(self, status: int) -> str:
        """Return human-readable compliance label for a single status code."""
        if status >= self.threshold:
            return "High-Risk Delinquent"
        return "Compliant"

    def is_disqualified(self, status: int) -> bool:
        """Returns True if the applicant is auto-disqualified by compliance rules."""
        return int(status) >= self.threshold


PAYMENT_STATUS_DESCRIPTIONS = {
    0: "On-Time (Compliant)",
    1: "30-59 Days Late (Compliant — Minor)",
    2: "60-89 Days Late (High-Risk)",
    3: "90+ Days Late (High-Risk — Critical)",
    4: "Prior Default (High-Risk — Auto-Disqualify)",
}
