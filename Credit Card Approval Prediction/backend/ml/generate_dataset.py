"""
generate_dataset.py
-------------------
Generates a synthetic credit card applicant dataset with 5,000 records.
Features:
  - credit_score         : int  [300, 850]
  - annual_income        : float (USD)
  - debt_to_income_ratio : float [0.05, 0.95]
  - employment_months    : int  [0, 360]
  - credit_history_months: int  [0, 300]
  - income_type          : str  {Salaried, Self-Employed, Unemployed}
  - payment_status       : int  {0: On-time, 1: 30-59d, 2: 60-89d, 3: 90+d, 4: Default}
  - num_open_accounts    : int  [0, 15]
  - monthly_expenses     : float (USD)

Label:
  - approved             : int  {0: Rejected, 1: Approved}
"""

import numpy as np
import pandas as pd
import os

SEED = 42
N = 5000

rng = np.random.default_rng(SEED)


def generate_credit_dataset(n: int = N) -> pd.DataFrame:
    # ── Raw features ──────────────────────────────────────────────────────────
    credit_score = rng.integers(300, 851, size=n)
    annual_income = rng.uniform(18_000, 250_000, size=n).round(2)
    debt_to_income = rng.uniform(0.05, 0.95, size=n).round(4)
    employment_months = rng.integers(0, 361, size=n)
    credit_history_months = rng.integers(0, 301, size=n)
    income_type = rng.choice(
        ["Salaried", "Self-Employed", "Unemployed"],
        size=n,
        p=[0.60, 0.30, 0.10],
    )
    # Payment status multi-class: 0=on-time, 1=30-59d, 2=60-89d, 3=90+d, 4=default
    payment_status = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.60, 0.15, 0.10, 0.10, 0.05])
    num_open_accounts = rng.integers(0, 16, size=n)
    monthly_expenses = (annual_income / 12 * debt_to_income * rng.uniform(0.8, 1.2, size=n)).round(2)

    # ── Rule-based label ──────────────────────────────────────────────────────
    # Weighted scoring to approximate realistic credit decisions
    score_norm = (credit_score - 300) / 550          # [0,1]
    income_norm = np.clip(annual_income / 150_000, 0, 1)
    dti_penalty = 1 - debt_to_income                 # high DTI → lower score
    employment_norm = np.clip(employment_months / 60, 0, 1)
    history_norm = np.clip(credit_history_months / 120, 0, 1)
    payment_penalty = np.where(payment_status == 0, 1.0,
                      np.where(payment_status == 1, 0.8,
                      np.where(payment_status == 2, 0.4,
                      np.where(payment_status == 3, 0.1, 0.0))))

    composite = (
        0.35 * score_norm
        + 0.20 * income_norm
        + 0.15 * dti_penalty
        + 0.10 * employment_norm
        + 0.10 * history_norm
        + 0.10 * payment_penalty
    )

    # Add noise then threshold
    composite += rng.normal(0, 0.05, size=n)
    approved = (composite >= 0.50).astype(int)

    # Hard override: any payment_status >= 3 → always rejected (compliance rule)
    approved[payment_status >= 3] = 0

    df = pd.DataFrame(
        {
            "credit_score": credit_score,
            "annual_income": annual_income,
            "debt_to_income_ratio": debt_to_income,
            "employment_months": employment_months,
            "credit_history_months": credit_history_months,
            "income_type": income_type,
            "payment_status": payment_status,
            "num_open_accounts": num_open_accounts,
            "monthly_expenses": monthly_expenses,
            "approved": approved,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_credit_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "credit_data.csv")
    out_path = os.path.normpath(out_path)
    df.to_csv(out_path, index=False)
    print(f"Dataset saved -> {out_path}")
    print(df.describe())
    print(f"\nApproval rate: {df['approved'].mean():.1%}")
    print(f"Class distribution:\n{df['approved'].value_counts()}")
