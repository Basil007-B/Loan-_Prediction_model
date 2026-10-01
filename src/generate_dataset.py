"""Generate a realistic synthetic loan-application dataset.

Run from the project root::

    python -m src.generate_dataset

How the target is created
-------------------------
1. Applicant attributes are sampled with realistic dependencies (age drives
   employment years and marital status, education and experience drive income,
   loan size depends on the loan purpose, and so on).
2. A *latent approval score* is computed from credit history, income, the
   EMI-to-income ratio, debt-to-income ratio, existing loans, employment
   history, savings and a few small demographic/location effects.
3. The approval probability is ``sigmoid(score)`` and the label is drawn from
   it. Real lenders also look at information we do not observe (employer
   quality, officer discretion, ...), so the label is intentionally *not* a
   deterministic function of the columns. This is what keeps the problem
   realistic: even a perfect model cannot reach 100% accuracy.
4. Missing values are injected AFTER the label is created, so the label is
   based on the true values but the model only ever sees the messy ones.

Note: this is synthetic data. The achievable accuracy is controlled by the
noise built into the generator, so it says nothing about real bank data.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.config import ASSUMED_ANNUAL_INTEREST_RATE, DATA_PATH, RANDOM_STATE

logger = logging.getLogger(__name__)

DEFAULT_N_ROWS = 15_000

# Signal strength and base approval rate of the latent score. A larger scale
# makes decisions more consistent (less unexplained noise); the shift moves
# the overall approval rate. Chosen so roughly 68% of applications are
# approved and the best achievable accuracy is about 84%, not 100%.
SCORE_SCALE = 1.6
SCORE_SHIFT = 0.5

LOAN_PURPOSES = ["Home", "Education", "Personal", "Business", "Vehicle", "Debt Consolidation"]
PURPOSE_PROBABILITIES = [0.30, 0.15, 0.20, 0.10, 0.15, 0.10]
# Median loan size expressed as a multiple of total annual income.
PURPOSE_LOAN_TO_INCOME = {
    "Home": 3.2,
    "Education": 1.0,
    "Personal": 0.45,
    "Business": 1.6,
    "Vehicle": 0.8,
    "Debt Consolidation": 0.6,
}
# Allowed repayment terms (months) and their probabilities per purpose.
PURPOSE_TERMS = {
    "Home": ([180, 240, 300, 360], [0.10, 0.20, 0.20, 0.50]),
    "Education": ([60, 84, 120, 180], [0.25, 0.35, 0.30, 0.10]),
    "Personal": ([12, 24, 36, 60, 84], [0.10, 0.25, 0.30, 0.25, 0.10]),
    "Business": ([36, 60, 120, 180], [0.25, 0.35, 0.30, 0.10]),
    "Vehicle": ([36, 60, 84], [0.30, 0.50, 0.20]),
    "Debt Consolidation": ([24, 36, 60, 84], [0.20, 0.30, 0.35, 0.15]),
}

# Fraction of values replaced by NaN, per column (real-world style gaps).
MISSING_RATES = {
    "Gender": 0.020,
    "Married": 0.005,
    "Dependents": 0.025,
    "Self_Employed": 0.050,
    "ApplicantIncome": 0.010,
    "LoanAmount": 0.030,
    "Loan_Amount_Term": 0.025,
    "Credit_History": 0.060,
    "EmploymentYears": 0.030,
    "DebtToIncomeRatio": 0.020,
    "ExistingLoans": 0.015,
    "Savings": 0.040,
    "LoanPurpose": 0.010,
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def _monthly_emi(principal: np.ndarray, term_months: np.ndarray) -> np.ndarray:
    """Standard EMI formula for a fixed-rate loan."""
    rate = ASSUMED_ANNUAL_INTEREST_RATE / 12
    growth = (1 + rate) ** term_months
    return principal * rate * growth / (growth - 1)


def generate_applicants(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample applicant attributes (no target, no missing values yet)."""
    age = np.clip(rng.normal(38, 10, n_rows), 21, 65).round().astype(int)
    gender = rng.choice(["Male", "Female"], n_rows, p=[0.80, 0.20])
    education = rng.choice(["Graduate", "Not Graduate"], n_rows, p=[0.78, 0.22])
    self_employed = rng.choice(["Yes", "No"], n_rows, p=[0.14, 0.86])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], n_rows, p=[0.38, 0.35, 0.27])

    # Older applicants are more likely to be married; married ones have more dependents.
    married_prob = np.clip(0.25 + (age - 21) * 0.02, 0.20, 0.85)
    married = np.where(rng.random(n_rows) < married_prob, "Yes", "No")
    dependents = np.where(
        married == "Yes",
        rng.choice(["0", "1", "2", "3+"], n_rows, p=[0.35, 0.25, 0.25, 0.15]),
        rng.choice(["0", "1", "2", "3+"], n_rows, p=[0.80, 0.12, 0.06, 0.02]),
    )

    employment_years = np.clip(rng.beta(2.0, 2.2, n_rows) * (age - 20), 0, None).round(1)

    # Income: log-normal, higher with education and experience.
    log_income = (
        10.55
        + 0.25 * (education == "Graduate")
        + 0.025 * np.minimum(employment_years, 20)
        + rng.normal(0, 0.45, n_rows)
    )
    applicant_income = np.exp(log_income)
    outlier = rng.random(n_rows) < 0.004  # a few extreme incomes to exercise outlier handling
    applicant_income[outlier] *= rng.uniform(5, 12, outlier.sum())
    applicant_income = (applicant_income.round(-2)).astype(float)

    has_coapplicant = (married == "Yes") & (rng.random(n_rows) < 0.65)
    coapplicant_income = np.where(
        has_coapplicant, np.exp(rng.normal(10.2, 0.5, n_rows)).round(-2), 0.0
    )
    total_income = applicant_income + coapplicant_income

    purpose = rng.choice(LOAN_PURPOSES, n_rows, p=PURPOSE_PROBABILITIES)
    median_ratio = np.array([PURPOSE_LOAN_TO_INCOME[p] for p in purpose])
    loan_amount = total_income * median_ratio * np.exp(rng.normal(0, 0.45, n_rows))
    loan_amount = np.maximum(loan_amount.round(-3), 5_000.0)

    term = np.array(
        [rng.choice(PURPOSE_TERMS[p][0], p=PURPOSE_TERMS[p][1]) for p in purpose], dtype=float
    )

    existing_loans = np.clip(rng.poisson(0.5 + 0.025 * (age - 21)), 0, 6)
    debt_to_income = np.clip(0.04 + 0.075 * existing_loans + rng.normal(0, 0.04, n_rows), 0.01, 0.85)

    savings = total_income * np.exp(rng.normal(np.log(0.45), 0.9, n_rows)) * (
        1 + 0.02 * employment_years
    )
    savings = savings.round(-2)

    # Good credit history is less likely with high debt burden / many loans.
    credit_prob = _sigmoid(1.9 - 2.5 * (debt_to_income - 0.2) - 0.25 * existing_loans)
    credit_history = (rng.random(n_rows) < credit_prob).astype(float)

    return pd.DataFrame(
        {
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": term,
            "Credit_History": credit_history,
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "Property_Area": property_area,
            "ApplicantAge": age,
            "EmploymentYears": employment_years,
            "ExistingLoans": existing_loans,
            "DebtToIncomeRatio": debt_to_income.round(3),
            "Savings": savings,
            "LoanPurpose": purpose,
        }
    )


def compute_approval_probability(df: pd.DataFrame) -> np.ndarray:
    """Latent approval score -> probability of approval (uses the TRUE values)."""
    total_income = df["ApplicantIncome"] + df["CoapplicantIncome"]
    emi_ratio = _monthly_emi(df["LoanAmount"].to_numpy(), df["Loan_Amount_Term"].to_numpy()) * 12 / total_income
    savings_cover = np.log1p(df["Savings"] / df["LoanAmount"])
    dependents = df["Dependents"].replace("3+", "3").astype(int)

    score = (
        -1.0
        + 2.4 * df["Credit_History"]
        + 0.9 * (np.log(total_income) - np.log(60_000))
        - 4.0 * emi_ratio
        - 3.0 * df["DebtToIncomeRatio"]
        - 0.25 * df["ExistingLoans"]
        + 0.06 * np.minimum(df["EmploymentYears"], 15)
        + 0.5 * savings_cover
        + 0.25 * (df["Education"] == "Graduate")
        - 0.20 * (df["Self_Employed"] == "Yes")
        + 0.30 * (df["Property_Area"] == "Semiurban")
        - 0.10 * (df["Property_Area"] == "Rural")
        + 0.15 * (df["Married"] == "Yes")
        - 0.10 * dependents
        - 0.30 * (df["ApplicantAge"] < 25)
        - 0.02 * np.maximum(df["ApplicantAge"] - 55, 0)
    )
    return _sigmoid(SCORE_SCALE * score.to_numpy() + SCORE_SHIFT)


def inject_missing_values(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Randomly blank out values to mimic incomplete application forms."""
    messy = df.copy()
    for column, rate in MISSING_RATES.items():
        mask = rng.random(len(messy)) < rate
        messy.loc[mask, column] = np.nan
    return messy


def generate_dataset(n_rows: int = DEFAULT_N_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Build the complete dataset including the ``Loan_Status`` target."""
    rng = np.random.default_rng(seed)
    applicants = generate_applicants(n_rows, rng)
    probability = compute_approval_probability(applicants)
    applicants["Loan_Status"] = (rng.random(n_rows) < probability).astype(int)
    logger.info("True approval rate: %.1f%%", 100 * applicants["Loan_Status"].mean())
    return inject_missing_values(applicants, rng)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Generate the synthetic loan dataset.")
    parser.add_argument("--rows", type=int, default=DEFAULT_N_ROWS, help="number of rows")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="random seed")
    args = parser.parse_args()

    dataset = generate_dataset(args.rows, args.seed)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATA_PATH, index=False)
    logger.info("Saved %d rows x %d columns to %s", *dataset.shape, DATA_PATH)


if __name__ == "__main__":
    main()
