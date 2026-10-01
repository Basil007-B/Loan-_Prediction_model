"""Data loading, feature engineering and the reusable preprocessing pipeline.

Everything that turns a raw applicant record into model input lives here and
is stored INSIDE the saved pipeline. The FastAPI app therefore never repeats
any preprocessing: it just sends the raw fields to ``pipeline.predict_proba``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    ASSUMED_ANNUAL_INTEREST_RATE,
    CATEGORICAL_INPUT_COLUMNS,
    DATA_PATH,
    ENGINEERED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_INPUT_COLUMNS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading and splitting
# --------------------------------------------------------------------------- #
def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV and drop exact duplicate rows."""
    df = pd.read_csv(path)
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        logger.warning("Dropping %d duplicate rows", n_duplicates)
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    """Separate X / y and make a stratified, reproducible train/test split.

    The split happens BEFORE any fitting, so the test set stays untouched.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


# --------------------------------------------------------------------------- #
# Feature engineering (stateless, so it cannot leak information)
# --------------------------------------------------------------------------- #
def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN (not inf) when the denominator is zero/negative/missing."""
    return numerator / denominator.where(denominator > 0)


def _monthly_emi(principal: pd.Series, term_months: pd.Series) -> pd.Series:
    """Estimated monthly instalment using a fixed assumed interest rate."""
    rate = ASSUMED_ANNUAL_INTEREST_RATE / 12
    term = term_months.where(term_months > 0)
    growth = (1 + rate) ** term
    return principal * rate * growth / (growth - 1)


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add business-meaningful ratio features to the raw applicant columns."""
    df = pd.DataFrame(X).copy()
    for column in NUMERIC_INPUT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in CATEGORICAL_INPUT_COLUMNS:
        df[column] = df[column].astype(object)

    dependents = pd.to_numeric(
        df["Dependents"].replace("3+", "3"), errors="coerce"
    )
    total_income = df["ApplicantIncome"] + df["CoapplicantIncome"].fillna(0)

    df["TotalIncome"] = total_income
    df["MonthlyIncome"] = total_income / 12
    df["LogTotalIncome"] = np.log1p(total_income.clip(lower=0))
    df["LoanToIncomeRatio"] = _safe_divide(df["LoanAmount"], total_income)
    df["EstimatedEMI"] = _monthly_emi(df["LoanAmount"], df["Loan_Amount_Term"])
    df["EMIToIncomeRatio"] = _safe_divide(df["EstimatedEMI"] * 12, total_income)
    df["IncomePerDependent"] = total_income / (dependents + 1)
    df["SavingsToLoanRatio"] = _safe_divide(df["Savings"], df["LoanAmount"])
    working_age = (df["ApplicantAge"] - 18).clip(lower=1)
    df["EmploymentStability"] = (df["EmploymentYears"] / working_age).clip(0, 1)

    engineered = df[FEATURE_COLUMNS + ENGINEERED_COLUMNS]
    return engineered.replace([np.inf, -np.inf], np.nan)


class OutlierClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Winsorise each column to percentile bounds learned on the TRAINING data."""

    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanpercentile(values, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.nanpercentile(values, self.upper_percentile, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Impute / clip / scale numeric columns and impute / one-hot encode categoricals."""
    numeric_columns = NUMERIC_INPUT_COLUMNS + ENGINEERED_COLUMNS
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", OutlierClipper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, CATEGORICAL_INPUT_COLUMNS),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    """Full pipeline: raw columns -> engineered features -> preprocessing -> model."""
    return Pipeline(
        [
            ("features", FunctionTransformer(engineer_features, validate=False)),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
