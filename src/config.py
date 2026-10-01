"""Central configuration: paths, column names and constants.

Every other module imports from here so that paths and column lists are
defined exactly once. Paths are built from this file's location, so the
project works on any machine and any operating system after a ``git clone``.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (relative to the project root, never absolute)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "loan_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "loan_prediction_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
TARGET = "Loan_Status"  # 1 = Approved, 0 = Rejected

NUMERIC_INPUT_COLUMNS = [
    "ApplicantIncome",      # annual income of the applicant
    "CoapplicantIncome",    # annual income of the co-applicant (0 if none)
    "LoanAmount",           # total amount requested
    "Loan_Amount_Term",     # repayment term in months
    "Credit_History",       # 1 = good credit record, 0 = bad
    "ApplicantAge",
    "EmploymentYears",
    "ExistingLoans",
    "DebtToIncomeRatio",    # existing debt payments / monthly income
    "Savings",
]
CATEGORICAL_INPUT_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "LoanPurpose",
]
FEATURE_COLUMNS = NUMERIC_INPUT_COLUMNS + CATEGORICAL_INPUT_COLUMNS

# Features created inside the pipeline from the raw inputs above.
ENGINEERED_COLUMNS = [
    "TotalIncome",
    "MonthlyIncome",
    "LogTotalIncome",
    "LoanToIncomeRatio",
    "EstimatedEMI",
    "EMIToIncomeRatio",
    "IncomePerDependent",
    "SavingsToLoanRatio",
    "EmploymentStability",
]

# --------------------------------------------------------------------------- #
# Modelling constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
# Interest rate used ONLY to estimate a monthly instalment (EMI) feature.
ASSUMED_ANNUAL_INTEREST_RATE = 0.09
