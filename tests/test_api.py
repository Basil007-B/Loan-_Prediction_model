"""API tests using FastAPI's TestClient.

Run from the project root with ``pytest``. Prediction tests need the trained
model file; if it is missing they are skipped with a helpful message
(run ``python -m src.train_model`` first).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.config import MODEL_PATH

client = TestClient(app)

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not found. Run `python -m src.train_model` first.",
)

VALID_APPLICATION = {
    "ApplicantIncome": 50000,
    "CoapplicantIncome": 10000,
    "LoanAmount": 250000,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "Property_Area": "Urban",
    "ApplicantAge": 30,
    "EmploymentYears": 5,
    "ExistingLoans": 1,
    "DebtToIncomeRatio": 0.25,
    "Savings": 100000,
    "LoanPurpose": "Home",
}

STRONG_APPLICATION = {
    **VALID_APPLICATION,
    "ApplicantIncome": 90000,
    "CoapplicantIncome": 20000,
    "LoanAmount": 220000,
    "Gender": "Female",
    "Dependents": "1",
    "Property_Area": "Semiurban",
    "ApplicantAge": 38,
    "EmploymentYears": 10,
    "DebtToIncomeRatio": 0.12,
    "Savings": 150000,
}

WEAK_APPLICATION = {
    **VALID_APPLICATION,
    "Credit_History": 0,
    "LoanAmount": 900000,
    "ExistingLoans": 5,
    "DebtToIncomeRatio": 0.6,
    "Savings": 1000,
}


def test_root_returns_running_message():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"].lower()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@requires_model
def test_valid_prediction_has_correct_structure():
    response = client.post("/predict", json=VALID_APPLICATION)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "loan_status", "probability"}
    assert body["prediction"] in (0, 1)
    assert body["loan_status"] == ("Approved" if body["prediction"] == 1 else "Rejected")
    assert 0.0 <= body["probability"] <= 1.0


@requires_model
def test_strong_applicant_scores_higher_than_weak_applicant():
    strong = client.post("/predict", json=STRONG_APPLICATION).json()
    weak = client.post("/predict", json=WEAK_APPLICATION).json()
    assert strong["probability"] > weak["probability"]
    assert strong["loan_status"] == "Approved"


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("ApplicantIncome", -1),
        ("LoanAmount", -500),
        ("LoanAmount", 0),
        ("Credit_History", 2),
        ("ApplicantAge", 12),
        ("ApplicantAge", 120),
        ("EmploymentYears", -3),
        ("ExistingLoans", -1),
        ("DebtToIncomeRatio", 1.5),
        ("DebtToIncomeRatio", -0.1),
        ("Education", "PhD"),
        ("Dependents", "7"),
    ],
)
def test_invalid_values_are_rejected(field, bad_value):
    payload = {**VALID_APPLICATION, field: bad_value}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_employment_years_cannot_exceed_working_age():
    payload = {**VALID_APPLICATION, "ApplicantAge": 20, "EmploymentYears": 15}
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("missing_field", list(VALID_APPLICATION))
def test_missing_fields_are_rejected(missing_field):
    payload = {key: value for key, value in VALID_APPLICATION.items() if key != missing_field}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_non_json_body_is_rejected():
    response = client.post("/predict", content="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
