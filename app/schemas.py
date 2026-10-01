"""Pydantic models that define and validate the API's input and output.

FastAPI automatically returns HTTP 422 (with a clear message) when a request
breaks any rule below, so invalid data never reaches the model.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoanApplication(BaseModel):
    """One loan application. Field names match the training dataset columns."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    ApplicantIncome: float = Field(..., ge=0, description="Annual applicant income")
    CoapplicantIncome: float = Field(..., ge=0, description="Annual co-applicant income (0 if none)")
    LoanAmount: float = Field(..., gt=0, description="Requested loan amount")
    Loan_Amount_Term: float = Field(..., ge=6, le=480, description="Repayment term in months")
    Credit_History: Literal[0, 1] = Field(..., description="1 = good credit history, 0 = bad")
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    Property_Area: Literal["Urban", "Semiurban", "Rural"]
    ApplicantAge: int = Field(..., ge=18, le=75, description="Age in years")
    EmploymentYears: float = Field(..., ge=0, le=60, description="Years of employment")
    ExistingLoans: int = Field(..., ge=0, le=20, description="Number of current loans")
    DebtToIncomeRatio: float = Field(..., ge=0, le=1, description="Existing debt payments / monthly income (0-1)")
    Savings: float = Field(..., ge=0, description="Total savings")
    LoanPurpose: Literal["Home", "Education", "Personal", "Business", "Vehicle", "Debt Consolidation"]

    @model_validator(mode="after")
    def check_employment_vs_age(self) -> "LoanApplication":
        """Nobody can have worked for more years than their adult life."""
        if self.EmploymentYears > self.ApplicantAge - 14:
            raise ValueError("EmploymentYears is not consistent with ApplicantAge")
        return self


class PredictionResponse(BaseModel):
    """What /predict returns."""

    prediction: Literal[0, 1] = Field(..., description="1 = approved, 0 = rejected")
    loan_status: Literal["Approved", "Rejected"]
    probability: float = Field(..., ge=0, le=1, description="Model probability that the loan is approved")
