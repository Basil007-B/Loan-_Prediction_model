"""FastAPI application that serves the trained loan-approval pipeline.

Start it from the project root::

    uvicorn app.main:app --reload

The saved pipeline already contains feature engineering, imputation, scaling,
encoding and the classifier, so this file only converts the request to a
DataFrame and calls ``predict`` / ``predict_proba``. No preprocessing is
repeated here.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import LoanApplication, PredictionResponse
from src.config import FEATURE_COLUMNS, MODEL_PATH

logger = logging.getLogger("loan_api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Try to load the model at startup; the API still starts if it is missing."""
    try:
        get_model()
    except FileNotFoundError as error:
        logger.warning(str(error))
    yield


app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predicts whether a loan application is likely to be approved.",
    version="1.0.0",
    lifespan=lifespan,
)


@lru_cache(maxsize=1)
def get_model():
    """Load the saved pipeline once and reuse it for every request."""
    model_path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run `python -m src.train_model` first."
        )
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)


@app.get("/")
def root() -> dict:
    """Simple message confirming the API is running."""
    return {"message": "Loan Prediction API is running", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    """Liveness check used by hosting platforms."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication) -> PredictionResponse:
    """Predict approval for one application."""
    try:
        model = get_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    row = pd.DataFrame([application.model_dump()])[FEATURE_COLUMNS]
    approved_probability = float(model.predict_proba(row)[0, 1])
    prediction = int(model.predict(row)[0])
    return PredictionResponse(
        prediction=prediction,
        loan_status="Approved" if prediction == 1 else "Rejected",
        probability=round(approved_probability, 4),
    )
