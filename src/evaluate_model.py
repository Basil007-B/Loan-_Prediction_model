"""Evaluation helpers: metrics, plots and feature importance.

Run from the project root to re-evaluate the saved model on the held-out
test set (the split is reproducible thanks to ``random_state=42``)::

    python -m src.evaluate_model
"""

from __future__ import annotations

import json
import logging

import joblib
import matplotlib

matplotlib.use("Agg")  # file output only; works on servers without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    FIGURES_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.features import load_dataset, split_data

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Return the standard classification metrics as plain floats."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def evaluate_pipeline(pipeline, X, y) -> dict:
    """Score a fitted pipeline on data it was NOT trained on."""
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]
    return compute_metrics(y, predictions, probabilities)


def get_model_importance(pipeline, top_n: int = 15) -> pd.DataFrame:
    """Feature importance (trees) or coefficients (linear) after preprocessing."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values, kind = model.feature_importances_, "importance"
    elif hasattr(model, "coef_"):
        values, kind = model.coef_.ravel(), "coefficient"
    else:
        raise ValueError("Model exposes neither feature_importances_ nor coef_")
    table = pd.DataFrame({"feature": names, kind: values})
    order = table[kind].abs().sort_values(ascending=False).index
    return table.loc[order].head(top_n).reset_index(drop=True)


def get_permutation_importance(pipeline, X_test, y_test, n_repeats: int = 5) -> pd.DataFrame:
    """Drop in ROC-AUC when each RAW input column is shuffled (model-agnostic)."""
    result = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc",
        n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=1,
    )
    table = pd.DataFrame(
        {"feature": X_test.columns, "auc_drop": result.importances_mean, "std": result.importances_std}
    )
    return table.sort_values("auc_drop", ascending=False).reset_index(drop=True)


def save_evaluation_plots(pipeline, X_test, y_test) -> None:
    """Save confusion matrix, ROC curve and permutation importance as PNGs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, display_labels=["Rejected", "Approved"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    importance = get_permutation_importance(pipeline, X_test, y_test).head(12)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(importance["feature"][::-1], importance["auc_drop"][::-1], color="#2a6f97")
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    pipeline = joblib.load(MODEL_PATH)
    _, X_test, _, y_test = split_data(load_dataset())

    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_evaluation_plots(pipeline, X_test, y_test)

    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        logger.info("%-10s %.4f", name, metrics[name])
    logger.info("Confusion matrix: %s", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
