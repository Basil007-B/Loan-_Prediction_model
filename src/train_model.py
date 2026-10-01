"""Train, compare, tune and save the loan-approval model.

Run from the project root::

    python -m src.train_model

Workflow
--------
1. Load the CSV and make a stratified 80/20 train/test split.
2. Compare candidate models with 5-fold cross-validation on the TRAINING set.
3. Pick the model with the best cross-validated ROC-AUC (F1 breaks ties).
4. Tune it with GridSearchCV (again only on the training set).
5. Evaluate ONCE on the untouched test set and save pipeline + metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CV_FOLDS,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.evaluate_model import (
    evaluate_pipeline,
    get_model_importance,
    save_evaluation_plots,
)
from src.features import build_pipeline, load_dataset, split_data

logger = logging.getLogger(__name__)

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
N_JOBS = -1
# Models whose cross-validated ROC-AUC is within this margin of the best one
# are treated as tied (differences this small are noise); F1 then decides.
AUC_TIE_TOLERANCE = 0.002


# --------------------------------------------------------------------------- #
# Candidate models and tuning grids
# --------------------------------------------------------------------------- #
def get_candidates() -> dict:
    """Return ``{name: (estimator, param_grid)}`` for every model we compare.

    ``class_weight='balanced'`` variants test the effect of the (mild) class
    imbalance without changing the data. Parameter names use the pipeline
    prefix ``model__``.
    """
    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": [None, "balanced"]},
        ),
        "Logistic Regression (balanced)": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
            {"model__max_depth": [3, 5, 7, 9], "model__min_samples_leaf": [10, 30, 60]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=1),
            {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [3, 5, 10]},
        ),
        "Random Forest (balanced)": (
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=1,
            ),
            {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [3, 5, 10]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }
    try:  # XGBoost is optional: skipped automatically if not installed
        from xgboost import XGBClassifier

        candidates["XGBoost"] = (
            XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            },
        )
    except ImportError:
        logger.warning("xgboost is not installed - skipping XGBoost")
    return candidates


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def compare_models(X_train, y_train, candidates: dict | None = None) -> pd.DataFrame:
    """Cross-validate every candidate on the training data only."""
    candidates = candidates or get_candidates()
    rows = []
    for name, (estimator, _) in candidates.items():
        logger.info("Cross-validating %s ...", name)
        try:
            scores = cross_validate(
                build_pipeline(estimator), X_train, y_train,
                cv=make_cv(), scoring=SCORING, n_jobs=N_JOBS,
            )
        except Exception as error:  # keep going if one optional model breaks
            logger.warning("Skipping %s: %s", name, error)
            continue
        row = {"model": name}
        row.update({metric: scores[f"test_{metric}"].mean() for metric in SCORING})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False)
    return table.reset_index(drop=True)


def select_best_model(comparison: pd.DataFrame, tolerance: float = AUC_TIE_TOLERANCE) -> str:
    """Pick a model from the CROSS-VALIDATION table (never from the test set).

    ROC-AUC is the primary criterion because it does not depend on a decision
    threshold. Models within ``tolerance`` of the best AUC are considered
    tied, and the highest cross-validated F1 breaks the tie. Accuracy alone is
    never used.
    """
    best_auc = comparison["roc_auc"].max()
    tied = comparison[comparison["roc_auc"] >= best_auc - tolerance]
    return str(tied.sort_values("f1", ascending=False).iloc[0]["model"])


def tune_model(name: str, X_train, y_train, candidates: dict | None = None) -> GridSearchCV:
    """Grid-search the chosen model with cross-validation (training data only)."""
    candidates = candidates or get_candidates()
    estimator, grid = candidates[name]
    search = GridSearchCV(
        build_pipeline(estimator),
        param_grid=grid,
        scoring={"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"},
        refit="roc_auc",
        cv=make_cv(),
        n_jobs=N_JOBS,
    )
    logger.info("Tuning %s over %d parameter combinations ...", name, _grid_size(grid))
    return search.fit(X_train, y_train)


def _grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def save_artifacts(pipeline, metadata: dict) -> None:
    """Persist the full pipeline (preprocessing + model) and its metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Saved model to %s (%.2f MB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1e6)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info("Train: %s | Test: %s", X_train.shape, X_test.shape)
    logger.info("Approval rate - train %.3f | test %.3f", y_train.mean(), y_test.mean())

    candidates = get_candidates()
    comparison = compare_models(X_train, y_train, candidates)
    comparison.to_csv(REPORTS_DIR / "model_comparison_cv.csv", index=False)
    logger.info("\n%s", comparison.round(4).to_string(index=False))

    best_name = select_best_model(comparison)
    logger.info("Selected by cross-validated ROC-AUC (F1 breaks ties): %s", best_name)

    search = tune_model(best_name, X_train, y_train, candidates)
    best_pipeline = search.best_estimator_
    best_index = search.best_index_
    cv_summary = {
        metric: float(search.cv_results_[f"mean_test_{metric}"][best_index])
        for metric in ("roc_auc", "f1", "accuracy")
    }
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV scores: %s", cv_summary)

    # The ONLY time the test set is used: a single final evaluation.
    test_metrics = evaluate_pipeline(best_pipeline, X_test, y_test)
    train_metrics = evaluate_pipeline(best_pipeline, X_train, y_train)
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        logger.info("TEST %-9s %.4f   (train %.4f)", metric, test_metrics[metric], train_metrics[metric])

    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    save_evaluation_plots(best_pipeline, X_test, y_test)
    importance = get_model_importance(best_pipeline)
    importance.to_csv(REPORTS_DIR / "model_importance.csv", index=False)
    logger.info("\n%s", importance.to_string(index=False))

    metadata = {
        "model_name": best_name,
        "best_params": search.best_params_,
        "cv_scores_best_params": cv_summary,
        "test_metrics": test_metrics,
        "train_metrics": {k: v for k, v in train_metrics.items() if k != "confusion_matrix"},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "approval_rate_train": float(y_train.mean()),
        "feature_columns": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_artifacts(best_pipeline, metadata)


if __name__ == "__main__":
    main()
