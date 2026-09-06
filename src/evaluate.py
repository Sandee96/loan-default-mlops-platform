"""
evaluate.py

Reusable evaluation functions: metrics, confusion matrix, ROC curve,
and feature importance plots. Does not duplicate training logic -
takes an already-fitted model and test data as input.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts/CI
import matplotlib.pyplot as plt
import numpy as np
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")


def compute_metrics(y_test, y_pred, y_proba) -> dict:
    """Return a dictionary of standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    logger.info("Computed metrics: %s", metrics)
    return metrics


def plot_confusion_matrix(y_test, y_pred, output_dir: Path = FIGURES_DIR) -> Path:
    """Generate and save a confusion matrix plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "confusion_matrix.png"

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info("Saved confusion matrix to %s", output_path)
    return output_path


def plot_roc_curve(model, X_test, y_test, output_dir: Path = FIGURES_DIR) -> Path:
    """Generate and save an ROC curve plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "roc_curve.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info("Saved ROC curve to %s", output_path)
    return output_path


def plot_feature_importance(model, feature_names, output_dir: Path = FIGURES_DIR) -> Path | None:
    """
    Generate and save a feature importance plot, if the model's final
    estimator supports it (tree-based models do, Logistic Regression
    uses coefficients instead - both are handled).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_importance.png"

    # model is a sklearn Pipeline; the actual classifier is the last step.
    estimator = model[-1] if hasattr(model, "__getitem__") else model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
    else:
        logger.warning("Model type does not support feature importance. Skipping plot.")
        return None

    order = np.argsort(importances)[::-1][:15]  # top 15 features
    top_features = np.array(feature_names)[order]
    top_importances = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features[::-1], top_importances[::-1])
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info("Saved feature importance plot to %s", output_path)
    return output_path


def run_full_evaluation(model, X_test, y_test) -> dict:
    """
    Run the complete evaluation suite: metrics + all plots.
    Returns the metrics dictionary for reuse (e.g. in reports or MLflow).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, X_test.columns.tolist())

    logger.info("Full evaluation complete.")
    return metrics


if __name__ == "__main__":
    import joblib

    from src.train import load_processed_data

    model = joblib.load("models/best_model.pkl")
    _, X_test, _, y_test = load_processed_data()
    run_full_evaluation(model, X_test, y_test)