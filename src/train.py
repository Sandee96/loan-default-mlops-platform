"""
train.py

Trains candidate models on the processed + feature-engineered data,
evaluates them, logs everything to MLflow, and saves the
best-performing model and its metrics locally.
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import engineer_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
EXPERIMENT_NAME = "Loan_Default_Risk_Prediction"

# Hyperparameters defined explicitly here (not buried in Pipeline calls)
# so they can be logged to MLflow cleanly.
MODEL_HYPERPARAMS = {
    "logistic_regression": {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 12,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}


def load_processed_data():
    """Load processed train/test splits and apply shared feature engineering."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    logger.info("Loaded processed data. X_train: %s, X_test: %s", X_train.shape, X_test.shape)

    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    logger.info("Applied feature engineering. X_train: %s, X_test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def build_candidate_models() -> dict:
    """Define candidate models as sklearn Pipelines (scaler + classifier)."""
    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**MODEL_HYPERPARAMS["logistic_regression"])),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**MODEL_HYPERPARAMS["random_forest"])),
        ]),
    }
    return candidates


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute standard classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def train_and_evaluate_all(X_train, X_test, y_train, y_test) -> dict:
    """
    Train every candidate model, log each run to MLflow, and
    collect fitted models + metrics for local comparison.
    """
    candidates = build_candidate_models()
    results = {}

    mlflow.set_experiment(EXPERIMENT_NAME)

    for name, pipeline in candidates.items():
        logger.info("Training model: %s", name)

        with mlflow.start_run(run_name=name):
            mlflow.set_tags({
                "project": "loan-default-mlops",
                "dataset": "UCI Default of Credit Card Clients",
            })
            mlflow.log_param("model_name", name)
            for param_name, param_value in MODEL_HYPERPARAMS[name].items():
                mlflow.log_param(param_name, param_value)

            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            mlflow.sklearn.log_model(pipeline, name="model")

            logger.info("Metrics for %s: %s", name, metrics)

        results[name] = {"model": pipeline, "metrics": metrics}

    return results


def select_best_model(results: dict):
    """Select the model with the highest ROC-AUC."""
    best_name = max(results, key=lambda name: results[name]["metrics"]["roc_auc"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]
    logger.info("Best model: %s (ROC-AUC: %.4f)", best_name, best_metrics["roc_auc"])
    return best_name, best_model, best_metrics


def save_best_model(model, output_path: Path = BEST_MODEL_PATH) -> None:
    """Save the best model to disk, creating directories as needed. Compressed to keep file size small."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path, compress=3)
    logger.info("Saved best model to %s", output_path)


def save_metrics(all_results: dict, best_name: str, output_path: Path = METRICS_PATH) -> None:
    """Save metrics for all trained models, flagging the best one, as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_model": best_name,
        "models": {name: res["metrics"] for name, res in all_results.items()},
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved metrics summary to %s", output_path)


def main() -> None:
    X_train, X_test, y_train, y_test = load_processed_data()
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
    best_name, best_model, best_metrics = select_best_model(results)

    save_best_model(best_model)
    save_metrics(results, best_name)

    logger.info("Training complete. Best model: %s, metrics: %s", best_name, best_metrics)


if __name__ == "__main__":
    main()