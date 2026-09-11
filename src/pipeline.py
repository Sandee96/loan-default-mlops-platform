"""
pipeline.py

Orchestrates the full training pipeline end-to-end:
data loading -> preprocessing -> feature engineering -> training ->
evaluation -> MLflow logging -> best model saving -> metadata ->
drift reference snapshot.

Reuses existing functions from data_loader, preprocess, train,
evaluate, and monitoring - no duplicated logic here.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.data_loader import main as load_data_step
from src.preprocess import run_preprocessing
from src.train import (
    load_processed_data,
    save_best_model,
    save_metrics,
    select_best_model,
    train_and_evaluate_all,
)
from src.evaluate import run_full_evaluation
from src.monitoring import save_reference_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
METADATA_PATH = MODELS_DIR / "model_metadata.json"
DATASET_NAME = "UCI Default of Credit Card Clients"


def get_next_version(metadata_path: Path = METADATA_PATH) -> int:
    """Read the previous metadata file, if any, and increment its version."""
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                previous = json.load(f)
            return previous.get("version", 0) + 1
        except (json.JSONDecodeError, KeyError):
            logger.warning("Could not read previous metadata cleanly. Starting at version 1.")
    return 1


def save_model_metadata(best_name: str, best_metrics: dict, feature_names: list,
                         output_path: Path = METADATA_PATH) -> None:
    """Write model metadata for versioning, generated from actual training results."""
    version = get_next_version(output_path)
    metadata = {
        "model_name": best_name,
        "version": version,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "roc_auc": best_metrics["roc_auc"],
        "dataset": DATASET_NAME,
        "features": feature_names,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata (version %d) to %s", version, output_path)


def run_pipeline() -> None:
    """Run the complete end-to-end pipeline."""
    logger.info("=== STEP 1/7: Data loading ===")
    load_data_step()

    logger.info("=== STEP 2/7: Preprocessing ===")
    run_preprocessing()

    logger.info("=== STEP 3/7: Loading processed data + feature engineering ===")
    X_train, X_test, y_train, y_test = load_processed_data()

    logger.info("=== STEP 4/7: Training + MLflow logging ===")
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
    best_name, best_model, best_metrics = select_best_model(results)

    logger.info("=== STEP 5/7: Evaluation (plots) ===")
    run_full_evaluation(best_model, X_test, y_test)

    logger.info("=== STEP 6/7: Saving best model + metadata ===")
    save_best_model(best_model)
    save_metrics(results, best_name)
    save_model_metadata(best_name, best_metrics, list(X_train.columns))

    logger.info("=== STEP 7/7: Saving drift reference dataset ===")
    save_reference_data()

    logger.info(
        "\n=== PIPELINE SUMMARY ===\n"
        "Best model: %s\n"
        "ROC-AUC: %.4f | Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f\n"
        "Model saved to: %s\n"
        "Metadata saved to: %s",
        best_name,
        best_metrics["roc_auc"],
        best_metrics["accuracy"],
        best_metrics["precision"],
        best_metrics["recall"],
        best_metrics["f1"],
        MODELS_DIR / "best_model.pkl",
        METADATA_PATH,
    )


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception:
        logger.exception("Pipeline failed.")
        raise