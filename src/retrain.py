"""
retrain.py

Automated retraining workflow, triggered by data drift detection.

IMPORTANT LIMITATION (documented per project requirements):
Real-time prediction requests do not contain the true future default
outcome, so genuinely new labelled data is not available at inference
time. For this student/demo project, retraining reuses the existing
labelled training data (data/processed/) as a stand-in for "updated"
data. This simulates the retraining workflow end-to-end (drift check
-> retrain -> evaluate -> compare -> promote) without claiming access
to labels that would not really exist in this scenario.

The candidate model is never allowed to overwrite the current
production model until it has been evaluated and found to meet or
exceed the current model's ROC-AUC.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.monitoring import run_drift_detection
from src.train import (
    load_processed_data,
    save_best_model,
    save_metrics,
    select_best_model,
    train_and_evaluate_all,
)
from src.pipeline import save_model_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
METRICS_PATH = MODELS_DIR / "metrics.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = Path("reports")
RETRAIN_SUMMARY_PATH = REPORTS_DIR / "retrain_summary.json"


def load_current_roc_auc(metrics_path: Path = METRICS_PATH) -> float:
    """
    Read the ROC-AUC of the currently deployed (production) model
    from the last saved metrics.json.
    """
    if not metrics_path.exists():
        logger.warning("No existing metrics.json found - treating current ROC-AUC as 0.0.")
        return 0.0

    with open(metrics_path) as f:
        data = json.load(f)

    best_name = data.get("best_model")
    if best_name is None or best_name not in data.get("models", {}):
        logger.warning("Could not determine current best model from metrics.json - treating ROC-AUC as 0.0.")
        return 0.0

    return data["models"][best_name]["roc_auc"]


def write_retrain_summary(summary: dict, output_path: Path = RETRAIN_SUMMARY_PATH) -> None:
    """Save a summary of this retraining run for later review."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved retraining summary to %s", output_path)


def run_retraining_workflow() -> dict:
    """
    Full retraining workflow:
    1. Run drift detection.
    2. If no drift, exit successfully without retraining.
    3. If drift detected, retrain candidate model(s) on available
       labelled data (simulated updated data - see module docstring).
    4. Compare candidate ROC-AUC against the current production model.
    5. Promote only if candidate meets or exceeds current ROC-AUC.
    6. Track everything in MLflow, tagged as a retraining run.
    """
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_detected": False,
        "retraining_triggered": False,
        "promoted": False,
        "reason": None,
        "current_roc_auc": None,
        "candidate_roc_auc": None,
        "candidate_model_name": None,
    }

    logger.info("=== Step 1: Running drift detection ===")
    drift_summary = run_drift_detection()
    summary["drift_detected"] = drift_summary["dataset_drift_detected"]
    summary["drift_details"] = drift_summary

    if not drift_summary["dataset_drift_detected"]:
        summary["reason"] = "no_drift_detected"
        logger.info("No drift detected. Exiting without retraining. Summary: %s", summary)
        write_retrain_summary(summary)
        return summary

    logger.info("=== Step 2: Drift detected - starting retraining ===")
    summary["retraining_triggered"] = True

    # NOTE: reuses existing labelled training data as a stand-in for
    # genuinely new labelled data, which isn't available in real-time
    # inference. See module docstring.
    X_train, X_test, y_train, y_test = load_processed_data()

    logger.info("=== Step 3: Training candidate model(s) ===")
    results = train_and_evaluate_all(
        X_train, X_test, y_train, y_test,
        extra_tags={"trigger": "retrain"},
    )
    candidate_name, candidate_model, candidate_metrics = select_best_model(results)
    summary["candidate_model_name"] = candidate_name
    summary["candidate_roc_auc"] = candidate_metrics["roc_auc"]

    logger.info("=== Step 4: Comparing candidate vs current production model ===")
    current_roc_auc = load_current_roc_auc()
    summary["current_roc_auc"] = current_roc_auc

    logger.info(
        "Candidate ROC-AUC: %.4f | Current production ROC-AUC: %.4f",
        candidate_metrics["roc_auc"], current_roc_auc,
    )

    if candidate_metrics["roc_auc"] >= current_roc_auc:
        logger.info("=== Step 5: Candidate meets or exceeds current model - promoting ===")
        save_best_model(candidate_model)
        save_metrics(results, candidate_name)
        save_model_metadata(candidate_name, candidate_metrics, list(X_train.columns))

        summary["promoted"] = True
        summary["reason"] = "candidate_met_or_exceeded_current_roc_auc"
    else:
        logger.info("=== Step 5: Candidate did not exceed current model - keeping current production model ===")
        summary["promoted"] = False
        summary["reason"] = "candidate_below_current_roc_auc"

    logger.info("Retraining workflow complete. Summary: %s", summary)
    write_retrain_summary(summary)
    return summary


if __name__ == "__main__":
    run_retraining_workflow()