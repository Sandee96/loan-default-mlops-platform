"""
monitoring.py

Prediction logging and data drift monitoring for the deployed model.

Reference data: a snapshot of the training features (data/reference/).
Current data: recent logged predictions (data/monitoring/predictions.csv).
Only matching raw feature columns are compared (engineered features
and prediction outputs are excluded from the drift comparison).
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MONITORING_DIR = Path("data/monitoring")
PREDICTIONS_LOG_PATH = MONITORING_DIR / "predictions.csv"

REFERENCE_DIR = Path("data/reference")
REFERENCE_DATA_PATH = REFERENCE_DIR / "reference_data.csv"
TRAINING_DATA_PATH = Path("data/processed/X_train.csv")

DRIFT_REPORTS_DIR = Path("reports/drift")
DRIFT_HTML_PATH = DRIFT_REPORTS_DIR / "data_drift_report.html"
DRIFT_JSON_PATH = DRIFT_REPORTS_DIR / "data_drift_summary.json"

DRIFT_SHARE_THRESHOLD = 0.5  # dataset drift if >= 50% of columns drift
MIN_CURRENT_ROWS = 30  # need at least this many logged predictions to run drift check

_write_lock = Lock()
_BASE_FIELDS = ["timestamp", "prediction", "probability", "risk_level", "model_version"]


# ---------------------------------------------------------------------------
# Prediction logging (unchanged from before)
# ---------------------------------------------------------------------------

def log_prediction(raw_input: dict, prediction: int, probability: float,
                    risk_level: str, model_version: int,
                    output_path: Path = PREDICTIONS_LOG_PATH) -> None:
    """Append one prediction record to the CSV log."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "model_version": model_version,
        **raw_input,
    }
    fieldnames = _BASE_FIELDS + list(raw_input.keys())

    try:
        with _write_lock:
            file_exists = output_path.exists()
            with open(output_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        logger.info("Logged prediction to %s", output_path)
    except Exception:
        logger.exception("Failed to log prediction - continuing without blocking the API response.")


def load_prediction_log(input_path: Path = PREDICTIONS_LOG_PATH):
    """Load the prediction log. Returns None if missing or empty."""
    if not input_path.exists():
        logger.warning("No prediction log found at %s.", input_path)
        return None
    df = pd.read_csv(input_path)
    if df.empty:
        logger.warning("Prediction log at %s is empty.", input_path)
        return None
    logger.info("Loaded prediction log with %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def save_reference_data(training_data_path: Path = TRAINING_DATA_PATH,
                         output_path: Path = REFERENCE_DATA_PATH) -> None:
    """
    Save a stable snapshot of the training features as the drift
    reference dataset. Run this once after training (e.g. from pipeline.py).
    """
    if not training_data_path.exists():
        raise FileNotFoundError(f"Training data not found at {training_data_path}. Run preprocessing first.")

    df = pd.read_csv(training_data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved reference dataset (%d rows) to %s", len(df), output_path)


def load_reference_data(input_path: Path = REFERENCE_DATA_PATH):
    """Load the saved reference dataset."""
    if not input_path.exists():
        logger.warning("No reference dataset found at %s. Run save_reference_data() first.", input_path)
        return None
    return pd.read_csv(input_path)


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------

def run_drift_detection(reference_df: pd.DataFrame = None, current_df: pd.DataFrame = None) -> dict:
    """
    Run Evidently drift detection comparing current predictions to the
    reference dataset. Only matching raw feature columns are compared.

    Returns a summary dict with dataset_drift_detected, drifted_features_count,
    total_features, and drift_share. Handles insufficient data gracefully by
    returning drift_detected=False with a reason.
    """
    if reference_df is None:
        reference_df = load_reference_data()
    if current_df is None:
        current_df = load_prediction_log()

    if reference_df is None or current_df is None:
        logger.warning("Missing reference or current data - skipping drift detection.")
        return {
            "dataset_drift_detected": False,
            "drifted_features_count": 0,
            "total_features": 0,
            "drift_share": 0.0,
            "reason": "missing_reference_or_current_data",
        }

    if len(current_df) < MIN_CURRENT_ROWS:
        logger.warning(
            "Only %d logged predictions available (need at least %d). Skipping drift detection.",
            len(current_df), MIN_CURRENT_ROWS,
        )
        return {
            "dataset_drift_detected": False,
            "drifted_features_count": 0,
            "total_features": 0,
            "drift_share": 0.0,
            "reason": f"insufficient_current_data ({len(current_df)}/{MIN_CURRENT_ROWS} rows)",
        }

    # Compare only columns present in both datasets (raw features only -
    # excludes prediction/probability/risk_level/model_version/timestamp).
    matching_cols = [c for c in reference_df.columns if c in current_df.columns]
    if not matching_cols:
        logger.warning("No matching columns between reference and current data. Skipping drift detection.")
        return {
            "dataset_drift_detected": False,
            "drifted_features_count": 0,
            "total_features": 0,
            "drift_share": 0.0,
            "reason": "no_matching_columns",
        }

    ref_subset = reference_df[matching_cols]
    cur_subset = current_df[matching_cols]

    logger.info("Running drift detection on %d matching columns, %d current rows.",
                len(matching_cols), len(cur_subset))

    report = Report([DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD)])
    my_eval = report.run(current_data=cur_subset, reference_data=ref_subset)

    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    my_eval.save_html(str(DRIFT_HTML_PATH))
    logger.info("Saved drift HTML report to %s", DRIFT_HTML_PATH)

    result = my_eval.dict()

    # metrics[0] is always the aggregate DriftedColumnsCount metric.
    aggregate = result["metrics"][0]
    drifted_count = aggregate["value"]["count"]
    drift_share = aggregate["value"]["share"]
    total_features = len(matching_cols)
    dataset_drift_detected = drift_share >= DRIFT_SHARE_THRESHOLD

    summary = {
        "dataset_drift_detected": bool(dataset_drift_detected),
        "drifted_features_count": int(drifted_count),
        "total_features": total_features,
        "drift_share": round(float(drift_share), 4),
    }

    with open(DRIFT_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved drift JSON summary to %s: %s", DRIFT_JSON_PATH, summary)

    return summary


if __name__ == "__main__":
    summary = run_drift_detection()
    logger.info("Drift detection result: %s", summary)