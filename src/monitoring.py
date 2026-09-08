"""
monitoring.py

Prediction logging and (later) drift monitoring for the deployed model.

This module handles logging every successful prediction to disk so it
can be used later as "current data" for drift detection and as a
signal for the retraining workflow.
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MONITORING_DIR = Path("data/monitoring")
PREDICTIONS_LOG_PATH = MONITORING_DIR / "predictions.csv"

# Guards concurrent writes from multiple requests in the same process.
# Doesn't protect against multiple separate processes writing at once,
# but that's an acceptable simplification for this project's scale.
_write_lock = Lock()

# Column order for the CSV. Raw input fields are added dynamically
# based on whatever the request contains, appended after these.
_BASE_FIELDS = ["timestamp", "prediction", "probability", "risk_level", "model_version"]


def log_prediction(raw_input: dict, prediction: int, probability: float,
                    risk_level: str, model_version: int,
                    output_path: Path = PREDICTIONS_LOG_PATH) -> None:
    """
    Append one prediction record to the CSV log.
    Creates the file (with header) on first write.
    Safe to call from multiple requests within the same process.
    """
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
        # Logging failures should never break a prediction response.
        logger.exception("Failed to log prediction - continuing without blocking the API response.")


def load_prediction_log(input_path: Path = PREDICTIONS_LOG_PATH):
    """
    Load the prediction log for downstream use (e.g. drift monitoring).
    Returns None if the log doesn't exist yet or has no rows.
    """
    import pandas as pd

    if not input_path.exists():
        logger.warning("No prediction log found at %s.", input_path)
        return None

    df = pd.read_csv(input_path)
    if df.empty:
        logger.warning("Prediction log at %s is empty.", input_path)
        return None

    logger.info("Loaded prediction log with %d rows.", len(df))
    return df