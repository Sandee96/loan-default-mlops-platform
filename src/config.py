"""
config.py

Centralized project-wide constants: shared file paths, dataset info,
and model settings, referenced in the technical documentation as the
intended single source of truth for these values.

Note: as of this submission, individual modules (data_loader.py,
preprocess.py, train.py, etc.) each define their own local copies of
the paths/settings relevant to them, for stability under deadline
constraints. This file documents the canonical values and is intended
as the refactor target for a future consolidation pass.
"""

from pathlib import Path

# --- Dataset ---
DATASET_ID = 350
DATASET_NAME = "UCI Default of Credit Card Clients"
TARGET_COL = "default"

# --- Reproducibility ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Directories ---
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
MONITORING_DIR = DATA_DIR / "monitoring"

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
DRIFT_REPORTS_DIR = REPORTS_DIR / "drift"

# --- Key file paths ---
RAW_DATA_PATH = RAW_DATA_DIR / "credit_default.csv"
REFERENCE_DATA_PATH = REFERENCE_DATA_DIR / "reference_data.csv"
PREDICTIONS_LOG_PATH = MONITORING_DIR / "predictions.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

DRIFT_HTML_PATH = DRIFT_REPORTS_DIR / "data_drift_report.html"
DRIFT_JSON_PATH = DRIFT_REPORTS_DIR / "data_drift_summary.json"
RETRAIN_SUMMARY_PATH = REPORTS_DIR / "retrain_summary.json"

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME = "Loan_Default_Risk_Prediction"

# --- Drift monitoring ---
DRIFT_SHARE_THRESHOLD = 0.5
MIN_CURRENT_ROWS_FOR_DRIFT = 30

# --- Risk classification thresholds (used by api/main.py) ---
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.60