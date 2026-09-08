"""
app.py

Streamlit monitoring dashboard for the loan default prediction platform.
Reads directly from actual project artifacts (model metadata, metrics,
prediction logs, drift summary) - no hardcoded or simulated numbers.
Handles missing files gracefully so the dashboard still loads even
before every artifact exists yet.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Default Risk Platform - Monitoring", layout="wide")

METRICS_PATH = Path("models/metrics.json")
METADATA_PATH = Path("models/model_metadata.json")
PREDICTIONS_LOG_PATH = Path("data/monitoring/predictions.csv")
DRIFT_SUMMARY_PATH = Path("reports/drift/data_drift_summary.json")


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_predictions():
    if not PREDICTIONS_LOG_PATH.exists():
        return None
    df = pd.read_csv(PREDICTIONS_LOG_PATH)
    if df.empty:
        return None
    return df


st.title("Loan Default Risk Prediction Platform")
st.caption("Real-time MLOps monitoring dashboard - project overview, model health, and drift status.")

st.markdown("---")

# --- Project overview ---
st.header("Project Overview")
st.write(
    "Real-time credit default risk prediction service with an automated MLOps lifecycle: "
    "MLflow experiment tracking, Evidently drift monitoring, and drift-triggered retraining. "
    "Built on the UCI Default of Credit Card Clients dataset."
)

st.markdown("---")

# --- Model info ---
st.header("Current Model")
metadata = load_json(METADATA_PATH)
metrics = load_json(METRICS_PATH)

if metadata is None:
    st.warning("No model metadata found yet. Run the training pipeline first (`python -m src.pipeline`).")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", metadata.get("model_name", "N/A"))
    col2.metric("Version", metadata.get("version", "N/A"))
    col3.metric("ROC-AUC", f"{metadata.get('roc_auc', 0):.4f}")
    training_date = metadata.get("training_date", "N/A")
    col4.metric("Last Trained", training_date.split("T")[0] if "T" in str(training_date) else training_date)

if metrics:
    best_name = metrics.get("best_model")
    best_metrics = metrics.get("models", {}).get(best_name, {})
    if best_metrics:
        st.subheader("Model Performance")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.4f}")
        m2.metric("Precision", f"{best_metrics.get('precision', 0):.4f}")
        m3.metric("Recall", f"{best_metrics.get('recall', 0):.4f}")
        m4.metric("F1", f"{best_metrics.get('f1', 0):.4f}")
        m5.metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.4f}")

st.markdown("---")

# --- Prediction monitoring ---
st.header("Prediction Monitoring")
predictions_df = load_predictions()

if predictions_df is None:
    st.info("No predictions logged yet. Make some requests to the /predict endpoint to see monitoring data here.")
else:
    total_predictions = len(predictions_df)
    avg_probability = predictions_df["probability"].mean()
    high_risk_count = (predictions_df["risk_level"] == "High").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", total_predictions)
    c2.metric("Avg. Default Probability", f"{avg_probability:.4f}")
    c3.metric("High-Risk Predictions", high_risk_count)

    st.subheader("Risk Level Distribution")
    risk_counts = predictions_df["risk_level"].value_counts()
    st.bar_chart(risk_counts)

st.markdown("---")

# --- Drift monitoring ---
st.header("Data Drift Monitoring")
drift_summary = load_json(DRIFT_SUMMARY_PATH)

if drift_summary is None:
    st.info(
        "No drift report generated yet. Run `python -m src.monitoring` "
        "(after logging enough predictions) to generate one."
    )
else:
    d1, d2, d3 = st.columns(3)

    drift_status = "Drift Detected" if drift_summary.get("dataset_drift_detected") else "No Drift Detected"
    d1.metric("Drift Status", drift_status)
    d2.metric(
        "Drifted Features",
        f"{drift_summary.get('drifted_features_count', 0)} / {drift_summary.get('total_features', 0)}",
    )
    d3.metric("Drift Share", f"{drift_summary.get('drift_share', 0):.2%}")

    if drift_summary.get("dataset_drift_detected"):
        st.warning(
            "Dataset drift detected. This would normally trigger the automated retraining workflow "
            "(`python -m src.retrain` or the scheduled GitHub Action)."
        )
    else:
        st.success("No significant dataset drift detected in the current monitoring window.")

    drift_html_path = Path("reports/drift/data_drift_report.html")
    if drift_html_path.exists():
        st.caption(f"Full drift report available at: `{drift_html_path}`")

st.markdown("---")
st.caption(
    "This dashboard reads live project artifacts (models/, data/monitoring/, reports/drift/). "
    "Retraining data in this demo project uses labeled/simulated data rather than true real-time "
    "outcomes, as documented in the project's technical documentation."
)