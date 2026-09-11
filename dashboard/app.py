"""
app.py

Streamlit monitoring dashboard for the loan default prediction platform.
Reads directly from actual project artifacts (model metadata, metrics,
prediction logs, drift summary, evaluation figures) - no hardcoded or
simulated numbers. Handles missing files gracefully.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Allow "from src.features import ..." to work when run as
# `streamlit run dashboard/app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import engineer_features  # noqa: E402

st.set_page_config(page_title="Loan Default Risk Prediction Platform", layout="wide")

METRICS_PATH = Path("models/metrics.json")
METADATA_PATH = Path("models/model_metadata.json")
MODEL_PATH = Path("models/best_model.pkl")
PREDICTIONS_LOG_PATH = Path("data/monitoring/predictions.csv")
DRIFT_SUMMARY_PATH = Path("reports/drift/data_drift_summary.json")
DRIFT_HTML_PATH = Path("reports/drift/data_drift_report.html")

FIGURES_DIR = Path("reports/figures")
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = FIGURES_DIR / "roc_curve.png"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"

LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.60


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_predictions():
    if not PREDICTIONS_LOG_PATH.exists():
        return None
    df = pd.read_csv(PREDICTIONS_LOG_PATH)
    return df if not df.empty else None


@st.cache_resource
def load_model_and_metadata():
    """Load the model once per session, matching api/main.py's approach."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


def classify_risk(probability: float) -> str:
    if probability < LOW_RISK_MAX:
        return "Low"
    if probability < MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


st.title("Loan Default Risk Prediction Platform")
st.caption("Real-time MLOps monitoring dashboard - project overview, model health, live prediction, and drift status.")

metadata = load_json(METADATA_PATH)
metrics = load_json(METRICS_PATH)

tab_overview, tab_predict, tab_monitoring, tab_performance, tab_drift = st.tabs(
    ["Overview", "Live Prediction", "Prediction Monitoring", "Model Performance", "Drift Monitoring"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.header("Project Overview")
    st.write(
        "Real-time credit default risk prediction service with an automated MLOps lifecycle: "
        "MLflow experiment tracking, Evidently drift monitoring, and drift-triggered retraining. "
        "Built on the UCI Default of Credit Card Clients dataset."
    )

    st.subheader("Current Model")
    if metadata is None:
        st.warning("No model metadata found yet. Run the training pipeline first (`python -m src.pipeline`).")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", metadata.get("model_name", "N/A"))
        col2.metric("Version", metadata.get("version", "N/A"))
        col3.metric("ROC-AUC", f"{metadata.get('roc_auc', 0):.4f}")
        training_date = metadata.get("training_date", "N/A")
        col4.metric("Last Trained", training_date.split("T")[0] if "T" in str(training_date) else training_date)

# ---------------------------------------------------------------------------
# Live Prediction
# ---------------------------------------------------------------------------
with tab_predict:
    st.header("🔮 Make a Live Prediction")
    model, model_metadata = load_model_and_metadata()

    if model is None:
        st.warning("No trained model found yet. Run `python -m src.pipeline` first.")
    else:
        st.caption("Fill in an applicant's details below to get a real-time default risk prediction from the current model.")

        with st.expander("ℹ️ What do the payment history fields mean?"):
            st.markdown(
                "- **Repayment Status** (PAY_0 → PAY_6): how the applicant paid each of the last 6 months, "
                "most recent first. `-1` = paid in full, `0` = paid the minimum on time, `1` = payment delayed "
                "1 month, `2` = delayed 2 months, and so on.\n"
                "- **Bill Amount** (BILL_AMT1 → BILL_AMT6): how much the credit card bill was that month, "
                "before any payment.\n"
                "- **Payment Amount** (PAY_AMT1 → PAY_AMT6): how much the applicant actually paid that month."
            )

        with st.form("live_prediction_form"):
            st.subheader("👤 Applicant Details")
            c1, c2, c3, c4, c5 = st.columns(5)
            limit_bal = c1.number_input("Credit Limit", min_value=1.0, value=200000.0, help="Total credit limit given to the applicant, in NT dollars.")
            sex = c2.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
            education = c3.selectbox(
                "Education", options=[1, 2, 3, 4],
                format_func=lambda x: {1: "Grad school", 2: "University", 3: "High school", 4: "Others"}[x],
            )
            marriage = c4.selectbox(
                "Marital Status", options=[1, 2, 3],
                format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x],
            )
            age = c5.number_input("Age", min_value=18, max_value=100, value=35)

            st.divider()
            st.subheader("💳 Repayment Status (most recent → 6 months ago)")
            st.caption("-1 = paid in full · 0 = paid minimum on time · 1+ = months payment was delayed")
            p1, p2, p3, p4, p5, p6 = st.columns(6)
            pay_0 = p1.number_input("This month", value=0, step=1, key="pay_0")
            pay_2 = p2.number_input("2 months ago", value=0, step=1, key="pay_2")
            pay_3 = p3.number_input("3 months ago", value=0, step=1, key="pay_3")
            pay_4 = p4.number_input("4 months ago", value=0, step=1, key="pay_4")
            pay_5 = p5.number_input("5 months ago", value=0, step=1, key="pay_5")
            pay_6 = p6.number_input("6 months ago", value=0, step=1, key="pay_6")

            st.divider()
            st.subheader("🧾 Bill Amount (what was owed each month)")
            b1, b2, b3, b4, b5, b6 = st.columns(6)
            bill_amt1 = b1.number_input("This month", value=50000.0, key="bill1")
            bill_amt2 = b2.number_input("2 months ago", value=48000.0, key="bill2")
            bill_amt3 = b3.number_input("3 months ago", value=46000.0, key="bill3")
            bill_amt4 = b4.number_input("4 months ago", value=44000.0, key="bill4")
            bill_amt5 = b5.number_input("5 months ago", value=42000.0, key="bill5")
            bill_amt6 = b6.number_input("6 months ago", value=40000.0, key="bill6")

            st.divider()
            st.subheader("💰 Payment Amount (what was actually paid each month)")
            a1, a2, a3, a4, a5, a6 = st.columns(6)
            pay_amt1 = a1.number_input("This month", min_value=0.0, value=2000.0, key="amt1")
            pay_amt2 = a2.number_input("2 months ago", min_value=0.0, value=2000.0, key="amt2")
            pay_amt3 = a3.number_input("3 months ago", min_value=0.0, value=2000.0, key="amt3")
            pay_amt4 = a4.number_input("4 months ago", min_value=0.0, value=2000.0, key="amt4")
            pay_amt5 = a5.number_input("5 months ago", min_value=0.0, value=2000.0, key="amt5")
            pay_amt6 = a6.number_input("6 months ago", min_value=0.0, value=2000.0, key="amt6")

            submitted = st.form_submit_button("🔮 Predict Default Risk", use_container_width=True)

        if submitted:
            raw_input = {
                "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
                "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
                "BILL_AMT1": bill_amt1, "BILL_AMT2": bill_amt2, "BILL_AMT3": bill_amt3,
                "BILL_AMT4": bill_amt4, "BILL_AMT5": bill_amt5, "BILL_AMT6": bill_amt6,
                "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
                "PAY_AMT4": pay_amt4, "PAY_AMT5": pay_amt5, "PAY_AMT6": pay_amt6,
            }
            raw_df = pd.DataFrame([raw_input])
            engineered_df = engineer_features(raw_df)
            engineered_df = engineered_df.reindex(columns=model_metadata["features"])

            prediction = int(model.predict(engineered_df)[0])
            probability = float(model.predict_proba(engineered_df)[0][1])
            risk_level = classify_risk(probability)

            st.divider()
            st.subheader("Result")
            r1, r2, r3 = st.columns(3)
            if prediction == 1:
                r1.error(f"**Prediction:** Default")
            else:
                r1.success(f"**Prediction:** No Default")
            r2.metric("Default Probability", f"{probability:.4f}")
            r3.metric("Risk Level", risk_level)

# ---------------------------------------------------------------------------
# Prediction Monitoring
# ---------------------------------------------------------------------------
with tab_monitoring:
    st.header("Prediction Monitoring")
    predictions_df = load_predictions()

    if predictions_df is None:
        st.info("No predictions logged yet. Make some requests to the /predict endpoint, or use the Live Prediction tab.")
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

# ---------------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------------
with tab_performance:
    st.header("Model Performance")

    if metrics:
        best_name = metrics.get("best_model")
        best_metrics = metrics.get("models", {}).get(best_name, {})
        if best_metrics:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.4f}")
            m2.metric("Precision", f"{best_metrics.get('precision', 0):.4f}")
            m3.metric("Recall", f"{best_metrics.get('recall', 0):.4f}")
            m4.metric("F1", f"{best_metrics.get('f1', 0):.4f}")
            m5.metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.4f}")
    else:
        st.info("No metrics found yet. Run the training pipeline first.")

    st.subheader("Evaluation Plots")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix")
        else:
            st.info("Confusion matrix not generated yet.")
    with img_col2:
        if ROC_CURVE_PATH.exists():
            st.image(str(ROC_CURVE_PATH), caption="ROC Curve")
        else:
            st.info("ROC curve not generated yet.")

    if FEATURE_IMPORTANCE_PATH.exists():
        st.image(str(FEATURE_IMPORTANCE_PATH), caption="Top Feature Importances")
    else:
        st.info("Feature importance plot not generated yet.")

# ---------------------------------------------------------------------------
# Drift Monitoring
# ---------------------------------------------------------------------------
with tab_drift:
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

        if DRIFT_HTML_PATH.exists():
            st.caption(f"Full drift report available at: `{DRIFT_HTML_PATH}`")

st.markdown("---")
st.caption(
    "This dashboard reads live project artifacts (models/, data/monitoring/, reports/). "
    "Retraining data in this demo project uses labeled/simulated data rather than true real-time "
    "outcomes, as documented in the project's technical documentation."
)