"""
main.py

FastAPI application exposing /health and /predict endpoints for the
loan default risk prediction model.

The model and its metadata are loaded once at startup. The exact same
feature engineering used in training (src.features.engineer_features)
is applied here, and features are reindexed to match the training
feature order exactly, to avoid training-serving skew.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from src.monitoring import log_prediction

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from src.features import engineer_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/best_model.pkl")
METADATA_PATH = Path("models/model_metadata.json")

# Risk classification thresholds
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.60

# Holds the loaded model, feature order, and version - populated at startup.
state: dict = {}


def load_model_and_metadata() -> None:
    """Load the trained model and its metadata from disk into `state`."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run the training pipeline first.")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}. Run the training pipeline first.")

    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    state["model"] = model
    state["model_version"] = metadata["version"]
    state["feature_order"] = metadata["features"]

    logger.info(
        "Loaded model '%s' (version %s) with %d expected features.",
        metadata.get("model_name"), metadata["version"], len(metadata["features"]),
    )


def classify_risk(probability: float) -> str:
    """Map a default probability to a risk category."""
    if probability < LOW_RISK_MAX:
        return "Low"
    if probability < MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading model and metadata...")
    try:
        load_model_and_metadata()
    except Exception:
        logger.exception("Failed to load model/metadata at startup.")
        raise
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Loan Default Risk Prediction API",
    description="Real-time credit default risk prediction service, part of an MLOps demo platform.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Basic health check, also confirms the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "model_version": state.get("model_version"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict default risk for a single applicant."""
    try:
        raw_df = pd.DataFrame([request.model_dump()])
        engineered_df = engineer_features(raw_df)

        # Reindex to the exact column order/set the model was trained on,
        # to guard against any schema drift between training and serving.
        engineered_df = engineered_df.reindex(columns=state["feature_order"])

        model = state["model"]
        prediction = int(model.predict(engineered_df)[0])
        probability = float(model.predict_proba(engineered_df)[0][1])
        risk_level = classify_risk(probability)

        log_prediction(
            raw_input=request.model_dump(),
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            model_version=state["model_version"],
        )

        logger.info(
            "Prediction made: prediction=%d, probability=%.4f, risk_level=%s",
            prediction, probability, risk_level,
        )
        

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            model_version=state["model_version"],
        )

    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")