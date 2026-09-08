"""
Tests for the FastAPI app.

Requires models/best_model.pkl and models/model_metadata.json to
already exist (run python -m src.pipeline at least once before running
this file) - the API loads the real trained model at startup, same as
production.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """
    Yields a TestClient inside a 'with' block, so FastAPI's lifespan
    startup (which loads the model into app state) actually runs.
    """
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_payload_returns_expected_structure(client, valid_predict_payload):
    response = client.post("/predict", json=valid_predict_payload)
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert "probability" in body
    assert "risk_level" in body
    assert "model_version" in body

    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_predict_missing_field_returns_422(client, valid_predict_payload):
    incomplete_payload = valid_predict_payload.copy()
    del incomplete_payload["LIMIT_BAL"]

    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_invalid_education_value_returns_422(client, valid_predict_payload):
    bad_payload = valid_predict_payload.copy()
    bad_payload["EDUCATION"] = 99  # outside allowed range 1-4

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422