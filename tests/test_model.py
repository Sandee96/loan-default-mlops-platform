"""
Tests for model training/prediction behavior.

Trains a small model on synthetic data directly in the test - does not
depend on the full data pipeline or models/best_model.pkl existing.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import engineer_features


def _build_small_model_and_data(sample_raw_df):
    # Repeat the tiny fixture to give the model enough rows to fit on.
    X = pd.concat([sample_raw_df] * 10, ignore_index=True)
    y = pd.Series([0, 1, 0] * 10)

    X_engineered = engineer_features(X)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_engineered, y)
    return model, X_engineered


def test_model_predicts_without_error(sample_raw_df):
    model, X_engineered = _build_small_model_and_data(sample_raw_df)
    predictions = model.predict(X_engineered)
    assert len(predictions) == len(X_engineered)


def test_model_predictions_are_binary(sample_raw_df):
    model, X_engineered = _build_small_model_and_data(sample_raw_df)
    predictions = model.predict(X_engineered)
    assert set(predictions).issubset({0, 1})


def test_model_probabilities_between_zero_and_one(sample_raw_df):
    model, X_engineered = _build_small_model_and_data(sample_raw_df)
    probabilities = model.predict_proba(X_engineered)[:, 1]

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)