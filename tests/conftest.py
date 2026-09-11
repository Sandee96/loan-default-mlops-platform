# tests/conftest.py
"""
Shared pytest fixtures for the test suite. Uses small synthetic
DataFrames so tests run fast and don't require rerunning the full
training pipeline.
"""

import pandas as pd
import pytest


@pytest.fixture
def raw_feature_columns():
    """Columns required before feature engineering, matching preprocess.py output."""
    return [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]


@pytest.fixture
def sample_raw_df(raw_feature_columns):
    """A small synthetic DataFrame with valid raw feature values."""
    data = {
        "LIMIT_BAL": [200000, 50000, 350000],
        "SEX": [1, 2, 1],
        "EDUCATION": [2, 1, 3],
        "MARRIAGE": [1, 2, 1],
        "AGE": [35, 28, 45],
        "PAY_0": [0, -1, 2],
        "PAY_2": [0, -1, 2],
        "PAY_3": [0, -1, 0],
        "PAY_4": [0, -1, 0],
        "PAY_5": [0, -1, 0],
        "PAY_6": [0, -1, 0],
        "BILL_AMT1": [50000, 1000, 0],
        "BILL_AMT2": [48000, 900, 0],
        "BILL_AMT3": [46000, 800, 0],
        "BILL_AMT4": [44000, 700, 0],
        "BILL_AMT5": [42000, 600, 0],
        "BILL_AMT6": [40000, 500, 0],
        "PAY_AMT1": [2000, 100, 0],
        "PAY_AMT2": [2000, 100, 0],
        "PAY_AMT3": [2000, 100, 0],
        "PAY_AMT4": [2000, 100, 0],
        "PAY_AMT5": [2000, 100, 0],
        "PAY_AMT6": [2000, 100, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_raw_df_with_dirty_categories(sample_raw_df):
    """Same sample data but with undocumented EDUCATION/MARRIAGE codes injected."""
    df = sample_raw_df.copy()
    df.loc[0, "EDUCATION"] = 0   # undocumented code
    df.loc[1, "EDUCATION"] = 6   # undocumented code
    df.loc[0, "MARRIAGE"] = 0    # undocumented code
    return df


@pytest.fixture
def sample_target():
    """Matching target values for sample_raw_df."""
    return pd.Series([0, 1, 0], name="default")


@pytest.fixture
def valid_predict_payload():
    """A single valid /predict request payload."""
    return {
        "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 50000, "BILL_AMT2": 48000, "BILL_AMT3": 46000,
        "BILL_AMT4": 44000, "BILL_AMT5": 42000, "BILL_AMT6": 40000,
        "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
        "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000,
    }