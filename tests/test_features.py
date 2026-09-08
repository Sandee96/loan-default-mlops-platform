"""Tests for src/features.py - feature engineering correctness."""

import numpy as np

from src.features import engineer_features


def test_engineer_features_adds_expected_columns(sample_raw_df):
    result = engineer_features(sample_raw_df)

    expected_new_cols = {
        "AVG_PAY_STATUS", "MAX_PAY_STATUS",
        "AVG_BILL_AMT", "AVG_PAY_AMT", "PAYMENT_TO_BILL_RATIO",
        "CREDIT_UTILIZATION", "PAYMENT_TREND",
    }
    assert expected_new_cols.issubset(set(result.columns))


def test_engineer_features_preserves_original_columns(sample_raw_df):
    original_cols = set(sample_raw_df.columns)
    result = engineer_features(sample_raw_df)
    assert original_cols.issubset(set(result.columns))


def test_engineer_features_no_nan_or_inf(sample_raw_df):
    result = engineer_features(sample_raw_df)
    numeric_result = result.select_dtypes(include=[np.number])

    assert not numeric_result.isnull().values.any(), "Found NaN values in engineered features"
    assert not np.isinf(numeric_result.values).any(), "Found infinite values in engineered features"


def test_engineer_features_handles_zero_bill_amount(sample_raw_df):
    """Row 3 in the fixture has BILL_AMT all zero - should not raise or produce inf."""
    result = engineer_features(sample_raw_df)
    last_row = result.iloc[2]

    assert np.isfinite(last_row["CREDIT_UTILIZATION"])
    assert np.isfinite(last_row["PAYMENT_TO_BILL_RATIO"])


def test_engineer_features_missing_column_raises(sample_raw_df):
    broken_df = sample_raw_df.drop(columns=["PAY_0"])
    try:
        engineer_features(broken_df)
        assert False, "Expected KeyError for missing required column"
    except KeyError:
        pass