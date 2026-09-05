"""
features.py

Reusable feature engineering functions for the six-month payment
history in the credit default dataset. These functions must be used
identically during training and inference to avoid training-serving
skew.

No model training happens here.
"""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Column groups, ordered from most recent (index 0) to oldest.
# Note: the dataset skips "PAY_1" - this is a known quirk of the
# original UCI dataset, not a bug.
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

EPSILON = 1e-6  # avoids division by zero without materially skewing ratios


def _check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for feature engineering: {missing}")


def add_pay_status_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add AVG_PAY_STATUS and MAX_PAY_STATUS from the 6-month payment status history."""
    df = df.copy()
    _check_required_columns(df, PAY_STATUS_COLS)

    df["AVG_PAY_STATUS"] = df[PAY_STATUS_COLS].mean(axis=1)
    df["MAX_PAY_STATUS"] = df[PAY_STATUS_COLS].max(axis=1)

    logger.info("Added AVG_PAY_STATUS and MAX_PAY_STATUS.")
    return df


def add_bill_and_payment_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add AVG_BILL_AMT, AVG_PAY_AMT, and PAYMENT_TO_BILL_RATIO."""
    df = df.copy()
    _check_required_columns(df, BILL_AMT_COLS + PAY_AMT_COLS)

    df["AVG_BILL_AMT"] = df[BILL_AMT_COLS].mean(axis=1)
    df["AVG_PAY_AMT"] = df[PAY_AMT_COLS].mean(axis=1)

    # Ratio of what they actually paid vs. what they were billed, on average.
    # Guard against division by zero / negative-or-zero bills.
    denom = df["AVG_BILL_AMT"].abs() + EPSILON
    df["PAYMENT_TO_BILL_RATIO"] = df["AVG_PAY_AMT"] / denom

    logger.info("Added AVG_BILL_AMT, AVG_PAY_AMT, and PAYMENT_TO_BILL_RATIO.")
    return df


def add_credit_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Add CREDIT_UTILIZATION = average bill amount relative to credit limit."""
    df = df.copy()
    _check_required_columns(df, BILL_AMT_COLS + ["LIMIT_BAL"])

    if "AVG_BILL_AMT" not in df.columns:
        df["AVG_BILL_AMT"] = df[BILL_AMT_COLS].mean(axis=1)

    denom = df["LIMIT_BAL"].abs() + EPSILON
    df["CREDIT_UTILIZATION"] = df["AVG_BILL_AMT"] / denom

    logger.info("Added CREDIT_UTILIZATION.")
    return df


def add_payment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PAYMENT_TREND: the slope of PAY_AMT over the 6-month window,
    oldest -> most recent. Positive = paying more over time,
    negative = paying less over time.
    """
    df = df.copy()
    _check_required_columns(df, PAY_AMT_COLS)

    # PAY_AMT_COLS is ordered most-recent-first; reverse to oldest-first
    # so the slope has a clear "time moving forward" meaning.
    ordered_cols = list(reversed(PAY_AMT_COLS))
    x = np.arange(len(ordered_cols))

    def _slope(row: pd.Series) -> float:
        y = row[ordered_cols].to_numpy(dtype=float)
        if np.all(y == y[0]):  # no variation -> undefined trend, treat as 0
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    df["PAYMENT_TREND"] = df.apply(_slope, axis=1)

    logger.info("Added PAYMENT_TREND.")
    return df


def _sanitize(df: pd.DataFrame, engineered_cols: list[str]) -> pd.DataFrame:
    """Replace any inf/-inf with 0 and fill any resulting NaNs with 0, for engineered columns only."""
    df = df.copy()
    df[engineered_cols] = df[engineered_cols].replace([np.inf, -np.inf], 0)
    df[engineered_cols] = df[engineered_cols].fillna(0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Applies all engineered features
    on top of the original columns (originals are preserved, not dropped).
    Must be called identically at training time and inference time.
    """
    df = add_pay_status_features(df)
    df = add_bill_and_payment_amount_features(df)
    df = add_credit_utilization(df)
    df = add_payment_trend(df)

    engineered_cols = [
        "AVG_PAY_STATUS", "MAX_PAY_STATUS",
        "AVG_BILL_AMT", "AVG_PAY_AMT", "PAYMENT_TO_BILL_RATIO",
        "CREDIT_UTILIZATION", "PAYMENT_TREND",
    ]
    df = _sanitize(df, engineered_cols)

    logger.info("Feature engineering complete. New shape: %s", df.shape)
    return df


if __name__ == "__main__":
    # Quick manual smoke test using processed training data, if present.
    from pathlib import Path

    x_train_path = Path("data/processed/X_train.csv")
    if x_train_path.exists():
        sample = pd.read_csv(x_train_path)
        result = engineer_features(sample)
        logger.info("Sample engineered columns:\n%s", result.head())
    else:
        logger.warning("data/processed/X_train.csv not found - run preprocess.py first.")