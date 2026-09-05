"""
preprocess.py

Loads the raw dataset, cleans it, and produces a stratified
train/test split saved under data/processed/.
No feature engineering happens here - that's features.py's job.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/credit_default.csv")
PROCESSED_DATA_DIR = Path("data/processed")

TARGET_COL = "default"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(input_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    try:
        df = pd.read_csv(input_path)
        logger.info("Loaded raw data from %s with shape %s", input_path, df.shape)
        return df
    except Exception:
        logger.exception("Failed to load raw data from %s", input_path)
        raise


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, if any."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        logger.info("Removed %d duplicate rows (%d -> %d).", before - after, before, after)
    else:
        logger.info("No duplicate rows found.")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Log and return a count of missing values per column."""
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        logger.info("Missing values found:\n%s", missing[missing > 0])
    else:
        logger.info("No missing values found.")
    return missing


def fix_invalid_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle undocumented category codes in EDUCATION and MARRIAGE.

    UCI documentation defines:
      EDUCATION: 1=grad school, 2=university, 3=high school, 4=others
      MARRIAGE:  1=married, 2=single, 3=others

    In practice the data contains extra codes (0, 5, 6 for EDUCATION;
    0 for MARRIAGE) that aren't documented. Strategy: fold all
    undocumented codes into the existing "others" category rather
    than dropping rows, so we don't lose data.
    """
    df = df.copy()

    if "EDUCATION" in df.columns:
        invalid_edu = ~df["EDUCATION"].isin([1, 2, 3, 4])
        n_invalid = invalid_edu.sum()
        if n_invalid > 0:
            logger.info("Remapping %d undocumented EDUCATION codes to 4 (others).", n_invalid)
            df.loc[invalid_edu, "EDUCATION"] = 4

    if "MARRIAGE" in df.columns:
        invalid_marriage = ~df["MARRIAGE"].isin([1, 2, 3])
        n_invalid = invalid_marriage.sum()
        if n_invalid > 0:
            logger.info("Remapping %d undocumented MARRIAGE codes to 3 (others).", n_invalid)
            df.loc[invalid_marriage, "MARRIAGE"] = 3

    return df


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Separate features (X) and target (y)."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info("Split into X %s and y %s.", X.shape, y.shape)
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series,
                      test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Train/test split -> X_train: %s, X_test: %s, y_train: %s, y_test: %s",
        X_train.shape, X_test.shape, y_train.shape, y_test.shape,
    )
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test,
                         output_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Save the split datasets to CSV, creating directories as needed."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(output_dir / "X_train.csv", index=False)
        X_test.to_csv(output_dir / "X_test.csv", index=False)
        y_train.to_csv(output_dir / "y_train.csv", index=False)
        y_test.to_csv(output_dir / "y_test.csv", index=False)
        logger.info("Saved processed train/test files to %s", output_dir)
    except Exception:
        logger.exception("Failed to save processed data to %s", output_dir)
        raise


def run_preprocessing() -> None:
    """Full preprocessing workflow, orchestrating the functions above."""
    df = load_data()
    df = remove_duplicates(df)
    check_missing_values(df)
    df = fix_invalid_categories(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    save_processed_data(X_train, X_test, y_train, y_test)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()