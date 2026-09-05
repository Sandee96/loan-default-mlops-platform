"""
data_loader.py

Fetches the UCI Default of Credit Card Clients dataset (ID 350) and
saves it as a single combined CSV under data/raw/.
No preprocessing or model training happens here.
"""

import logging
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_PATH = RAW_DATA_DIR / "credit_default.csv"
DATASET_ID = 350


def fetch_dataset(dataset_id: int = DATASET_ID) -> pd.DataFrame:
    """
    Fetch the UCI dataset by ID, combine features and target
    into a single DataFrame, and rename the target column to 'default'.
    """
    try:
        logger.info("Fetching dataset with ID %s from UCI repository...", dataset_id)
        dataset = fetch_ucirepo(id=dataset_id)

        X = dataset.data.features
        y = dataset.data.targets

        logger.info("Fetched features shape: %s, target shape: %s", X.shape, y.shape)

        df = pd.concat([X, y], axis=1)

        # The UCI target column name varies; grab whatever the last column is
        # (the target) and rename it explicitly to 'default'.
        target_col = y.columns[0]
        df = df.rename(columns={target_col: "default"})

        logger.info("Combined DataFrame shape: %s", df.shape)
        return df

    except Exception:
        logger.exception("Failed to fetch or combine the dataset.")
        raise


def save_raw_data(df: pd.DataFrame, output_path: Path = RAW_DATA_PATH) -> None:
    """Save the raw combined DataFrame to CSV, creating directories as needed."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved raw dataset to %s", output_path)
    except Exception:
        logger.exception("Failed to save raw dataset to %s", output_path)
        raise


def load_raw_data(input_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the previously saved raw dataset from CSV."""
    try:
        df = pd.read_csv(input_path)
        logger.info("Loaded raw dataset from %s with shape %s", input_path, df.shape)
        return df
    except Exception:
        logger.exception("Failed to load raw dataset from %s", input_path)
        raise


def main() -> None:
    df = fetch_dataset()
    save_raw_data(df)
    logger.info("Data loading complete. Preview:\n%s", df.head())


if __name__ == "__main__":
    main()