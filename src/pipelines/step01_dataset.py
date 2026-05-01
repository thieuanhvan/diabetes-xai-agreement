import logging
import pandas as pd

from datasets.dataset_registry import get_active_dataset
from datasets.loader import load_active_dataset
from src.utils.config import load_config


def run_step01_load_dataset():
    """
    STEP 01
    Load dataset and inspect.
    Uses loader.load_active_dataset() so include_columns from registry
    is applied automatically (filter to subset of features).
    """

    logging.info("============================================================")
    logging.info("STEP 01 - LOAD AND INSPECT DATASET")
    logging.info("============================================================")

    df, config = load_active_dataset()

    dataset_path = config["path"]
    target = config["target"]

    # ----------------------------------------
    # FAST TEST MODE — đọc từ configs/default.yaml
    # ----------------------------------------
    # Để chạy nhanh khi dev, sửa configs/default.yaml:
    #   debug:
    #     sample_size: 2500     # nhanh nhất, ~30s/combo
    #     sample_size: 25000    # trung bình, ~3 phút/combo
    #     sample_size: null     # full dataset (production), ~3-30 phút/combo
    cfg = load_config()
    DEBUG_SAMPLE_SIZE = cfg["debug"]["sample_size"]

    if DEBUG_SAMPLE_SIZE is not None and len(df) > DEBUG_SAMPLE_SIZE:
        logging.info(f"DEBUG MODE: sampling {DEBUG_SAMPLE_SIZE} rows from {len(df)}")
        df = df.sample(n=DEBUG_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    logging.info(f"Dataset rows after sampling: {len(df)}")


    logging.info(f"Dataset config: {config}")
    logging.info(f"Active dataset : {config['slug']}")
    logging.info(f"Dataset path   : {dataset_path}")
    logging.info(f"Target column  : {target}")

    logging.info(f"Rows    : {df.shape[0]}")
    logging.info(f"Columns : {df.shape[1]}")

    logging.info(f"Columns: {list(df.columns)}")

    logging.info("\n%s", df.head())

    logging.info("Target distribution:")
    logging.info(df[target].value_counts())

    logging.info("Target ratio:")
    logging.info(df[target].value_counts(normalize=True))

    missing = df.isnull().sum()

    logging.info("Missing values:")
    if missing.sum() == 0:
        logging.info("No missing values")
    else:
        logging.info(missing[missing > 0])

    logging.info("STEP 01 COMPLETED")

    return df, config
