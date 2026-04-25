"""
Standalone fix script: regenerate RandomForest Permutation FI only.

Output:
    outputs/<DATASET_SLUG>/fi/random_forest_feature_importance.csv
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_SLUG = "cdc_brfss_2015_rebuilt"
TARGET_COL = "Diabetes_binary"
MODEL_NAME = "random_forest"

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / f"{DATASET_SLUG}.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / DATASET_SLUG / "fi"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_REPEATS = 10
SCORING = "roc_auc"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    logging.info("============================================================")
    logging.info("STANDALONE FIX: RANDOM FOREST FEATURE IMPORTANCE")
    logging.info("============================================================")

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Dataset slug: %s", DATASET_SLUG)
    logging.info("Dataset path: %s", DATASET_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = pd.read_csv(DATASET_PATH)
    logging.info("Dataset shape: %s", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    logging.info("X shape: %s", X.shape)
    logging.info("y distribution:\n%s", y.value_counts().sort_index())

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    logging.info("Train size: %d", len(X_train))
    logging.info("Test size : %d", len(X_test))

    # --------------------------------------------------------
    # Train RandomForest
    # --------------------------------------------------------
    logging.info("Training RandomForest")

    rf_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    t0 = time.time()
    rf_model.fit(X_train, y_train)
    train_time = time.time() - t0

    logging.info("RandomForest trained in %.2f seconds", train_time)

    # --------------------------------------------------------
    # Permutation Importance
    # --------------------------------------------------------
    logging.info(
        "Running Permutation Importance: scoring=%s, n_repeats=%d, n_jobs=1",
        SCORING,
        N_REPEATS,
    )

    t0 = time.time()
    result = permutation_importance(
        rf_model,
        X_test,
        y_test,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        scoring=SCORING,
        n_jobs=1,  # Important: avoid Windows/PyCharm multiprocessing pickle issue.
    )
    fi_time = time.time() - t0

    importances = np.abs(result.importances_mean)

    fi_df = (
        pd.DataFrame(
            {
                "feature": X.columns.astype(str),
                "importance": importances,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    output_path = OUTPUT_DIR / f"{MODEL_NAME}_feature_importance.csv"
    fi_df.to_csv(output_path, index=False)

    logging.info("============================================================")
    logging.info("DONE")
    logging.info("FI rows: %d", len(fi_df))
    logging.info("FI time: %.2f seconds", fi_time)
    logging.info("Saved to: %s", output_path)
    logging.info("============================================================")


if __name__ == "__main__":
    main()