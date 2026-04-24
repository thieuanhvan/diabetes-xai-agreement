"""
run_statistical_tests_basic.py

Flexible statistical analysis reproduction:

- Continuous features:
    + If defined in dataset_registry → use that
    + Else auto-detect numeric columns
    + Or allow manual override

- Categorical features:
    + Everything else

Implements:
- Student's t-test for continuous
- Chi-square for categorical
"""

import logging
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

from src.datasets.dataset_registry import DATASETS, ACTIVE_DATASET
from utils.project_paths import get_outputs_dir, get_log_file


# =========================================================
# OPTIONAL: Manual override for continuous features
# If None → auto detect / registry detect
# Example:
# MANUAL_CONTINUOUS = ["Age"]
# =========================================================
MANUAL_CONTINUOUS = None


# =========================================================
# Logging
# =========================================================
def setup_logging():

    log_file = get_log_file("statistical_basic")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 70)
    logging.info("STATISTICAL TESTS STARTED")
    logging.info("=" * 70)


# =========================================================
# Detect continuous columns
# =========================================================
def detect_continuous_columns(df, dataset_config):

    # 1️⃣ Manual override
    if MANUAL_CONTINUOUS is not None:
        logging.info("Using MANUAL_CONTINUOUS definition.")
        return MANUAL_CONTINUOUS

    # 2️⃣ If registry defines numeric_features
    if "numeric_features" in dataset_config:
        logging.info("Using numeric_features from dataset_registry.")
        return dataset_config["numeric_features"]

    # 3️⃣ Auto-detect numeric columns
    logging.info("Auto-detecting numeric columns as continuous.")
    return [
        col for col in df.columns
        if np.issubdtype(df[col].dtype, np.number)
    ]


# =========================================================
# Main
# =========================================================
def main():

    setup_logging()

    dataset_config = DATASETS[ACTIVE_DATASET]
    dataset_path = dataset_config["path"]
    target_col = dataset_config["target"]

    logging.info(f"Active dataset: {ACTIVE_DATASET}")
    logging.info(f"Path: {dataset_path}")

    df = pd.read_csv(dataset_path)
    logging.info(f"Dataset shape: {df.shape}")

    continuous_cols = detect_continuous_columns(df, dataset_config)

    # Remove target if accidentally included
    continuous_cols = [c for c in continuous_cols if c != target_col]

    logging.info(f"Continuous features: {continuous_cols}")

    descriptive_results = []
    ttest_results = []
    chi_results = []

    for col in df.columns:

        if col == target_col:
            continue

        # --------------------------------------------------
        # Continuous → T-test
        # --------------------------------------------------
        if col in continuous_cols:

            group_pos = df[df[target_col] == 1][col]
            group_neg = df[df[target_col] == 0][col]

            descriptive_results.append({
                "feature": col,
                "type": "continuous",
                "mean_positive": group_pos.mean(),
                "std_positive": group_pos.std(),
                "mean_negative": group_neg.mean(),
                "std_negative": group_neg.std()
            })

            if len(group_pos) > 1 and len(group_neg) > 1:
                t_stat, p_value = ttest_ind(
                    group_pos,
                    group_neg,
                    equal_var=False
                )

                logging.info(f"T-test {col} | p-value = {p_value:.6f}")

                ttest_results.append({
                    "feature": col,
                    "t_statistic": t_stat,
                    "p_value": p_value
                })

        # --------------------------------------------------
        # Categorical → Chi-square
        # --------------------------------------------------
        else:

            contingency = pd.crosstab(df[col], df[target_col])

            descriptive_results.append({
                "feature": col,
                "type": "categorical",
                "levels": df[col].nunique()
            })

            if contingency.shape[0] > 1:
                chi2, p_value, dof, expected = chi2_contingency(contingency)

                logging.info(f"Chi-square {col} | p-value = {p_value:.6f}")

                chi_results.append({
                    "feature": col,
                    "chi2_statistic": chi2,
                    "p_value": p_value,
                    "degrees_of_freedom": dof
                })

    # Warnings if empty
    if len(ttest_results) == 0:
        logging.warning("No continuous features found. T-test not performed.")

    if len(chi_results) == 0:
        logging.warning("No categorical features found. Chi-square not performed.")

    output_dir = get_outputs_dir("analysis/statistical_basic")

    pd.DataFrame(descriptive_results).to_csv(
        output_dir / "descriptive_statistics.csv",
        index=False
    )

    pd.DataFrame(ttest_results).to_csv(
        output_dir / "t_test_results.csv",
        index=False
    )

    pd.DataFrame(chi_results).to_csv(
        output_dir / "chi_square_results.csv",
        index=False
    )

    logging.info("Saved descriptive_statistics.csv")
    logging.info("Saved t_test_results.csv")
    logging.info("Saved chi_square_results.csv")

    logging.info("=" * 70)
    logging.info("STATISTICAL TESTS COMPLETED SUCCESSFULLY.")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()