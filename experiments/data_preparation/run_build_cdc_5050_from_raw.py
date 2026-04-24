"""
Rebuild CDC BRFSS 2021 Diabetes Dataset (50/50 Balanced)

This script reconstructs a balanced diabetes dataset from the original
CDC BRFSS dataset (~236k records).

Pipeline:

1. Load raw dataset
2. Remove prediabetes (class = 1)
3. Convert labels to binary
4. Random downsampling to balance classes
5. Save rebuilt dataset
6. Save detailed summary statistics

Author: ML Pipeline Healthcare Project
"""

import os
import json
import pandas as pd


# =========================================================
# DETECT PROJECT ROOT
# =========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

print("Project root:", PROJECT_ROOT)


# =========================================================
# FILE PATHS
# =========================================================

RAW_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "cdc_diabetes_012_health_indicators_BRFSS2021.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "cdc_brfss_2021_diabetes_5050_rebuilt.csv"
)

SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "cdc_brfss_2021_diabetes_5050_rebuilt_summary.json"
)

RANDOM_STATE = 42


# =========================================================
# MAIN
# =========================================================

def main():

    print("===================================================")
    print(" CDC BRFSS 2021 DATASET REBUILD (50/50 BALANCED) ")
    print("===================================================")

    # ----------------------------------------------------
    # STEP 1 — LOAD DATASET
    # ----------------------------------------------------

    print("\n[STEP 1] Loading raw dataset")

    if not os.path.exists(RAW_FILE):

        print("ERROR: Raw dataset not found")
        print("Expected path:", RAW_FILE)
        return

    df = pd.read_csv(RAW_FILE)

    original_rows = len(df)
    original_features = len(df.columns)

    print("Dataset loaded successfully")
    print("Rows:", original_rows)
    print("Columns:", original_features)

    # ----------------------------------------------------
    # STEP 2 — ORIGINAL CLASS DISTRIBUTION
    # ----------------------------------------------------

    print("\n[STEP 2] Original class distribution")

    original_distribution = df["Diabetes_012"].value_counts().to_dict()

    print(df["Diabetes_012"].value_counts())

    # ----------------------------------------------------
    # STEP 3 — REMOVE PREDIABETES
    # ----------------------------------------------------

    print("\n[STEP 3] Removing prediabetes (class = 1)")

    df = df[df["Diabetes_012"] != 1]

    rows_after_removal = len(df)

    after_removal_distribution = df["Diabetes_012"].value_counts().to_dict()

    print("Remaining rows:", rows_after_removal)

    print("\nClass distribution after removal")

    print(df["Diabetes_012"].value_counts())

    # ----------------------------------------------------
    # STEP 4 — CONVERT TO BINARY
    # ----------------------------------------------------

    print("\n[STEP 4] Converting labels to binary")

    df["diabetes_binary"] = df["Diabetes_012"].map({
        0: 0,
        2: 1
    })

    df = df.drop(columns=["Diabetes_012"])

    binary_distribution = df["diabetes_binary"].value_counts().to_dict()

    print("\nBinary class distribution")

    print(df["diabetes_binary"].value_counts())

    # ----------------------------------------------------
    # STEP 5 — BALANCE DATASET
    # ----------------------------------------------------

    print("\n[STEP 5] Balancing dataset")

    df_class0 = df[df["diabetes_binary"] == 0]
    df_class1 = df[df["diabetes_binary"] == 1]

    class0_size = len(df_class0)
    class1_size = len(df_class1)

    print("Class 0:", class0_size)
    print("Class 1:", class1_size)

    imbalance_ratio = round(class0_size / class1_size, 2)

    print("Imbalance ratio:", imbalance_ratio, ": 1")

    print("\nDownsampling class 0")

    df_class0_sample = df_class0.sample(
        n=class1_size,
        random_state=RANDOM_STATE
    )

    df_balanced = pd.concat([
        df_class0_sample,
        df_class1
    ])

    df_balanced = df_balanced.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    final_rows = len(df_balanced)

    final_distribution = df_balanced["diabetes_binary"].value_counts().to_dict()

    print("\nBalanced dataset size:", final_rows)

    print("\nBalanced distribution")

    print(df_balanced["diabetes_binary"].value_counts())

    # ----------------------------------------------------
    # STEP 6 — SAVE DATASET
    # ----------------------------------------------------

    print("\n[STEP 6] Saving dataset")

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df_balanced.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Dataset saved:", OUTPUT_FILE)

    # ----------------------------------------------------
    # STEP 7 — SAVE DETAILED SUMMARY
    # ----------------------------------------------------

    print("\n[STEP 7] Saving detailed summary")

    summary = {

        "dataset_name": "CDC BRFSS 2021 Diabetes (Rebuilt 50/50)",

        "original_dataset": {
            "rows": int(original_rows),
            "features": int(original_features)
        },

        "original_class_distribution": original_distribution,

        "after_remove_prediabetes": {
            "rows": int(rows_after_removal),
            "class_distribution": after_removal_distribution
        },

        "binary_conversion": {
            "0": "no_diabetes",
            "1": "diabetes"
        },

        "class_imbalance_before_balancing": {
            "class0_no_diabetes": int(class0_size),
            "class1_diabetes": int(class1_size),
            "imbalance_ratio": imbalance_ratio
        },

        "balancing_method": "random_downsampling",

        "random_state": RANDOM_STATE,

        "final_dataset": {
            "rows": int(final_rows),
            "class_distribution": final_distribution
        }

    }

    with open(SUMMARY_FILE, "w") as f:

        json.dump(summary, f, indent=4)

    print("Summary saved:", SUMMARY_FILE)

    # ----------------------------------------------------

    print("\n===================================================")
    print(" DATASET REBUILD COMPLETED ")
    print("===================================================")

    print("\nOutput dataset:")
    print(OUTPUT_FILE)

    print("\nSummary file:")
    print(SUMMARY_FILE)


# =========================================================

if __name__ == "__main__":

    main()