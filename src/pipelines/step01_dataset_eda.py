"""
STEP 01 - Dataset EDA

Generate automatic EDA charts without hardcoding column names.
Works for different datasets by detecting columns dynamically.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def detect_target_column(df):
    """
    Try to detect target column automatically.
    """

    candidates = [
        "Diabetes_binary",
        "diabetes_binary",
        "Outcome",
        "target",
        "label",
        "class"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def run_step01_dataset_eda(df, outputs_dir):

    eda_dir = os.path.join(outputs_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)

    print("\nSTEP 01 - DATASET EDA")
    print("--------------------------------")

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    # ----------------------------
    # Detect target column
    # ----------------------------

    target_col = detect_target_column(df)

    if target_col:
        print("Detected target column:", target_col)

    # ----------------------------
    # 1. Class distribution
    # ----------------------------

    if target_col:

        plt.figure()

        df[target_col].value_counts().plot(kind="bar")

        plt.title("Class Distribution")
        plt.xlabel(target_col)
        plt.ylabel("Count")

        plt.tight_layout()

        path = os.path.join(eda_dir, "class_distribution.png")
        plt.savefig(path)

        plt.close()

        print("Saved:", path)

    # ----------------------------
    # 2. Numeric feature distributions
    # ----------------------------

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Only plot first 5 columns to avoid too many charts
    for col in numeric_cols[:5]:

        plt.figure()

        sns.histplot(df[col], bins=30)

        plt.title(f"{col} Distribution")

        plt.tight_layout()

        path = os.path.join(eda_dir, f"{col}_distribution.png")
        plt.savefig(path)

        plt.close()

        print("Saved:", path)

    # ----------------------------
    # 3. Correlation heatmap
    # ----------------------------

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] > 2:

        plt.figure(figsize=(10, 8))

        corr = numeric_df.corr()

        sns.heatmap(corr, cmap="coolwarm", center=0)

        plt.title("Feature Correlation Heatmap")

        plt.tight_layout()

        path = os.path.join(eda_dir, "correlation_heatmap.png")
        plt.savefig(path)

        plt.close()

        print("Saved:", path)

    print("\nEDA completed\n")