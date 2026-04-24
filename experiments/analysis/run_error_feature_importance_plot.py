"""
Plot top features associated with model prediction errors.

This script reads error analysis results and generates
bar charts showing features most associated with prediction errors.

Input:
    outputs/analysis/<dataset>_error_analysis.csv

Output:
    outputs/analysis/plots/<dataset>_error_feature_importance.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils.project_paths import OUTPUT_DIR
from datasets.dataset_registry import DATASETS


ANALYSIS_DIR = OUTPUT_DIR / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_error_importance(dataset_name):

    csv_file = ANALYSIS_DIR / f"{dataset_name}_error_analysis.csv"

    if not csv_file.exists():
        print(f"Skipping {dataset_name} (no error analysis file)")
        return

    print("Generating error feature plot:", dataset_name)

    df = pd.read_csv(csv_file, index_col=0)

    if df.shape[0] < 2:
        print("Not enough rows for analysis.")
        return

    correct = df.loc[False]
    errors = df.loc[True]

    diff = (errors - correct).abs()

    diff = diff.sort_values(ascending=False)

    top_features = diff.head(10)

    plt.figure(figsize=(8,5))

    top_features[::-1].plot(kind="barh")

    plt.title(f"Top Features Associated with Prediction Errors\n{dataset_name}")
    plt.xlabel("Absolute Difference")
    plt.ylabel("Feature")

    plt.tight_layout()

    output_file = PLOTS_DIR / f"{dataset_name}_error_feature_importance.png"

    plt.savefig(output_file)
    plt.close()

    print("Saved:", output_file.resolve())


def main():

    for dataset_name, cfg in DATASETS.items():

        if cfg.get("task") != "classification":
            continue

        plot_error_importance(dataset_name)


if __name__ == "__main__":
    main()