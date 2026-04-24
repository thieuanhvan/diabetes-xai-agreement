"""
Run model error analysis for classification datasets.

This script trains a baseline RandomForest model and analyzes
prediction errors (False Positives / False Negatives).

Outputs will be saved to:
    outputs/analysis/<dataset_name>_error_analysis.csv
"""

import pandas as pd

from utils.project_paths import OUTPUT_DIR
from datasets.dataset_registry import DATASETS
from dataset_profiler import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Output directory
ANALYSIS_OUTPUT = OUTPUT_DIR / "analysis"
ANALYSIS_OUTPUT.mkdir(parents=True, exist_ok=True)


def run_error_analysis(dataset_name):
    """
    Train model and analyze prediction errors.
    """

    print("\nRunning error analysis:", dataset_name)

    # Load dataset
    X, y, cfg = load_dataset(dataset_name)

    # Convert categorical features to numeric
    X = pd.get_dummies(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train baseline model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Create dataframe for analysis
    df = X_test.copy()
    df["true"] = y_test.values
    df["pred"] = y_pred

    # Identify prediction errors
    df["error"] = df["true"] != df["pred"]

    # Aggregate feature statistics for errors
    error_stats = df.groupby("error").mean(numeric_only=True)

    # Save results
    output_file = ANALYSIS_OUTPUT / f"{dataset_name}_error_analysis.csv"
    error_stats.to_csv(output_file)

    print("Saved:", output_file.resolve())


def main():
    """
    Run error analysis for all classification datasets.
    """

    for dataset_name, cfg in DATASETS.items():

        # Skip regression datasets
        if cfg.get("task") != "classification":
            print(f"\nSkipping regression dataset: {dataset_name}")
            continue

        run_error_analysis(dataset_name)


if __name__ == "__main__":
    main()