"""
FCM Reproduction Experiment (Active Dataset)

This script reproduces the Fuzzy Cognitive Map (FCM) experiment
similar to the method described in Hoyos et al.

It loads the ACTIVE_DATASET defined in dataset_registry.py,
computes correlation-based FCM weights, and produces:

outputs/fcm/
    fcm_network_graph.png
    fcm_feature_importance.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))

# ---------------------------------------------------------
# Imports from project
# ---------------------------------------------------------

from src.datasets.dataset_registry import DATASETS, ACTIVE_DATASET
from experiments.fcm.builder import build_correlation_fcm_weights
from experiments.fcm.visualizer import plot_fcm_radial_graph


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fcm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.2


# ---------------------------------------------------------
# Dataset loader (using dataset_registry)
# ---------------------------------------------------------

def load_active_dataset():

    if ACTIVE_DATASET not in DATASETS:
        raise Exception(f"ACTIVE_DATASET '{ACTIVE_DATASET}' not found in dataset_registry")

    dataset_config = DATASETS[ACTIVE_DATASET]

    dataset_path = PROJECT_ROOT / dataset_config["path"]
    target_column = dataset_config["target"]

    print("Active dataset:", ACTIVE_DATASET)
    print("Dataset path:", dataset_path)

    if not dataset_path.exists():
        raise Exception(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    return df, target_column


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def run():

    print("Running FCM reproduction experiment")

    df, TARGET_COLUMN = load_active_dataset()

    if TARGET_COLUMN not in df.columns:
        raise Exception(f"Target column '{TARGET_COLUMN}' not found in dataset")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    feature_names = list(X.columns)

    # -----------------------------------------------------
    # Standardize features
    # -----------------------------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # combine features + target
    X_full = np.column_stack([X_scaled, y.values])

    node_names = feature_names + ["target"]

    # -----------------------------------------------------
    # Build FCM weight matrix
    # -----------------------------------------------------

    W = build_correlation_fcm_weights(X_full)

    target_index = len(node_names) - 1

    # -----------------------------------------------------
    # Compute feature importance
    # -----------------------------------------------------

    importance_list = []

    for i, feature in enumerate(feature_names):

        weight = W[i, target_index]

        importance_list.append(
            (feature, abs(weight))
        )

    importance_df = pd.DataFrame(
        importance_list,
        columns=["feature", "importance"]
    ).sort_values(
        "importance",
        ascending=False
    )

    importance_file = OUTPUT_DIR / "fcm_feature_importance.csv"

    importance_df.to_csv(
        importance_file,
        index=False
    )

    print("Saved feature importance:", importance_file)

    # -----------------------------------------------------
    # Plot FCM graph
    # -----------------------------------------------------

    graph_file = OUTPUT_DIR / "fcm_network_graph.png"

    plot_fcm_radial_graph(
        W,
        node_names,
        graph_file,
        threshold=THRESHOLD
    )

    print("Saved FCM graph:", graph_file)

    print("FCM experiment finished successfully")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run()