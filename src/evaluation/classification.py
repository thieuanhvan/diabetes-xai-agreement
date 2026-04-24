"""
Classification evaluation utilities.

This module provides:
- model evaluation metrics
- confusion matrix plotting
- results aggregation for visualization
"""

import os
import logging
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.utils.project_paths import get_outputs_dir


# ---------------------------------------------------------
# Evaluate classification pipeline
# ---------------------------------------------------------

def evaluate_classification_pipeline(pipeline, X_test, y_test, model_name):

    """
    Evaluate a trained classification pipeline.

    Returns a metrics dictionary used by later pipeline steps.
    """

    logging.info(f"Evaluating model: {model_name}")

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = np.nan

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # -----------------------------------------------------
    # Save confusion matrix plot
    # -----------------------------------------------------

    save_confusion_matrix(cm, model_name)

    # -----------------------------------------------------
    # Return metrics
    # -----------------------------------------------------

    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity
    }

    return metrics


# ---------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------

def save_confusion_matrix(cm, model_name):

    output_dir = get_outputs_dir("confusion_matrix")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(path)
    plt.close()

    logging.info(f"Saved confusion matrix: {path}")


# ---------------------------------------------------------
# Combine results for visualization
# ---------------------------------------------------------

def concatenate_results(baseline_results, proposed_results):

    """
    Combine baseline and proposed results into a single DataFrame.
    """

    records = []

    for r in baseline_results:
        records.append(r)

    if isinstance(proposed_results, dict):
        records.append(proposed_results)

    elif isinstance(proposed_results, list):
        for r in proposed_results:
            records.append(r)

    df = pd.DataFrame(records)

    return df