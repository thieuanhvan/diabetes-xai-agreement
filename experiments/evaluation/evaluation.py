"""
Evaluation utilities.
Supports both classification and regression tasks.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, task="classification"):
    """
    Evaluate trained pipelines model.

    Parameters
    ----------
    model : fitted sklearn pipelines
    X_test : test features
    y_test : true labels
    task : "classification" or "regression"

    Returns
    -------
    dict of evaluation metrics
    """
    y_pred = model.predict(X_test)

    if task == "classification":
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        # Try ROC-AUC if probability exists
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                results["roc_auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            pass

        return results

    if task == "regression":
        return {
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

    raise ValueError(f"Unsupported task type: {task}")


def evaluate_classification_extended(y_true, y_pred, y_proba) -> dict:
    """
    Extended classification metrics used in healthcare:
      - accuracy, f1, roc_auc
      - sensitivity (recall for positive class)
      - specificity (true negative rate)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC may fail if only one class exists in y_true (rare but possible)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }