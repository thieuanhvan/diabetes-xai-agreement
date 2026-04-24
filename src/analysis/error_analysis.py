from pathlib import Path
import logging

import numpy as np
import pandas as pd


def run_error_analysis(model, X_test, y_test, output_dir):
    """
    Perform error analysis for a trained classification model.

    Parameters
    ----------
    model : fitted sklearn model or Pipeline
        Trained model used for prediction.
    X_test : pandas.DataFrame or array-like
        Test features.
    y_test : pandas.Series or array-like
        True labels of the test set.
    output_dir : str or Path
        Directory used to save error analysis artifacts.

    Returns
    -------
    dict
        Paths to saved artifacts and summary counts.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure X_test is always a DataFrame
    if isinstance(X_test, pd.DataFrame):
        df = X_test.copy().reset_index(drop=True)
    else:
        X_np = np.asarray(X_test)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        df = pd.DataFrame(X_np).reset_index(drop=True)

    # Ensure y_test is always a Series
    if isinstance(y_test, pd.Series):
        y_true = y_test.reset_index(drop=True)
    else:
        y_true = pd.Series(np.asarray(y_test).ravel(), name="y_true")

    # Predict
    y_pred = model.predict(X_test)
    y_pred = pd.Series(np.asarray(y_pred).ravel(), name="y_pred")

    # Attach labels
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    # Error subsets
    false_positive = df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1)].copy()
    false_negative = df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0)].copy()

    # Save raw samples
    fp_samples_path = output_dir / "false_positive_samples.csv"
    fn_samples_path = output_dir / "false_negative_samples.csv"

    false_positive.to_csv(fp_samples_path, index=False)
    false_negative.to_csv(fn_samples_path, index=False)

    # Save feature means if available
    feature_cols = [c for c in df.columns if c not in ["y_true", "y_pred"]]

    fp_mean_path = output_dir / "false_positive_feature_mean.csv"
    fn_mean_path = output_dir / "false_negative_feature_mean.csv"

    if len(false_positive) > 0:
        false_positive[feature_cols].mean(numeric_only=True).to_csv(fp_mean_path, header=["mean"])
    else:
        pd.DataFrame(columns=["mean"]).to_csv(fp_mean_path)

    if len(false_negative) > 0:
        false_negative[feature_cols].mean(numeric_only=True).to_csv(fn_mean_path, header=["mean"])
    else:
        pd.DataFrame(columns=["mean"]).to_csv(fn_mean_path)

    summary = {
        "num_false_positive": int(len(false_positive)),
        "num_false_negative": int(len(false_negative)),
        "false_positive_samples_path": str(fp_samples_path),
        "false_negative_samples_path": str(fn_samples_path),
        "false_positive_feature_mean_path": str(fp_mean_path),
        "false_negative_feature_mean_path": str(fn_mean_path),
    }

    logging.info(f"Saved error analysis outputs to {output_dir}")

    return summary