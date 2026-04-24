"""
SHAP Explainability Module
Generate SHAP explanations and feature importance.
"""

import logging
import time
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline


def run_tree_shap_analysis(model, X_test, output_dir, sample_size=200, model_name="model"):
    """
    Run SHAP analysis for a tree-based model (or sklearn Pipeline).
    Returns dict with importance_df and shap_time_s.
    """
    logging.info(f"Running SHAP analysis for model: {model_name}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, Pipeline):
        steps = list(model.named_steps.values())
        estimator  = steps[-1]
        preprocess = steps[0]
        X_processed = pd.DataFrame(
            preprocess.transform(X_test),
            columns=[str(c) for c in X_test.columns]
        )
    else:
        estimator   = model
        X_processed = X_test.copy()

    feature_names = [str(c) for c in X_processed.columns]

    if len(X_processed) > sample_size:
        X_sample = X_processed.sample(sample_size, random_state=42)
    else:
        X_sample = X_processed

    # ── SHAP with timing ─────────────────────────────────────────────────────
    t0 = time.time()
    try:

        logging.info("Initializing TreeExplainer...")
        explainer  = shap.TreeExplainer(estimator)
        #logging.info("TreeExplainer: OK")
        logging.info("TreeExplainer initialized successfully")

        logging.info(f"Computing SHAP on sample size: {len(X_sample)}")
        logging.info("Starting SHAP value computation...")
        shap_values = explainer.shap_values(X_sample)
        logging.info("SHAP values computed")

    except Exception:
        logging.warning("TreeExplainer failed, falling back to KernelExplainer")
        background  = shap.sample(X_sample, 50)
        explainer   = shap.KernelExplainer(estimator.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)

    shap_time_s = round(time.time() - t0, 2)
    logging.info(f"SHAP computation time: {shap_time_s}s")

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    # ── Plots ─────────────────────────────────────────────────────────────────
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(output_dir / f"{model_name}_shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()
    logging.info(f"Saved: {output_dir / f'{model_name}_shap_summary.png'}")

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(output_dir / f"{model_name}_shap_bar.png", bbox_inches="tight", dpi=300)
    plt.close()
    logging.info(f"Saved: {output_dir / f'{model_name}_shap_bar.png'}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    csv_path = output_dir / f"{model_name}_shap_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    logging.info(f"Saved: {csv_path}")
    logging.info("SHAP analysis completed")

    return {"importance_df": importance_df, "shap_time_s": shap_time_s}


def run_linear_shap_analysis(model, X_test, output_dir, model_name="logistic_regression"):
    """LinearSHAP for Logistic Regression. Returns dict with importance_df and shap_time_s."""
    logging.info(f"Running LinearSHAP for model: {model_name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, Pipeline):
        estimator   = model.named_steps["model"]
        preprocess  = model.named_steps["preprocessor"]
        X_processed = pd.DataFrame(
            preprocess.transform(X_test),
            columns=[str(c) for c in X_test.columns]
        )
    else:
        estimator   = model
        X_processed = X_test.copy()

    t0 = time.time()
    explainer   = shap.LinearExplainer(estimator, X_processed)
    shap_values = explainer.shap_values(X_processed)
    shap_time_s = round(time.time() - t0, 2)
    logging.info(f"LinearSHAP computation time: {shap_time_s}s")

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.array(shap_values)

    plt.figure()
    shap.summary_plot(shap_values, X_processed, show=False)
    plt.savefig(output_dir / f"{model_name}_shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_processed, plot_type="bar", show=False)
    plt.savefig(output_dir / f"{model_name}_shap_bar.png", bbox_inches="tight", dpi=300)
    plt.close()

    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    [str(c) for c in X_processed.columns],
        "importance": importance,
    }).sort_values(by="importance", ascending=False)

    csv_path = output_dir / f"{model_name}_shap_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    logging.info(f"Saved: {csv_path}")
    logging.info("LinearSHAP analysis completed")

    return {"importance_df": importance_df, "shap_time_s": shap_time_s}


def run_shap_analysis(model, X_test, output_dir, model_name="model"):
    """Auto-dispatch SHAP: TreeSHAP for tree models, LinearSHAP for LR."""
    if isinstance(model, Pipeline):
        core = model.named_steps.get("model", list(model.named_steps.values())[-1])
    else:
        core = model

    model_type = type(core).__name__.lower()

    skip_types   = ("kneighbors", "mlp", "svc", "svr")
    linear_types = ("logistic", "linearregression", "ridge", "sgd")

    if any(t in model_type for t in skip_types):
        raise ValueError(f"Model '{model_type}' not supported for SHAP (too slow).")
    elif any(t in model_type for t in linear_types):
        return run_linear_shap_analysis(model, X_test, output_dir, model_name)
    else:
        return run_tree_shap_analysis(model, X_test, output_dir, model_name=model_name)