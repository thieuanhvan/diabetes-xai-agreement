"""
Model Stability Analysis
- Run multiple random seeds for each dataset and model
- Report mean ± std for accuracy, f1, roc_auc, sensitivity, specificity
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd


from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.project_paths import get_outputs_dir, get_log_file
from datasets.dataset_registry import get_all_dataset_names
from datasets.loader import load_dataset_by_name
from experiments.evaluation.evaluation import evaluate_classification_extended



# Models that usually benefit from scaling
MODELS_REQUIRING_SCALING = {"logistic_regression", "svm_rbf", "knn", "ann_mlp"}

DEFAULT_SEEDS = [1, 7, 21, 42, 99]
TEST_SIZE = 0.2


def setup_logging():
    log_file = get_log_file("stability")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 60)
    logging.info("MODEL STABILITY ANALYSIS STARTED")
    logging.info("=" * 60)


def _apply_task_transform(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    task = cfg.get("task", "classification")
    target = cfg["target"]

    if task == "regression_to_binary":
        threshold = df[target].median()
        df[target] = (df[target] > threshold).astype(int)

    return df


def _build_xy(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.Series]:
    target = cfg["target"]
    include_columns = cfg.get("include_columns")

    if include_columns is not None:
        cols = [c for c in include_columns if c != target]
        X = df[cols]
    else:
        X = df.drop(columns=[target])

    y = df[target]
    return X, y


def _to_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X, drop_first=True)


def _predict_proba_compat(model, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_test))
        # Normalize to 0..1
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-12:
            return np.zeros_like(scores, dtype=float)
        return (scores - s_min) / (s_max - s_min)

    # Fallback
    preds = model.predict(X_test)
    return np.asarray(preds, dtype=float)


def run_stability_for_dataset(dataset_name: str, seeds: List[int]) -> pd.DataFrame:
    df, cfg = load_dataset_by_name(dataset_name)
    df = _apply_task_transform(df, cfg)

    slug = cfg.get("slug", dataset_name)
    target = cfg["target"]

    X_raw, y = _build_xy(df, cfg)
    X = _to_numeric_features(X_raw)

    task = cfg.get("task", "classification")
    is_classification = task in {"classification", "regression_to_binary"}

    models = get_all_models()
    rows = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.get("test_size", TEST_SIZE),
            random_state=seed,
            stratify=y if is_classification else None,
        )

        for name, estimator in models.items():
            # Recreate estimator for safety (some models keep state)
            model = estimator

            if name in MODELS_REQUIRING_SCALING:
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", estimator),
                ])

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = _predict_proba_compat(model, X_test)

            metrics = evaluate_classification_extended(y_test, y_pred, y_proba)

            rows.append({
                "dataset": slug,
                "model": name,
                "seed": seed,
                **metrics,
            })

    result = pd.DataFrame(rows)

    out_dir = get_outputs_dir("analysis/stability")
    out_path = out_dir / f"stability_raw_{slug}.csv"
    result.to_csv(out_path, index=False)

    logging.info(f"Saved raw stability: {out_path}")
    return result


def summarize_stability(raw_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "f1", "roc_auc", "sensitivity", "specificity"]

    summary = (
        raw_df
        .groupby(["dataset", "model"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten columns
    summary.columns = [
        "_".join([c for c in col if c]).strip("_")
        for col in summary.columns.to_flat_index()
    ]

    return summary


def main():
    setup_logging()

    names = get_all_dataset_names()
    logging.info(f"Datasets detected: {names}")

    all_raw = []
    for ds in names:
        logging.info(f"Running stability: {ds}")
        raw = run_stability_for_dataset(ds, DEFAULT_SEEDS)
        all_raw.append(raw)

    all_raw_df = pd.concat(all_raw, ignore_index=True)
    summary_df = summarize_stability(all_raw_df)

    out_dir = get_outputs_dir("analysis/stability")
    out_path = out_dir / "stability_summary_all_datasets.csv"
    summary_df.to_csv(out_path, index=False)

    logging.info(f"Saved stability summary: {out_path}")
    logging.info("MODEL STABILITY ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()