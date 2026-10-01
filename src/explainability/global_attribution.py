"""
Fit one model and compute its global attributions (SHAP and Permutation
Importance) for a single train/test split.

Shared by the multi-seed analyses (label axis on NHANES, seed robustness on
BRFSS). Model hyperparameters mirror configs/default.yaml; only
``random_state`` changes with the seed.

Differences from the MAPR pipeline (step05), made on purpose:
    - Permutation Importance keeps the SIGNED mean and every repeat
      (``importances_``) instead of ``abs(importances_mean)``; negative values
      are clipped to 0 only when a non-negative vector is needed downstream.
    - SHAP for tree models uses ``shap_n`` test rows (default: all) instead
      of a fixed 200-row sample.
    - Optional survey weights: weighted mean |SHAP| and weighted PI scoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODELS = ("xgboost", "random_forest", "logistic_regression")


CLASS_WEIGHTING = ("pipeline", "none", "balanced")


def build_model(name: str, seed: int, n_jobs: int = 1, class_weighting: str = "pipeline",
                pos_ratio: float | None = None) -> Pipeline:
    """
    class_weighting
        pipeline  as in the MAPR pipeline: LR and RF balanced, XGBoost unweighted
        none      no reweighting for any model
        balanced  reweighting for every model (XGBoost: scale_pos_weight =
                  n_negative / n_positive, needs ``pos_ratio``)
    """
    if class_weighting not in CLASS_WEIGHTING:
        raise ValueError(f"class_weighting must be one of {CLASS_WEIGHTING}")
    cw = None if class_weighting == "none" else "balanced"
    if name == "logistic_regression":
        est = LogisticRegression(max_iter=1000, class_weight=cw, random_state=seed)
    elif name == "random_forest":
        est = RandomForestClassifier(n_estimators=200, class_weight=cw,
                                     random_state=seed, n_jobs=n_jobs)
    elif name == "xgboost":
        spw = 1.0
        if class_weighting == "balanced":
            if not pos_ratio:
                raise ValueError("pos_ratio is required for balanced XGBoost")
            spw = (1.0 - pos_ratio) / pos_ratio
        est = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, eval_metric="logloss", random_state=seed,
                            n_jobs=n_jobs, scale_pos_weight=spw)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("scaler", StandardScaler()), ("model", est)])


@dataclass
class AttributionResult:
    auc: float
    auc_weighted: Optional[float]
    shap_mean_abs: pd.Series
    shap_mean_abs_weighted: Optional[pd.Series]
    pi_mean: pd.Series
    pi_std: pd.Series
    pi_mean_weighted: Optional[pd.Series]
    timings: Dict[str, float] = field(default_factory=dict)


def _positive_class(values) -> np.ndarray:
    values = np.asarray(values[1] if isinstance(values, list) else values)
    if values.ndim == 3:                      # (n, features, classes)
        values = values[:, :, 1]
    return values


def _tree_shap_chunk(est, X: np.ndarray) -> np.ndarray:
    return _positive_class(shap.TreeExplainer(est).shap_values(X, check_additivity=False))


def _shap_values(pipe: Pipeline, name: str, X_bg: np.ndarray, X_eval: np.ndarray,
                 n_jobs: int = 1) -> np.ndarray:
    est = pipe.named_steps["model"]
    if name == "logistic_regression":
        return _positive_class(shap.LinearExplainer(est, X_bg).shap_values(X_eval))
    if n_jobs > 1 and len(X_eval) >= 2 * n_jobs:
        # TreeSHAP is single-threaded; split rows across processes (exact, same result).
        from joblib import Parallel, delayed
        chunks = np.array_split(X_eval, n_jobs)
        parts = Parallel(n_jobs=n_jobs)(delayed(_tree_shap_chunk)(est, c) for c in chunks)
        return np.vstack(parts)
    return _tree_shap_chunk(est, X_eval)


def fit_and_attribute(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
    w_test: Optional[pd.Series] = None,
    shap_n: Optional[int] = None,
    pi_repeats: int = 10,
    n_jobs: int = 1,
    pi_n: Optional[int] = None,
    class_weighting: str = "pipeline",
    shap_jobs: int = 1,
    w_train: Optional[pd.Series] = None,
) -> AttributionResult:
    """``pi_n``: stratified subsample of the test set for Permutation
    Importance (default: whole test set); AUC and SHAP are unaffected."""
    feats = list(X_train.columns)
    timings: Dict[str, float] = {}

    t0 = time.time()
    pipe = build_model(name, seed, n_jobs, class_weighting, pos_ratio=float(np.mean(y_train)))
    if w_train is None:
        pipe.fit(X_train, y_train)
    else:  # survey-weighted training (combined with class weighting where set)
        pipe.fit(X_train, y_train, model__sample_weight=np.asarray(w_train, dtype=float))
    timings["fit_s"] = time.time() - t0

    prob = pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, prob))
    auc_w = float(roc_auc_score(y_test, prob, sample_weight=w_test)) if w_test is not None else None

    # --- SHAP on the scaled design matrix ---
    t0 = time.time()
    scaler = pipe.named_steps["scaler"]
    Xs_test = scaler.transform(X_test)
    idx = np.arange(len(X_test))
    if shap_n is not None and shap_n < len(idx):
        idx = np.random.default_rng(seed).choice(idx, shap_n, replace=False)
    sv = np.abs(_shap_values(pipe, name, Xs_test, Xs_test[idx], n_jobs=shap_jobs))
    shap_mean = pd.Series(sv.mean(axis=0), index=feats)
    shap_w = None
    if w_test is not None:
        w = np.asarray(w_test, dtype=float)[idx]
        shap_w = pd.Series((sv * w[:, None]).sum(axis=0) / w.sum(), index=feats)
    timings["shap_s"] = time.time() - t0

    # --- Permutation Importance (ROC-AUC drop), signed, all repeats kept ---
    t0 = time.time()
    X_pi, y_pi, w_pi = X_test, y_test, w_test
    if pi_n is not None and pi_n < len(X_test):
        from sklearn.model_selection import train_test_split
        keep, _ = train_test_split(np.arange(len(X_test)), train_size=pi_n,
                                   random_state=seed, stratify=np.asarray(y_test))
        X_pi, y_pi = X_test.iloc[keep], y_test.iloc[keep]
        w_pi = w_test.iloc[keep] if w_test is not None else None
    pi = permutation_importance(pipe, X_pi, y_pi, n_repeats=pi_repeats,
                                random_state=seed, scoring="roc_auc", n_jobs=1)
    pi_mean = pd.Series(pi.importances_mean, index=feats)
    pi_std = pd.Series(pi.importances_std, index=feats)
    pi_w = None
    if w_test is not None:
        piw = permutation_importance(pipe, X_pi, y_pi, n_repeats=pi_repeats,
                                     random_state=seed, scoring="roc_auc", n_jobs=1,
                                     sample_weight=np.asarray(w_pi, dtype=float))
        pi_w = pd.Series(piw.importances_mean, index=feats)
    timings["pi_s"] = time.time() - t0

    return AttributionResult(auc, auc_w, shap_mean, shap_w, pi_mean, pi_std, pi_w, timings)
