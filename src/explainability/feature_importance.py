"""
Feature Importance Module
==========================
Permutation Importance as the UNIFIED FI method for all models
(XGBoost, Random Forest, Logistic Regression, ...).

Rationale:
    - Permutation Importance is model-agnostic (Breiman, 2001), giving
      identical semantics across tree-based and linear models. This makes
      SHAP-vs-FI agreement comparable across architectures (apples-to-apples).
    - Replaces model-specific Gain (XGBoost) and MDI (Random Forest), which
      have distinct mathematical meanings and cannot be directly compared
      across models.
    - For Logistic Regression (which lacks `feature_importances_`), this
      is the only principled native-FI alternative.

Metric: mean decrease in ROC-AUC on the test set when a feature's values
are randomly permuted. Larger = more important.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def run_feature_importance_analysis(model, X, y, output_dir, model_name="model",
                                    n_repeats: int = 10,
                                    random_state: int = 42,
                                    scoring: str = "roc_auc"):
    """
    Compute Permutation Importance for any fitted classifier (with or
    without a sklearn Pipeline wrapper).

    Parameters
    ----------
    model : fitted estimator or sklearn Pipeline
        The full pipeline (preprocessor + classifier) is expected.
        Permutation is performed on raw X (pipeline handles preprocessing).
    X : pd.DataFrame
        Test features (raw, before preprocessing).
    y : pd.Series or np.ndarray
        True labels for the test set.
    output_dir : Path
        Directory to write the CSV.
    model_name : str
        Used as filename prefix.
    n_repeats : int
        Number of permutation repeats. sklearn default 5; we use 10 for
        more stable estimates.
    random_state : int
        Seed for reproducibility (matches repo convention).
    scoring : str
        Metric to measure degradation. "roc_auc" is appropriate for
        binary classification with class imbalance (BRFSS case).

    Returns
    -------
    (fi_df, fi_time_s) : tuple
        fi_df — DataFrame with columns ["feature", "importance"], sorted desc.
                'importance' is the mean drop in ROC-AUC across n_repeats.
        fi_time_s — wall-clock time in seconds (rounded to 4 decimals).
    """
    logging.info(f"Running Permutation Importance for: {model_name} "
                 f"(scoring={scoring}, n_repeats={n_repeats})")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Van change n_jobs value because not see fi/random_forest_feature_importance.csv
    # date 2026-04-25
    #result = permutation_importance(
    #    model, X, y,
    #    n_repeats=n_repeats,
    #    random_state=random_state,
    #    scoring=scoring,
    #    n_jobs=-1,
    #)
    # Use n_jobs=1 to avoid multiprocessing pickling issues on Windows/PyCharm,
    # especially for RandomForest pipelines.
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1,
    )

    fi_time_s = round(time.time() - t0, 4)

    # sklearn returns importances_mean aligned with X.columns (raw features).
    # We use |importance| to guard against tiny negative values from noise.
    importances = np.abs(result.importances_mean)

    fi_df = (pd.DataFrame({
                "feature":    X.columns.astype(str),
                "importance": importances,
             })
             .sort_values(by="importance", ascending=False)
             .reset_index(drop=True))

    output_path = output_dir / f"{model_name}_feature_importance.csv"
    fi_df.to_csv(output_path, index=False)
    logging.info(f"Permutation FI saved to: {output_path} ({fi_time_s}s)")

    return fi_df, fi_time_s