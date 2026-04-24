import logging
from pathlib import Path

from src.utils.project_paths import get_outputs_dir
from src.explainability.shap_analysis import run_shap_analysis
from src.explainability.feature_importance import run_feature_importance_analysis


def run_step05_explainability(data, model, model_name="model"):
    """
    STEP 05 — SHAP + Permutation Feature Importance.

    Notes (updated 19/04/2026):
      - FI is now Permutation Importance (model-agnostic), replacing
        model-specific Gain (XGB) / MDI (RF). This unifies FI semantics
        across all models and enables apples-to-apples SHAP-vs-FI agreement
        analysis in Step 12.
      - Permutation runs on raw X_test (pipeline handles preprocessing),
        so feature names align naturally with SHAP output (no "num__" prefix).

    Returns dict with shap_time_s and fi_time_s.
    """
    logging.info("============================================================")
    logging.info("STEP 05 - EXPLAINABILITY (SHAP + PERMUTATION FI)")
    logging.info("============================================================")

    X_test = data["X_test"]
    y_test = data["y_test"]

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_output_dir = get_outputs_dir("shap")
    shap_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Running SHAP analysis for: {model_name}")
    try:
        shap_result = run_shap_analysis(
            model, X_test,
            output_dir=shap_output_dir,
            model_name=model_name
        )
        shap_df     = shap_result["importance_df"]
        shap_time_s = shap_result["shap_time_s"]
        logging.info(f"SHAP analysis completed in {shap_time_s}s")
    except ValueError as e:
        logging.warning(f"SHAP skipped: {e}")
        shap_df, shap_time_s = None, None

    # ── Permutation Feature Importance (universal, model-agnostic) ───────────
    fi_output_dir = get_outputs_dir("fi")
    fi_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fi_df, fi_time_s = run_feature_importance_analysis(
            model, X_test, y_test,
            output_dir=fi_output_dir,
            model_name=model_name,
        )
        logging.info(f"Permutation FI completed in {fi_time_s}s")
    except Exception as e:
        logging.warning(f"Permutation FI failed: {e}")
        fi_df, fi_time_s = None, None

    logging.info("STEP 05 COMPLETED")

    return {
        "shap":               shap_df,
        "shap_time_s":        shap_time_s,
        "feature_importance": fi_df,
        "fi_time_s":          fi_time_s,
    }