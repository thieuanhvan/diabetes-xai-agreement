import logging
from pathlib import Path

from src.analysis.error_analysis import run_error_analysis
from src.analysis.fairness_analysis import run_fairness_analysis
from src.utils.project_paths import get_outputs_dir
from src.datasets.dataset_registry import ACTIVE_MODEL


def run_step07_error_analysis(data, model):
    """
    STEP 07
    Error analysis + Equalized Odds fairness analysis.

    Outputs:
        analysis/false_positive_samples.csv
        analysis/false_negative_samples.csv
        analysis/{model}_fairness_equalized_odds_summary.csv
        analysis/{model}_fairness_age_detail.csv
        analysis/{model}_fairness_sex_detail.csv
        analysis/{model}_fairness_income_detail.csv
    """

    logging.info("============================================================")
    logging.info("STEP 07 - ERROR ANALYSIS + FAIRNESS (EQUALIZED ODDS)")
    logging.info("============================================================")

    output_dir = get_outputs_dir("analysis")

    # ── Standard error analysis ───────────────────────────────────────────────
    artifacts = run_error_analysis(
        model=model,
        X_test=data["X_test"],
        y_test=data["y_test"],
        output_dir=output_dir,
    )

    # ── Equalized Odds fairness analysis ─────────────────────────────────────
    try:
        fairness_results = run_fairness_analysis(
            model=model,
            X_test=data["X_test"],
            y_test=data["y_test"],
            output_dir=output_dir,
            model_name=ACTIVE_MODEL,
        )
        artifacts["fairness"] = fairness_results

        # Log summary table
        logging.info("Equalized Odds Summary:")
        for _, row in fairness_results["summary_df"].iterrows():
            logging.info(
                f"  {row['attribute']:8s}: "
                f"dTPR={row['delta_tpr']:.3f}  "
                f"dFPR={row['delta_fpr']:.3f}  "
                f"EO={row['eo_violation']:.3f}  "
                f"[{row['severity']}]"
            )

    except Exception as e:
        logging.warning(f"Fairness analysis failed: {e}")

    logging.info(f"Saved error analysis outputs to {output_dir}")
    logging.info("STEP 07 COMPLETED")

    return artifacts