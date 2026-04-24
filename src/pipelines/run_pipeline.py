import logging
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.utils.logging_config import configure_logging

from src.pipelines.step01_dataset import run_step01_load_dataset
from src.pipelines.step01_dataset_eda import run_step01_dataset_eda
from src.pipelines.step02_preprocessing import run_step02_preprocessing
from src.pipelines.step03_baseline_models import run_step03_baseline_models
from src.pipelines.step04_proposed_model import run_step04_proposed_models
from src.pipelines.step05_explainability import run_step05_explainability
from src.pipelines.step06_visualization import run_step06_visualization
from src.pipelines.step07_error_analysis import run_step07_error_analysis
from src.pipelines.step08_statistical_test import run_step08_statistical_test
from src.pipelines.step09_paper_tables import run_step09_generate_tables
from src.pipelines.step10_reproducibility import run_step10_reproducibility
from src.pipelines.step11_auto_report import run_step11_auto_report
from src.utils.project_paths import get_outputs_dir


def run_all_steps():

    # ============================================================
    # STEP 1
    # ============================================================
    df, config = run_step01_load_dataset()

    run_step01_dataset_eda(df, get_outputs_dir())

    # ============================================================
    # STEP 2
    # ============================================================
    data = run_step02_preprocessing(df, config)

    # ============================================================
    # STEP 3 — returns (results, trained_models)
    # ============================================================
    baseline_results, all_baseline_models = run_step03_baseline_models(data)

    # ============================================================
    # STEP 4
    # ============================================================
    proposed_results = run_step04_proposed_models(data)

    best_model      = proposed_results["best_model"]
    best_model_name = proposed_results["best_model_name"]

    # ── Select XAI model: ACTIVE_MODEL or best_model ─────────────────────────
    # Import inside function to avoid circular import chain
    from src.datasets.dataset_registry import ACTIVE_MODEL  # noqa
    all_trained_models = {
        **all_baseline_models,
        **proposed_results["models"],
    }

    if ACTIVE_MODEL and ACTIVE_MODEL in all_trained_models:
        xai_model      = all_trained_models[ACTIVE_MODEL]
        xai_model_name = ACTIVE_MODEL
        logging.info(f"Using ACTIVE_MODEL for XAI: {xai_model_name}")
    else:
        xai_model      = best_model
        xai_model_name = best_model_name
        logging.info(f"ACTIVE_MODEL not found, using best model: {xai_model_name}")

    logging.info(f"XAI model: {xai_model_name}")

    # ============================================================
    # STEP 5 — SHAP + FI, returns xai_results with timing
    # ============================================================
    xai_results = run_step05_explainability(
        data, xai_model, model_name=xai_model_name
    )

    # ============================================================
    # STEP 6
    # ============================================================
    run_step06_visualization(
        baseline_results,
        proposed_results["metrics"]
    )

    # ============================================================
    # STEP 7 — Error analysis + Fairness (Equalized Odds)
    # ============================================================
    run_step07_error_analysis(data, xai_model)

    # ============================================================
    # STEP 8
    # ============================================================
    run_step08_statistical_test(
        baseline_results,
        proposed_results["metrics"]
    )

    # ============================================================
    # STEP 9
    # ============================================================
    run_step09_generate_tables(
        baseline_results,
        proposed_results["metrics"]
    )

    # ============================================================
    # STEP 10 — Reproducibility with XAI timing
    # ============================================================
    run_step10_reproducibility(
        data,
        baseline_results,
        proposed_results,
        xai_results=xai_results
    )

    # ============================================================
    # STEP 11
    # ============================================================
    run_step11_auto_report()


def main():

    configure_logging()

    logging.info("============================================================")
    logging.info("PIPELINE RUN STARTED")
    logging.info("============================================================")

    run_all_steps()


if __name__ == "__main__":
    main()