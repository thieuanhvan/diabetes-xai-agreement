"""
STEP06 - RESULT VISUALIZATION

Combine baseline and proposed model results
and generate comparison plots.
"""

import logging

from src.utils.project_paths import get_outputs_dir
from evaluation.classification import concatenate_results
from reporting.plots import generate_model_comparison_plots


def run_step06_visualization(baseline_results, proposed_results):

    logging.info("============================================================")
    logging.info("STEP 06 - RESULT VISUALIZATION")
    logging.info("============================================================")

    # Combine baseline + proposed metrics
    results = concatenate_results(
        baseline_results,
        proposed_results
    )

    plots_dir = get_outputs_dir("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    generate_model_comparison_plots(
        results,
        plots_dir / "model_performance_comparison.png"  # pass FILE path
    )

    logging.info("STEP 06 COMPLETED")

    return results