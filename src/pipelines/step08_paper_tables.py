import logging
import pandas as pd
from pathlib import Path

from utils.project_paths import get_outputs_dir


def run_step08_generate_tables(baseline_results, proposed_metrics):

    logging.info("============================================================")
    logging.info("STEP 08 - GENERATE RESULT TABLES")
    logging.info("============================================================")

    output_dir = get_outputs_dir("tables")

    records = []

    # ==============================
    # baseline models
    # ==============================

    for metrics in baseline_results:

        records.append({
            "model":            metrics["model"],
            "accuracy":         metrics["accuracy"],
            "f1":               metrics["f1"],
            "roc_auc":          metrics["roc_auc"],
            "sensitivity":      metrics["sensitivity"],
            "specificity":      metrics["specificity"],
            "training_time_s":  metrics.get("training_time_s", None),
        })

    # ==============================
    # proposed models
    # ==============================

    if isinstance(proposed_metrics, dict):
        proposed_metrics = [proposed_metrics]

    for metrics in proposed_metrics:

        records.append({
            "model":            metrics["model"],
            "accuracy":         metrics["accuracy"],
            "f1":               metrics["f1"],
            "roc_auc":          metrics["roc_auc"],
            "sensitivity":      metrics["sensitivity"],
            "specificity":      metrics["specificity"],
            "training_time_s":  metrics.get("training_time_s", None),
        })

    df = pd.DataFrame(records)

    output_path = Path(output_dir) / "model_comparison_table.csv"

    df.to_csv(output_path, index=False)

    logging.info(f"Saved comparison table to {output_path}")
    logging.info("STEP 08 COMPLETED")

    return df