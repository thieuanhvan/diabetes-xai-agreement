import logging
from pathlib import Path
import pandas as pd
from scipy.stats import ttest_ind

from utils.project_paths import get_outputs_dir


def run_step08_statistical_test(baseline_results, proposed_metrics):

    logging.info("============================================================")
    logging.info("STEP 08 - STATISTICAL TEST")
    logging.info("============================================================")

    output_dir = get_outputs_dir("analysis")

    # proposed_metrics là list -> lấy phần tử đầu
    if isinstance(proposed_metrics, list):
        proposed_metrics = proposed_metrics[0]

    proposed_acc = proposed_metrics["accuracy"]

    records = []

    for metrics in baseline_results:

        model_name = metrics["model"]
        baseline_acc = metrics["accuracy"]

        stat, pvalue = ttest_ind([baseline_acc], [proposed_acc])

        records.append({
            "baseline_model": model_name,
            "baseline_accuracy": baseline_acc,
            "proposed_accuracy": proposed_acc,
            "p_value": pvalue
        })

    df = pd.DataFrame(records)

    output_path = Path(output_dir) / "statistical_test_results.csv"
    df.to_csv(output_path, index=False)

    logging.info(f"Saved statistical test results to {output_path}")
    logging.info("STEP 08 COMPLETED")

    return df