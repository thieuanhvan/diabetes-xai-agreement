import logging
import json
import platform
try:
    import pkg_resources
except ImportError:
    pkg_resources = None
from pathlib import Path

from utils.project_paths import get_outputs_dir


def run_step10_reproducibility(data, baseline_results, proposed_results,
                                xai_results=None):
    """
    STEP 10 — Reproducibility.
    xai_results: dict from step05 containing shap_time_s, fi_time_s.
    """
    logging.info("============================================================")
    logging.info("STEP 10 - REPRODUCIBILITY")
    logging.info("============================================================")

    output_dir = get_outputs_dir("reproducibility")

    rows = data["X_train"].shape[0] + data["X_test"].shape[0]
    cols = data["X_train"].shape[1]

    summary = {
        "dataset": {"rows": int(rows), "features": int(cols)},
        "baseline_models": baseline_results,
        "proposed_model":  proposed_results,
    }

    # ── XAI timing ────────────────────────────────────────────────────────────
    if xai_results:
        summary["xai_timing"] = {
            "shap_time_s": xai_results.get("shap_time_s"),
            "fi_time_s":   xai_results.get("fi_time_s"),
            "note": ("shap_time_s = wall-clock seconds for SHAP computation "
                     "(TreeSHAP or LinearSHAP on sample_size=200). "
                     "fi_time_s = time to read tree feature_importances_ "
                     "(near-instant, intrinsic to training).")
        }

    summary_path = Path(output_dir) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4, default=str)
    logging.info(f"Saved experiment summary to {summary_path}")

    # ── Environment ───────────────────────────────────────────────────────────

    packages = {}
    if pkg_resources is not None:
        packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    env = {
        "python_version": platform.python_version(),
        "platform":       platform.platform(),
        "packages":      packages
    }

    env_path = Path(output_dir) / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=4)
    logging.info(f"Saved environment info to {env_path}")

    logging.info("STEP 10 COMPLETED")
    return {"summary": summary_path, "environment": env_path}