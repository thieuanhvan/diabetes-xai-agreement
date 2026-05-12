"""
STEP 09: Reproducibility logging.

Writes two artifacts into outputs/<dataset>/reproducibility/:
    experiment_summary.json  -- dataset shape, model metrics, XAI timings.
    environment.json         -- Python version, platform, installed package
                                versions (collected via importlib.metadata).

Updated 12/05/2026: the XAI-timing `note` text now accurately describes
the current FI implementation (Permutation Importance via
sklearn.inspection.permutation_importance, in use since 19/04/2026),
replacing the stale reference to tree feature_importances_. Package
collection switched from pkg_resources (deprecated in setuptools 67+)
to importlib.metadata, which fixes silent failures that produced an
empty packages dict on some environments.
"""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path

from utils.project_paths import get_outputs_dir


def _collect_installed_packages() -> dict:
    """Return a sorted {name: version} map for installed distributions.

    Prefers importlib.metadata (stdlib, Python 3.8+); falls back to
    pkg_resources for older environments. Returns {} only if both
    strategies fail. Names are lowercased for stable cross-run comparison.
    """
    try:
        from importlib.metadata import distributions

        pkgs: dict[str, str] = {}
        for dist in distributions():
            try:
                name = dist.metadata["Name"]
            except (KeyError, AttributeError):
                name = None
            if not name:
                continue
            pkgs[name.lower()] = dist.version

        if pkgs:
            return dict(sorted(pkgs.items()))

        logging.warning(
            "importlib.metadata returned an empty distribution list; "
            "trying pkg_resources fallback."
        )
    except Exception as e:
        logging.warning(
            f"importlib.metadata.distributions() failed: {e!r}; "
            "trying pkg_resources fallback."
        )

    try:
        import pkg_resources

        pkgs = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        return dict(sorted(pkgs.items()))
    except Exception as e:
        logging.warning(
            f"pkg_resources fallback also failed: {e!r}; "
            "returning empty package map."
        )
        return {}


def run_step09_reproducibility(data, baseline_results, proposed_results,
                                xai_results=None):
    """Write experiment_summary.json and environment.json.

    Parameters
    ----------
    data : dict with "X_train" and "X_test" (used for row + feature count).
    baseline_results : list[dict] of per-model metrics (LR, RF, DT, kNN).
    proposed_results : dict of per-model metrics + serialized estimators
        for the proposed block (XGBoost, ANN).
    xai_results : optional dict with "shap_time_s" and "fi_time_s"
        (wall-clock seconds returned by step05_explainability).
    """
    logging.info("============================================================")
    logging.info("STEP 09 - REPRODUCIBILITY")
    logging.info("============================================================")

    output_dir = get_outputs_dir("reproducibility")

    rows = data["X_train"].shape[0] + data["X_test"].shape[0]
    cols = data["X_train"].shape[1]

    summary = {
        "dataset": {"rows": int(rows), "features": int(cols)},
        "baseline_models": baseline_results,
        "proposed_model":  proposed_results,
    }

    if xai_results:
        summary["xai_timing"] = {
            "shap_time_s": xai_results.get("shap_time_s"),
            "fi_time_s":   xai_results.get("fi_time_s"),
            "note": (
                "shap_time_s = wall-clock seconds for SHAP computation: "
                "TreeSHAP on sample_size=200 (seeded random_state=42) for "
                "XGBoost and RandomForest; LinearSHAP (closed-form) on the "
                "full test set for LogisticRegression. "
                "fi_time_s = wall-clock seconds for Permutation Importance "
                "via sklearn.inspection.permutation_importance "
                "(n_repeats=10, scoring='roc_auc', n_jobs=1, "
                "random_state=42) on the full held-out 20% test partition. "
                "Both attribution methods produce one global importance "
                "vector per (model, cohort) cell."
            ),
        }

    summary_path = Path(output_dir) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4, default=str)
    logging.info(f"Saved experiment summary to {summary_path}")

    packages = _collect_installed_packages()

    env = {
        "python_version": platform.python_version(),
        "platform":       platform.platform(),
        "packages":       packages,
    }

    env_path = Path(output_dir) / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=4)
    logging.info(
        f"Saved environment info to {env_path} "
        f"({len(packages)} packages logged)"
    )

    logging.info("STEP 09 COMPLETED")
    return {"summary": summary_path, "environment": env_path}