import logging
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import spearmanr

from src.utils.project_paths import get_outputs_dir
from src.datasets.dataset_registry import ACTIVE_DATASET, ACTIVE_MODEL


# ============================================================
# CONFIG
# ============================================================
TOP_K_LIST = [5, 10, 15]


# ============================================================
# UTIL
# ============================================================
def normalize_feature_name(name):
    """
    Normalize feature names to match between SHAP and FI
    Example:
        num__BMI -> BMI
    """
    if "__" in name:
        return name.split("__")[-1]
    return name


def compute_topk_overlap(shap_features, fi_features, k):
    shap_topk = shap_features[:k]
    fi_topk = fi_features[:k]

    overlap = len(set(shap_topk).intersection(set(fi_topk)))
    return overlap


# ============================================================
# MAIN
# ============================================================
def main():

    logging.basicConfig(level=logging.INFO)

    logging.info("======================================================================")
    logging.info("RUN FEATURE AGREEMENT ANALYSIS")
    logging.info("======================================================================")

    # -------------------------------------------------------
    # PATHS
    # -------------------------------------------------------
    base_dir = get_outputs_dir()

    shap_path = base_dir / "shap" / f"{ACTIVE_MODEL}_shap_feature_importance.csv"
    fi_path = base_dir / "fi" / f"{ACTIVE_MODEL}_feature_importance.csv"

    logging.info(f"ACTIVE_DATASET = {ACTIVE_DATASET}")
    logging.info(f"ACTIVE_MODEL   = {ACTIVE_MODEL}")
    logging.info(f"SHAP path: {shap_path}")
    logging.info(f"FI path:   {fi_path}")

    # -------------------------------------------------------
    # CHECK FILE
    # -------------------------------------------------------
    if not shap_path.exists():
        raise FileNotFoundError(f"Missing SHAP file: {shap_path}")

    if not fi_path.exists():
        raise FileNotFoundError(f"Missing Feature Importance file: {fi_path}")

    # -------------------------------------------------------
    # LOAD
    # -------------------------------------------------------
    shap_df = pd.read_csv(shap_path)
    fi_df = pd.read_csv(fi_path)

    logging.info(f"Loaded SHAP file with shape: {shap_df.shape}")
    logging.info(f"Loaded FI file with shape:   {fi_df.shape}")

    # -------------------------------------------------------
    # NORMALIZE FEATURE NAMES (🔥 FIX QUAN TRỌNG)
    # -------------------------------------------------------
    shap_df["feature"] = shap_df["feature"].apply(normalize_feature_name)
    fi_df["feature"] = fi_df["feature"].apply(normalize_feature_name)

    # -------------------------------------------------------
    # SORT DESCENDING
    # -------------------------------------------------------
    shap_df = shap_df.sort_values(by="importance", ascending=False)
    fi_df = fi_df.sort_values(by="importance", ascending=False)

    shap_features = shap_df["feature"].tolist()
    fi_features = fi_df["feature"].tolist()

    # -------------------------------------------------------
    # INTERSECTION CHECK
    # -------------------------------------------------------
    common_features = list(set(shap_features).intersection(set(fi_features)))

    if len(common_features) == 0:
        raise ValueError("No overlapping features found between SHAP and FI files.")

    logging.info(f"Number of overlapping features: {len(common_features)}")

    # -------------------------------------------------------
    # ALIGN DATA FOR CORRELATION
    # -------------------------------------------------------
    shap_map = dict(zip(shap_df["feature"], shap_df["importance"]))
    fi_map = dict(zip(fi_df["feature"], fi_df["importance"]))

    shap_vals = []
    fi_vals = []

    for f in common_features:
        shap_vals.append(shap_map[f])
        fi_vals.append(fi_map[f])

    shap_vals = np.array(shap_vals)
    fi_vals = np.array(fi_vals)

    # -------------------------------------------------------
    # SPEARMAN CORRELATION
    # -------------------------------------------------------
    corr, p_value = spearmanr(shap_vals, fi_vals)

    logging.info(f"Spearman correlation: {corr:.4f}")
    logging.info(f"P-value: {p_value:.6f}")

    # -------------------------------------------------------
    # TOP-K OVERLAP
    # -------------------------------------------------------
    overlap_results = []

    for k in TOP_K_LIST:
        overlap = compute_topk_overlap(shap_features, fi_features, k)

        overlap_results.append({
            "k": k,
            "overlap": overlap,
            "overlap_ratio": overlap / k
        })

        logging.info(f"Top-{k} overlap: {overlap}/{k}")

    overlap_df = pd.DataFrame(overlap_results)

    # -------------------------------------------------------
    # SAVE OUTPUT
    # -------------------------------------------------------
    output_dir = base_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "agreement_summary.csv"

    summary_df = pd.DataFrame({
        "spearman_corr": [corr],
        "p_value": [p_value]
    })

    summary_df.to_csv(summary_path, index=False)

    overlap_path = output_dir / "topk_overlap.csv"
    overlap_df.to_csv(overlap_path, index=False)

    logging.info(f"Saved agreement summary to: {summary_path}")
    logging.info(f"Saved top-k overlap to: {overlap_path}")

    logging.info("======================================================================")
    logging.info("FEATURE AGREEMENT ANALYSIS COMPLETED")
    logging.info("======================================================================")


if __name__ == "__main__":
    main()