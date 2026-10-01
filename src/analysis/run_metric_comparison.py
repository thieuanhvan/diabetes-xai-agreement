"""
Post-MAPR metric comparison on the 18 existing attribution vectors.

Answers the question both MAPR 2026 reviewers raised: is cABC Group-A
Jaccard (J_A) informative beyond top-K overlap, Spearman, Kendall and RBO?
No model is retrained; inputs are the 18 global importance CSVs
(3 cohorts x 3 models x {SHAP, PI}) already under outputs/<dataset>/.

Comparison axes (45 pairs):
    cross-method      SHAP vs PI, same cohort and model           9 pairs
    cross-model-SHAP  model pairs, SHAP                           9 pairs
    cross-model-PI    model pairs, PI (absent from MAPR)          9 pairs
    cross-year-SHAP   cohort pairs, SHAP                          9 pairs
    cross-year-PI     cohort pairs, PI                            9 pairs

Writes outputs/metric_comparison/:
    pairs.csv            one row per pair, all metrics
    axis_means.csv       mean of each metric per axis
    vectors.csv          per-vector Group A size, boundary features and their
                         importance relative to the mean importance
    metric_rank_corr.csv Spearman correlation between metrics across pairs

Usage:
    python -m src.analysis.run_metric_comparison
"""

from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.evaluation.agreement_metrics import cabc_partition, compare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULT_DIR = OUTPUT_DIR / "metric_comparison"

YEARS = ["2015", "2021", "2023"]
MODELS = ["xgboost", "random_forest", "logistic_regression"]
METHODS = ["SHAP", "PI"]
MODEL_LABEL = {"xgboost": "XGB", "random_forest": "RF", "logistic_regression": "LR"}

Key = Tuple[str, str, str]  # (year, model, method)


def load_vector(year: str, model: str, method: str) -> pd.Series:
    """Global importance vector as Series[feature -> |importance|]."""
    ds = OUTPUT_DIR / f"cdc_brfss_diabetes_{year}"
    if method == "SHAP":
        path = ds / "shap" / f"{model}_shap_feature_importance.csv"
    else:
        path = ds / "fi" / f"{model}_feature_importance.csv"
    df = pd.read_csv(path)
    s = pd.Series(pd.to_numeric(df["importance"]).abs().to_numpy(), index=df["feature"].astype(str))
    if s.index.duplicated().any():
        raise ValueError(f"Duplicate features in {path}")
    return s.sort_values(ascending=False)


def load_all() -> Dict[Key, pd.Series]:
    vectors = {(y, m, k): load_vector(y, m, k) for y in YEARS for m in MODELS for k in METHODS}
    logging.info("Loaded %d importance vectors", len(vectors))
    return vectors


def build_pairs(v: Dict[Key, pd.Series]) -> pd.DataFrame:
    rows = []

    def add(axis: str, label: str, k1: Key, k2: Key) -> None:
        rows.append({"axis": axis, "pair": label,
                     "run_1": "/".join(k1), "run_2": "/".join(k2), **compare(v[k1], v[k2])})

    for y in YEARS:
        for m in MODELS:
            add("cross-method", f"{y} {MODEL_LABEL[m]} SHAP-PI", (y, m, "SHAP"), (y, m, "PI"))
    for k in METHODS:
        for y in YEARS:
            for m1, m2 in combinations(MODELS, 2):
                add(f"cross-model-{k}", f"{y} {MODEL_LABEL[m1]}-{MODEL_LABEL[m2]} {k}", (y, m1, k), (y, m2, k))
    for k in METHODS:
        for m in MODELS:
            for y1, y2 in combinations(YEARS, 2):
                add(f"cross-year-{k}", f"{MODEL_LABEL[m]} {k} {y1}-{y2}", (y1, m, k), (y2, m, k))
    return pd.DataFrame(rows)


def build_vectors(v: Dict[Key, pd.Series]) -> pd.DataFrame:
    rows = []
    for (y, m, k), s in v.items():
        part = cabc_partition(s)
        above_mean = set(s[s > part.mean_importance].index)
        last_a, first_b = part.group_a[-1], (part.group_b or part.group_c or [None])[0]
        rows.append({
            "year": y, "model": m, "method": k,
            "size_A": part.size_a,
            "A": ",".join(part.group_a),
            "A_equals_above_mean": int(set(part.group_a) == above_mean),
            "last_A": last_a,
            "last_A_over_mean": s[last_a] / part.mean_importance,
            "first_B": first_b,
            "first_B_over_mean": s[first_b] / part.mean_importance if first_b else float("nan"),
            "top5": ",".join(s.index[:5]),
        })
    return pd.DataFrame(rows)


METRIC_COLS = ["J_A", "OC_A", "size_ratio_A", "contradiction_A",
               "top5", "top6", "top10", "spearman", "kendall_tau_b", "weighted_tau",
               "rbo_ext_p90", "rbo_ext_p80", "rbo_ext_p50"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    v = load_all()
    pairs = build_pairs(v)
    vectors = build_vectors(v)

    axis_means = pairs.groupby("axis", sort=False)[METRIC_COLS].mean()
    axis_means.insert(0, "n_pairs", pairs.groupby("axis", sort=False).size())
    rank_corr = pairs[["J_A", "top6", "top10", "spearman", "kendall_tau_b",
                       "weighted_tau", "rbo_ext_p90", "rbo_ext_p50"]].corr(method="spearman")

    pairs.round(4).to_csv(RESULT_DIR / "pairs.csv", index=False)
    axis_means.round(4).to_csv(RESULT_DIR / "axis_means.csv")
    vectors.round(4).to_csv(RESULT_DIR / "vectors.csv", index=False)
    rank_corr.round(4).to_csv(RESULT_DIR / "metric_rank_corr.csv")

    n_lt1 = int((pairs["J_A"] < 1).sum())
    n_contra = int(pairs["contradiction_A"].sum())
    n_ja_eq_ratio = int((pairs["J_A"] - pairs["size_ratio_A"]).abs().lt(1e-12).sum())
    logging.info("Pairs: %d | J_A < 1: %d | contradictions: %d | J_A == size ratio: %d",
                 len(pairs), n_lt1, n_contra, n_ja_eq_ratio)
    logging.info("Distinct top-5 sets across vectors: %d", vectors["top5"].map(lambda t: frozenset(t.split(","))).nunique())
    logging.info("Group A equals above-mean set: %d / %d", vectors["A_equals_above_mean"].sum(), len(vectors))
    logging.info("Axis means:\n%s", axis_means.round(4).to_string())
    logging.info("Outputs written to %s", RESULT_DIR)


if __name__ == "__main__":
    main()
