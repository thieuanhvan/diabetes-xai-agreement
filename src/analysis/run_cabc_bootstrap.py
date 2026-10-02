"""
Patient-level bootstrap of Group-A size and cross-model agreement.

Uses the per-patient SHAP values already stored by the boundary-patient
analysis (outputs/boundary_analysis/shap_per_patient/, XGBoost and Logistic
Regression, 2,000 test patients per cohort, seed 42). Models are NOT
retrained, so this quantifies sampling variability over patients only; it
says nothing about variability across training seeds or splits.
Random Forest is not covered because per-patient RF SHAP values were not
stored.

Writes outputs/metric_comparison/:
    bootstrap_group_a.csv     per (cohort, model): P(|A| = k), most frequent
                              membership flips
    bootstrap_xgb_lr.csv      per cohort: distribution of XGB-LR J_A and the
                              95% percentile interval of Spearman rho

Usage:
    python -m src.analysis.run_cabc_bootstrap [--n-boot 2000] [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluation.agreement_metrics import cabc_partition, jaccard

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SHAP_DIR = OUTPUT_DIR / "boundary_analysis" / "shap_per_patient"
RESULT_DIR = OUTPUT_DIR / "metric_comparison"

YEARS = ["2015", "2021", "2023"]
MODELS = ["xgboost", "logistic_regression"]


def load_abs_shap(year: str, model: str) -> tuple[np.ndarray, list[str]]:
    ds = SHAP_DIR / f"cdc_brfss_diabetes_{year}"
    names = [line.strip() for line in (ds / "feature_names.txt").read_text().splitlines() if line.strip()]
    values = np.abs(np.load(ds / f"{model}_shap_values.npy")).astype(float)
    if values.shape[1] != len(names):
        raise ValueError(f"Feature count mismatch in {ds}")
    return values, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    group_rows, pair_rows = [], []
    for year in YEARS:
        data = {m: load_abs_shap(year, m) for m in MODELS}
        names = data[MODELS[0]][1]
        if data[MODELS[1]][1] != names:
            raise ValueError(f"Feature order differs between models for {year}")
        n = data[MODELS[0]][0].shape[0]

        point = {m: cabc_partition(pd.Series(data[m][0].mean(0), index=names)) for m in MODELS}
        sizes = {m: [] for m in MODELS}
        flips = {m: Counter() for m in MODELS}
        ja, rho = [], []

        for _ in range(args.n_boot):
            idx = rng.integers(0, n, n)  # same patients for both models
            boot = {m: pd.Series(data[m][0][idx].mean(0), index=names) for m in MODELS}
            parts = {m: cabc_partition(boot[m]) for m in MODELS}
            for m in MODELS:
                sizes[m].append(parts[m].size_a)
                flips[m].update(set(parts[m].group_a) ^ set(point[m].group_a))
            ja.append(jaccard(parts[MODELS[0]].group_a, parts[MODELS[1]].group_a))
            rho.append(spearmanr(boot[MODELS[0]].values, boot[MODELS[1]].values).statistic)

        for m in MODELS:
            s = np.array(sizes[m])
            row = {"year": year, "model": m, "n_patients": n, "n_boot": args.n_boot,
                   "size_A_point": point[m].size_a, "A_point": ",".join(point[m].group_a)}
            for k in range(int(s.min()), int(s.max()) + 1):
                row[f"P(size_A={k})"] = float((s == k).mean())
            row["top_flips"] = ";".join(f"{f}:{c / args.n_boot:.4f}" for f, c in flips[m].most_common(3))
            group_rows.append(row)

        ja = np.round(np.array(ja), 4)
        pair_rows.append({
            "year": year, "pair": "XGB-LR SHAP", "n_boot": args.n_boot,
            "J_A_point": jaccard(point[MODELS[0]].group_a, point[MODELS[1]].group_a),
            "J_A_distribution": ";".join(f"{v}:{(ja == v).mean():.4f}" for v in sorted(set(ja))),
            "spearman_p2.5": float(np.percentile(rho, 2.5)),
            "spearman_p97.5": float(np.percentile(rho, 97.5)),
        })
        logging.info("Bootstrap done for %s", year)

    g = pd.DataFrame(group_rows)
    p = pd.DataFrame(pair_rows)
    g.to_csv(RESULT_DIR / "bootstrap_group_a.csv", index=False)
    p.to_csv(RESULT_DIR / "bootstrap_xgb_lr.csv", index=False)
    logging.info("Group A size:\n%s", g.to_string(index=False))
    logging.info("XGB-LR pair:\n%s", p.to_string(index=False))


if __name__ == "__main__":
    main()
