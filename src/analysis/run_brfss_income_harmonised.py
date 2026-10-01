"""
Correction to the MAPR 2026 Income-fairness result (no retraining).

BRFSS codes household income in 8 bins (2015), 11 bins (2021) and 7 bins
(2023). The MAPR fairness gap delta-TPR = max - min over bins therefore
depends mechanically on the number of bins. Here the per-bin counts already
stored by the MAPR pipeline (outputs/<cohort>/analysis/<model>_fairness_
income_detail.csv: n, n_positive, TPR) are pooled into five intervals
common to all three releases, and delta-TPR is recomputed:

    <$15K, $15-25K, $25-35K, $35-50K, >=$50K

TP per bin = TPR x n_positive, so pooled TPR = sum(TP) / sum(n_positive).

Writes outputs/brfss_income_harmonised/delta_tpr.csv
Usage:
    python -m src.analysis.run_brfss_income_harmonised
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs"
RESULT_DIR = OUT / "brfss_income_harmonised"
MODELS = ["xgboost", "random_forest", "logistic_regression"]

COMMON = {
    # 2015 (8 bins)
    "<$10K": "<$15K", "$10-15K": "<$15K", "$15-20K": "$15-25K", "$20-25K": "$15-25K",
    "$25-35K": "$25-35K", "$35-50K": "$35-50K", "$50-75K": ">=$50K", "$75K+": ">=$50K",
    # 2021 additions (11 bins)
    "$75-100K": ">=$50K", "$100-150K": ">=$50K", "$150-200K": ">=$50K", ">$200K": ">=$50K",
    # 2023 (7 bins)
    "<$15K": "<$15K", "$15-25K": "$15-25K", "$50-100K": ">=$50K", "$100-200K": ">=$50K",
    "$200K+": ">=$50K",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    rows = []
    for year in ["2015", "2021", "2023"]:
        for model in MODELS:
            d = pd.read_csv(OUT / f"cdc_brfss_diabetes_{year}" / "analysis" /
                            f"{model}_fairness_income_detail.csv")
            unknown = set(d["group_label"]) - set(COMMON)
            if unknown:
                raise ValueError(f"Unmapped income bins in {year}/{model}: {unknown}")
            d["tp"] = d["TPR"] * d["n_positive"]
            d["common"] = d["group_label"].map(COMMON)
            h = d.groupby("common")[["tp", "n_positive"]].sum()
            h["TPR"] = h["tp"] / h["n_positive"]
            rows.append(dict(year=year, model=model, n_bins_native=len(d),
                             delta_tpr_native=d["TPR"].max() - d["TPR"].min(),
                             delta_tpr_harmonised=h["TPR"].max() - h["TPR"].min()))
    res = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(RESULT_DIR / "delta_tpr.csv", index=False)
    wide = res.pivot(index="model", columns="year", values=["delta_tpr_native", "delta_tpr_harmonised"])
    logging.info("Income delta-TPR, native vs harmonised (5 common bins):\n%s", wide.round(3).to_string())


if __name__ == "__main__":
    main()
