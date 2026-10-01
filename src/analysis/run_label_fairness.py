"""
Fairness consequence of the label axis on NHANES.

Question: a screening model trained on the DIAGNOSIS label learns who has
already been diagnosed. Does it then miss laboratory-confirmed diabetes
unevenly across socioeconomic and demographic groups?

Design (same paired samples and splits as run_label_axis.py):
    train label  L in {diag, lab, total}
    truth        T in {diag, lab, total}     (every L is scored against every T)
    threshold    capacity-matched: the top q of test predictions are flagged,
                 q = prevalence of L in the training set (a screening
                 programme sized to the label it was built for)
    groups       fixed two- or three-level contrasts, identical in both
                 cycles, so the gap does not depend on the number of bins
                 (the MAPR Income artefact):
                   Income     PIR < 2 (bands 1-2)  vs  PIR >= 2 (bands 3-5)
                   Education  high school or less  vs  some college or more
                   Sex        female vs male
                   Age        18-44, 45-64, 65+   (gap = max - min)
    metrics      TPR per group (weighted by the MEC exam weight WTMEC and
                 unweighted), TPR gap, group AUC

Outputs (outputs/label_fairness/):
    group_metrics.csv   one row per cycle x seed x model x L x T x attribute x group
    gaps.csv            TPR gap per cycle x seed x model x L x T x attribute
    gaps_summary.csv    mean and 2.5/97.5 percentiles across seeds

Usage:
    python -m src.analysis.run_label_fairness --seeds 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.analysis.run_label_axis import CYCLES, FEATURES, LABELS, load_cycle
from src.explainability.global_attribution import MODELS, build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "outputs" / "label_fairness"


def groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    age = pd.cut(df["Age"], [17, 44, 64, 200], labels=["18-44", "45-64", "65+"]).astype(str)
    return {
        "Income": np.where(df["Income"] <= 2, "PIR<2", "PIR>=2"),
        "Education": np.where(df["Education"] <= 4, "HS_or_less", "some_college+"),
        "Sex": np.where(df["Sex"] == 1, "male", "female"),
        "Age": age.to_numpy(),
    }


def _wtpr(y: np.ndarray, flag: np.ndarray, w: np.ndarray) -> float:
    pos = y == 1
    return float((w[pos] * flag[pos]).sum() / w[pos].sum()) if pos.any() else np.nan


def run(seeds: int) -> None:
    rows = []
    for cycle in CYCLES:
        df = load_cycle(cycle)
        strata = df["diag"].astype(str) + df["lab"].astype(str)
        grp_all = groups(df)
        for seed in range(seeds):
            tr, te = train_test_split(df.index, test_size=0.2, random_state=seed, stratify=strata)
            w = df.loc[te, "WTMEC"].to_numpy(dtype=float)
            for model in MODELS:
                for L in LABELS:
                    ytr = df.loc[tr, L]
                    pipe = build_model(model, seed, n_jobs=1).fit(df.loc[tr, FEATURES], ytr)
                    prob = pipe.predict_proba(df.loc[te, FEATURES])[:, 1]
                    q = float(ytr.mean())
                    flag = (prob >= np.quantile(prob, 1.0 - q)).astype(float)
                    for T in LABELS:
                        y = df.loc[te, T].to_numpy()
                        for attr, g in grp_all.items():
                            gte = np.asarray(g)[np.asarray(te)]
                            for level in sorted(set(gte)):
                                m = gte == level
                                rows.append(dict(
                                    cycle=cycle, seed=seed, model=model, train_label=L, truth=T,
                                    attribute=attr, group=level, n=int(m.sum()), n_pos=int(y[m].sum()),
                                    tpr=float(flag[m][y[m] == 1].mean()) if y[m].sum() else np.nan,
                                    tpr_weighted=_wtpr(y[m], flag[m], w[m]),
                                    fpr=float(flag[m][y[m] == 0].mean()),
                                    auc=float(roc_auc_score(y[m], prob[m])) if 0 < y[m].sum() < m.sum() else np.nan,
                                ))
            logging.info("%s seed=%d done", cycle, seed)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    gm = pd.DataFrame(rows)
    gm.round(5).to_csv(RESULT_DIR / "group_metrics.csv", index=False)

    keys = ["cycle", "seed", "model", "train_label", "truth", "attribute"]
    gaps = gm.groupby(keys).agg(
        tpr_gap=("tpr", lambda s: s.max() - s.min()),
        tpr_gap_weighted=("tpr_weighted", lambda s: s.max() - s.min()),
        min_group_n_pos=("n_pos", "min"),
    ).reset_index()
    # Signed gap for the two-level socioeconomic contrasts (disadvantaged minus advantaged)
    piv = gm.pivot_table(index=keys, columns="group", values="tpr_weighted")
    gaps = gaps.merge(pd.DataFrame({
        "signed_gap_weighted": np.where(
            piv.index.get_level_values("attribute") == "Income", piv.get("PIR<2") - piv.get("PIR>=2"),
            np.where(piv.index.get_level_values("attribute") == "Education",
                     piv.get("HS_or_less") - piv.get("some_college+"), np.nan)),
    }, index=piv.index).reset_index(), on=keys, how="left")
    gaps.round(5).to_csv(RESULT_DIR / "gaps.csv", index=False)

    s_keys = ["cycle", "model", "train_label", "truth", "attribute"]
    g = gaps.groupby(s_keys)[["tpr_gap_weighted", "signed_gap_weighted"]]
    summary = pd.concat({"mean": g.mean(), "p2.5": g.quantile(0.025), "p97.5": g.quantile(0.975)}, axis=1)
    summary.columns = [f"{c}_{stat}" for stat, c in summary.columns]
    summary.round(4).to_csv(RESULT_DIR / "gaps_summary.csv")
    view = gaps[gaps.attribute.isin(["Income", "Education"])].groupby(
        ["cycle", "attribute", "train_label", "truth"])["signed_gap_weighted"].mean().unstack("truth")
    logging.info("Signed weighted TPR gap (disadvantaged - advantaged), mean over models x seeds:\n%s",
                 view.round(3).to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    run(args.seeds)


if __name__ == "__main__":
    main()
