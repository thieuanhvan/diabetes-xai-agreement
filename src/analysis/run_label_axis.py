"""
Label axis on NHANES: does XAI agreement change when the prediction target
changes from a diagnosis label to a laboratory label?

Design (paired): within each cycle the SAME respondents and the SAME
train/test split (per seed) are used for every label, so only the target
changes. The split is stratified on the joint (diagnosis, HbA1c) cell.

Labels
    diag   Diabetes_self (doctor-diagnosed, self-reported)
    lab    Diabetes_lab_a1c (HbA1c >= 6.5 %)
    total  diag OR lab (epidemiological "total diabetes")

Sample: adults in data/cdc_nhanes_diabetes_<cycle>.csv with a valid HbA1c,
borderline respondents (DIQ010 = 3) excluded, complete on the features.

Features: the NHANES counterparts of the BRFSS predictors (14 columns, see
FEATURES). Laboratory values, treatment status and survey-design variables
are never used as features.

Stages
    grid      fit 3 models x 3 labels x seeds per cycle; write global SHAP and
              PI vectors (unweighted and survey-weighted) + model performance
    pairs     agreement metrics (src.evaluation.agreement_metrics.compare)
              for label pairs, model pairs and method pairs, per seed, plus a
              seed axis (same configuration, different seed) as noise floor
    shift     feature-level change in attribution share between labels,
              corrected resampled t-test across seeds

Usage:
    python -m src.analysis.run_label_axis --stage grid --seeds 10
    python -m src.analysis.run_label_axis --stage pairs
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.agreement_metrics import compare
from src.explainability.global_attribution import MODELS, fit_and_attribute

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "outputs" / "label_axis"

CYCLES = ["2017-2020", "2021-2023"]
LABELS = ["diag", "lab", "total"]
FEATURES = [
    "HighBP", "HighChol", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity_LTPA",          # leisure-time activity; within-cycle use only
    "AnyHealthcare",              # proxy for BRFSS NoDocbcCost (different construct)
    "GenHlth", "MentHlth",        # MentHlth = PHQ-9 severity band in days
    "Sex", "Age", "Education", "Income",
]


def load_cycle(cycle: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"cdc_nhanes_diabetes_{cycle}.csv")
    n0 = len(df)
    df = df[(df["Diabetes_borderline"] == 0) & df["Diabetes_lab_a1c"].notna()]
    df = df.dropna(subset=FEATURES + ["Diabetes_self", "WTMEC"]).copy()
    df["diag"] = df["Diabetes_self"].astype(int)
    df["lab"] = df["Diabetes_lab_a1c"].astype(int)
    df["total"] = (df["diag"] | df["lab"]).astype(int)
    logging.info("%s: %d -> %d rows | prevalence diag %.3f lab %.3f total %.3f",
                 cycle, n0, len(df), df.diag.mean(), df.lab.mean(), df.total.mean())
    return df.reset_index(drop=True)


def run_grid(seeds: int, shap_n: int | None, n_jobs: int, train_weighted: bool = False) -> None:
    vec_rows, perf_rows = [], []
    for cycle in CYCLES:
        df = load_cycle(cycle)
        strata = df["diag"].astype(str) + df["lab"].astype(str)
        for seed in range(seeds):
            tr, te = train_test_split(df.index, test_size=0.2, random_state=seed, stratify=strata)
            Xtr, Xte = df.loc[tr, FEATURES], df.loc[te, FEATURES]
            w_te = df.loc[te, "WTMEC"]
            for label in LABELS:
                for model in MODELS:
                    t0 = time.time()
                    r = fit_and_attribute(model, Xtr, df.loc[tr, label], Xte, df.loc[te, label],
                                          seed=seed, w_test=w_te, shap_n=shap_n, n_jobs=n_jobs,
                                          w_train=df.loc[tr, "WTMEC"] if train_weighted else None)
                    key = dict(cycle=cycle, label=label, model=model, seed=seed)
                    perf_rows.append({**key, "n_train": len(tr), "n_test": len(te),
                                      "pos_test": int(df.loc[te, label].sum()),
                                      "auc": r.auc, "auc_weighted": r.auc_weighted, **r.timings})
                    for method, weighting, vec in [
                        ("SHAP", "unweighted", r.shap_mean_abs),
                        ("SHAP", "weighted", r.shap_mean_abs_weighted),
                        ("PI", "unweighted", r.pi_mean),
                        ("PI", "weighted", r.pi_mean_weighted),
                    ]:
                        for f, v in vec.items():
                            row = {**key, "method": method, "weighting": weighting,
                                   "feature": f, "importance": float(v)}
                            if method == "PI" and weighting == "unweighted":
                                row["pi_std"] = float(r.pi_std[f])
                            vec_rows.append(row)
                    logging.info("%s seed=%d %-5s %-20s AUC=%.3f (%.1fs)",
                                 cycle, seed, label, model, r.auc, time.time() - t0)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(vec_rows).to_csv(RESULT_DIR / "vectors.csv", index=False)
        pd.DataFrame(perf_rows).to_csv(RESULT_DIR / "performance.csv", index=False)
    logging.info("Grid written to %s", RESULT_DIR)


def _vector(g: pd.DataFrame) -> pd.Series:
    """Importance vector; PI is clipped at 0 (a negative drop = no importance)."""
    return g.set_index("feature")["importance"].clip(lower=0.0)


def run_pairs() -> None:
    v = pd.read_csv(RESULT_DIR / "vectors.csv")
    vecs = {k: _vector(g) for k, g in
            v.groupby(["cycle", "seed", "weighting", "method", "model", "label"])}
    rows = []

    def add(axis, a, b, ka, kb, ctx):
        rows.append({"axis": axis, "level_1": a, "level_2": b, **ctx,
                     **compare(vecs[ka], vecs[kb])})

    for (cycle, seed, weighting) in sorted({k[:3] for k in vecs}):
        base = (cycle, seed, weighting)
        for method in ["SHAP", "PI"]:
            for model in MODELS:
                for a, b in combinations(LABELS, 2):
                    add("label", a, b, base + (method, model, a), base + (method, model, b),
                        dict(cycle=cycle, seed=seed, weighting=weighting, method=method, model=model, label=""))
            for label in LABELS:
                for a, b in combinations(MODELS, 2):
                    add("model", a, b, base + (method, a, label), base + (method, b, label),
                        dict(cycle=cycle, seed=seed, weighting=weighting, method=method, model="", label=label))
        for model in MODELS:
            for label in LABELS:
                add("method", "SHAP", "PI", base + ("SHAP", model, label), base + ("PI", model, label),
                    dict(cycle=cycle, seed=seed, weighting=weighting, method="", model=model, label=label))

    # Noise floor: same cycle/weighting/method/model/label, different seed
    # (different split and model randomness). A factor matters only if its
    # disagreement exceeds this floor.
    configs = sorted({(k[0], k[2], k[3], k[4], k[5]) for k in vecs})
    seeds = sorted({k[1] for k in vecs})
    for cycle, weighting, method, model, label in configs:
        for s1, s2 in combinations(seeds, 2):
            add("seed", str(s1), str(s2),
                (cycle, s1, weighting, method, model, label), (cycle, s2, weighting, method, model, label),
                dict(cycle=cycle, seed=-1, weighting=weighting, method=method, model=model, label=label))

    pairs = pd.DataFrame(rows)
    pairs.round(4).to_csv(RESULT_DIR / "pairs.csv", index=False)
    cols = ["J_A", "OC_A", "size_ratio_A", "contradiction_A", "top5",
            "spearman", "kendall_tau_b", "weighted_tau", "rbo_ext_p90"]
    grouped = pairs.groupby(["weighting", "axis"])[cols]
    summary = pd.concat({"mean": grouped.mean(), "p2.5": grouped.quantile(0.025),
                         "p97.5": grouped.quantile(0.975)}, axis=1)
    summary.columns = [f"{c}_{stat}" for stat, c in summary.columns]
    summary.round(4).to_csv(RESULT_DIR / "pairs_summary.csv")
    # Exceedance over the seed noise floor: for each configuration, the share
    # of factor pairs whose metric falls below the 5th percentile of the
    # seed pairs of the same configuration (lower = more disagreement).
    exc_rows = []
    for metric in ["J_A", "spearman", "rbo_ext_p90"]:
        for (cycle, weighting, method, model), seed_g in pairs[pairs.axis == "seed"].groupby(
                ["cycle", "weighting", "method", "model"]):
            floor = seed_g[metric].quantile(0.05)
            lab = pairs[(pairs.axis == "label") & (pairs.cycle == cycle) & (pairs.weighting == weighting)
                        & (pairs.method == method) & (pairs.model == model)]
            exc_rows.append(dict(metric=metric, cycle=cycle, weighting=weighting, method=method,
                                 model=model, seed_p5=floor, label_mean=lab[metric].mean(),
                                 seed_mean=seed_g[metric].mean(),
                                 label_share_below_seed_p5=float((lab[metric] < floor).mean())))
    exc = pd.DataFrame(exc_rows)
    exc.round(4).to_csv(RESULT_DIR / "label_vs_seed_floor.csv", index=False)
    logging.info("Label pairs below the seed 5th percentile (unweighted):\n%s",
                 exc[exc.weighting == "unweighted"].pivot_table(
                     index=["metric", "cycle"], columns="method",
                     values="label_share_below_seed_p5").round(3).to_string())
    logging.info("Pairs: %d rows\n%s", len(pairs),
                 pairs.groupby(["weighting", "axis"])[cols].mean().round(3).to_string())


def run_shift() -> None:
    """Feature-level shift in attribution SHARE between labels.

    For each cycle x weighting x method x model x feature x label pair, the
    per-seed difference in normalised attribution share (importance / sum)
    is tested with the Nadeau-Bengio corrected resampled t-test, which
    inflates the variance by (1/J + n_test/n_train) because the J seeds
    share data. Set-level metrics (J_A, top-K) can miss a shift of this kind
    when the essential set itself does not change.
    """
    from scipy.stats import t as t_dist

    v = pd.read_csv(RESULT_DIR / "vectors.csv")
    v["importance"] = v["importance"].clip(lower=0.0)
    keys = ["cycle", "weighting", "method", "model", "label", "seed"]
    v["share"] = v["importance"] / v.groupby(keys)["importance"].transform("sum")
    wide = v.pivot_table(index=["cycle", "weighting", "method", "model", "seed", "feature"],
                         columns="label", values="share").reset_index()
    perf = pd.read_csv(RESULT_DIR / "performance.csv")
    ratio = (perf["n_test"] / perf["n_train"]).median()

    rows = []
    for a, b in combinations(LABELS, 2):
        wide["d"] = wide[a] - wide[b]
        for key, g in wide.groupby(["cycle", "weighting", "method", "model", "feature"]):
            d = g["d"].to_numpy()
            J = len(d)
            var = d.var(ddof=1)
            se = np.sqrt((1.0 / J + ratio) * var) if var > 0 else np.nan
            tval = d.mean() / se if se and se > 0 else np.nan
            pval = 2 * t_dist.sf(abs(tval), J - 1) if np.isfinite(tval) else np.nan
            rows.append(dict(zip(["cycle", "weighting", "method", "model", "feature"], key),
                             label_1=a, label_2=b, n_seeds=J,
                             mean_share_1=g[a].mean(), mean_share_2=g[b].mean(),
                             diff=d.mean(), t_corrected=tval, p_corrected=pval))
    out = pd.DataFrame(rows)
    # Benjamini-Hochberg within each family: cycle x weighting x method x label pair
    # (14 features x 3 models = 42 tests per family).
    from scipy.stats import false_discovery_control
    fam = ["cycle", "weighting", "method", "label_1", "label_2"]
    out["q_bh"] = np.nan
    for _, idx in out.groupby(fam).groups.items():
        p = out.loc[idx, "p_corrected"]
        ok = p.notna()
        out.loc[p[ok].index, "q_bh"] = false_discovery_control(p[ok].to_numpy(), method="bh")
    out.round(5).to_csv(RESULT_DIR / "feature_shift.csv", index=False)
    sig = out[(out.weighting == "unweighted") & (out.label_1 == "diag") & (out.label_2 == "lab")
              & (out.q_bh < 0.05)]
    logging.info("diag vs lab, unweighted, BH q < 0.05:\n%s",
                 sig.sort_values(["cycle", "method", "diff"])[
                     ["cycle", "method", "model", "feature", "diff", "t_corrected", "p_corrected", "q_bh"]
                 ].round(4).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["grid", "pairs", "shift", "all"], default="all")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--shap-n", type=int, default=None, help="test rows for SHAP (default: all)")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--train-weighted", action="store_true",
                   help="sensitivity analysis: train with the NHANES MEC weight (WTMEC) as "
                        "sample weight; results go to outputs/label_axis_trainweighted/")
    args = p.parse_args()
    global RESULT_DIR
    if args.train_weighted:
        RESULT_DIR = PROJECT_ROOT / "outputs" / "label_axis_trainweighted"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    if args.stage in ("grid", "all"):
        run_grid(args.seeds, args.shap_n, args.n_jobs, args.train_weighted)
    if args.stage in ("pairs", "all"):
        run_pairs()
    if args.stage in ("shift", "all"):
        run_shift()


if __name__ == "__main__":
    main()
