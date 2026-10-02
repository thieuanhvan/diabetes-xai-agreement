"""
BRFSS multi-seed attribution grid: the MAPR factorial (3 cohorts x 3 models
x {SHAP, PI}) re-run over several seeds, so that every agreement number gets
a seed noise floor and a spread. Same harness as the NHANES label axis
(src.explainability.global_attribution).

Each seed changes the stratified 80/20 split and the model randomness.
Features: the 17-predictor common schema of the MAPR paper.

Cost control on ~250k rows per cohort (documented deviations from MAPR):
    --shap-n     test rows used for SHAP, XGBoost and LR (default 2000; MAPR 200)
    --shap-n-rf  test rows used for SHAP, Random Forest (default 200 = MAPR)
    --pi-n    stratified test rows used for Permutation Importance
              (default 10000; MAPR used the full test set)

Class-weighting ablation (reviewer R#1, cross-model confound):
    --class-weighting pipeline | none | balanced

Stages
    cw-compare  compare model/method agreement across the class-weighting
                regimes (needs pairs.csv of at least two regimes)
    grid   write outputs/brfss_multiseed/<class_weighting>/vectors.csv and
           performance.csv
    pairs  agreement for model pairs, method pairs, cohort pairs and the
           seed axis (same configuration, different seed)

Usage:
    python -m src.analysis.run_brfss_multiseed --stage all --seeds 5
    python -m src.analysis.run_brfss_multiseed --stage all --seeds 5 --class-weighting none
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.agreement_metrics import compare
from src.explainability.global_attribution import CLASS_WEIGHTING, MODELS, fit_and_attribute

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_ROOT = PROJECT_ROOT / "outputs" / "brfss_multiseed"

YEARS = ["2015", "2021", "2023"]
TARGET = "Diabetes_binary"
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income",
]


def run_grid(out_dir: Path, seeds: int, shap_n: int, shap_n_rf: int, pi_n: int, n_jobs: int,
             cw: str, models: list[str], shap_jobs: int) -> None:
    """Resumable: results are appended after every (cohort, seed, model) run
    and runs already present in performance.csv are skipped, so an
    interrupted job can simply be restarted with the same arguments."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path, perf_path = out_dir / "vectors.csv", out_dir / "performance.csv"
    vec_rows = pd.read_csv(vec_path, dtype={"year": str}).to_dict("records") if vec_path.exists() else []
    perf_rows = pd.read_csv(perf_path, dtype={"year": str}).to_dict("records") if perf_path.exists() else []
    done = {(str(r["year"]), int(r["seed"]), r["model"]) for r in perf_rows}
    for year in YEARS:
        todo = [(s, m) for s in range(seeds) for m in models if (year, s, m) not in done]
        if not todo:
            continue
        df = pd.read_csv(DATA_DIR / f"cdc_brfss_diabetes_{year}.csv", usecols=FEATURES + [TARGET])
        y = df[TARGET].astype(int)
        for seed in sorted({s for s, _ in todo}):
            Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], y, test_size=0.2,
                                                  random_state=seed, stratify=y)
            for model in [m for s, m in todo if s == seed]:
                t0 = time.time()
                is_rf = model == "random_forest"
                r = fit_and_attribute(model, Xtr, ytr, Xte, yte, seed=seed,
                                      shap_n=shap_n_rf if is_rf else shap_n,
                                      pi_n=pi_n, n_jobs=n_jobs, class_weighting=cw,
                                      shap_jobs=shap_jobs)
                key = dict(year=year, model=model, seed=seed, class_weighting=cw)
                perf_rows.append({**key, "n_train": len(Xtr), "n_test": len(Xte),
                                  "auc": r.auc, **r.timings})
                for method, vec in [("SHAP", r.shap_mean_abs), ("PI", r.pi_mean)]:
                    for f, v in vec.items():
                        row = {**key, "method": method, "feature": f, "importance": float(v)}
                        if method == "PI":
                            row["pi_std"] = float(r.pi_std[f])
                        vec_rows.append(row)
                pd.DataFrame(vec_rows).to_csv(vec_path, index=False)
                pd.DataFrame(perf_rows).to_csv(perf_path, index=False)
                logging.info("%s seed=%d %-20s AUC=%.3f (%.0fs: fit %.0f, shap %.0f, pi %.0f)",
                             year, seed, model, r.auc, time.time() - t0,
                             r.timings["fit_s"], r.timings["shap_s"], r.timings["pi_s"])
    logging.info("Grid written to %s", out_dir)


def run_pairs(out_dir: Path) -> None:
    v = pd.read_csv(out_dir / "vectors.csv", dtype={"year": str})
    vecs = {k: g.set_index("feature")["importance"].clip(lower=0.0)
            for k, g in v.groupby(["year", "seed", "method", "model"])}
    rows = []

    def add(axis, a, b, ka, kb, ctx):
        rows.append({"axis": axis, "level_1": a, "level_2": b, **ctx, **compare(vecs[ka], vecs[kb])})

    seeds = sorted({k[1] for k in vecs})
    for year in YEARS:
        for seed in seeds:
            for model in MODELS:
                add("method", "SHAP", "PI", (year, seed, "SHAP", model), (year, seed, "PI", model),
                    dict(year=year, seed=seed, method="", model=model))
            for method in ["SHAP", "PI"]:
                for a, b in combinations(MODELS, 2):
                    add("model", a, b, (year, seed, method, a), (year, seed, method, b),
                        dict(year=year, seed=seed, method=method, model=""))
    for seed in seeds:
        for method in ["SHAP", "PI"]:
            for model in MODELS:
                for a, b in combinations(YEARS, 2):
                    add("year", a, b, (a, seed, method, model), (b, seed, method, model),
                        dict(year="", seed=seed, method=method, model=model))
    for year in YEARS:
        for method in ["SHAP", "PI"]:
            for model in MODELS:
                for s1, s2 in combinations(seeds, 2):
                    add("seed", str(s1), str(s2), (year, s1, method, model), (year, s2, method, model),
                        dict(year=year, seed=-1, method=method, model=model))

    pairs = pd.DataFrame(rows)
    pairs.to_csv(out_dir / "pairs.csv", index=False)
    cols = ["J_A", "OC_A", "size_ratio_A", "contradiction_A", "top5",
            "spearman", "kendall_tau_b", "weighted_tau", "rbo_ext_p90"]
    g = pairs.groupby("axis")[cols]
    summary = pd.concat({"mean": g.mean(), "p2.5": g.quantile(0.025), "p97.5": g.quantile(0.975)}, axis=1)
    summary.columns = [f"{c}_{stat}" for stat, c in summary.columns]
    summary.to_csv(out_dir / "pairs_summary.csv")
    by_method = pairs[pairs.axis.isin(["model", "year", "seed"])].groupby(["axis", "method"])[cols].mean()
    logging.info("Pairs: %d\n%s\n\nBy method:\n%s", len(pairs), g.mean().round(3).to_string(),
                 by_method.round(3).to_string())


def compare_class_weighting() -> None:
    """Reviewer R#1: the MAPR cross-model comparison mixes architecture with
    class weighting (LR and RF balanced, XGBoost not). Compare cross-model
    and cross-method agreement under the three weighting regimes on the
    seeds they share. If the cross-model gap persists when every model uses
    the same weighting, it is not a class-weighting artefact."""
    frames = []
    for cw in CLASS_WEIGHTING:
        f = OUT_ROOT / cw / "pairs.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["class_weighting"] = cw
            frames.append(d)
    if len(frames) < 2:
        logging.warning("Need pairs.csv for at least two class-weighting regimes; found %d", len(frames))
        return
    pairs = pd.concat(frames, ignore_index=True)
    common = set.intersection(*[set(f["seed"]) for f in frames]) - {-1}
    pairs = pairs[pairs.axis.isin(["model", "method"]) & pairs.seed.isin(common)]
    cols = ["J_A", "OC_A", "contradiction_A", "spearman", "kendall_tau_b", "rbo_ext_p90"]
    summary = pairs.groupby(["class_weighting", "axis", "method"], dropna=False)[cols].mean()
    by_pair = pairs[pairs.axis == "model"].assign(pair=lambda d: d.level_1 + "-" + d.level_2) \
        .groupby(["class_weighting", "method", "pair"])[["J_A", "spearman", "rbo_ext_p90"]].mean()
    out = OUT_ROOT / "class_weighting_comparison"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "axis_summary.csv")
    by_pair.to_csv(out / "model_pairs.csv")
    logging.info("Seeds compared: %s\n%s\n\n%s", sorted(common), summary.round(3).to_string(),
                 by_pair.round(3).to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["grid", "pairs", "all", "cw-compare"], default="all")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--shap-n", type=int, default=2000)
    p.add_argument("--shap-n-rf", type=int, default=200,
                   help="SHAP rows for Random Forest (exact TreeSHAP on ~38k-leaf trees "
                        "costs ~5 s per row; 200 = MAPR setting)")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--shap-jobs", type=int, default=1,
                   help="processes for TreeSHAP; each holds a full copy of the model "
                        "(an unconstrained BRFSS forest is ~1-2 GB), so keep 1 on small machines")
    p.add_argument("--pi-n", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--class-weighting", choices=CLASS_WEIGHTING, default="pipeline")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    out_dir = OUT_ROOT / args.class_weighting
    if args.stage in ("grid", "all"):
        run_grid(out_dir, args.seeds, args.shap_n, args.shap_n_rf, args.pi_n, args.n_jobs,
                 args.class_weighting, args.models, args.shap_jobs)
    if args.stage in ("pairs", "all"):
        run_pairs(out_dir)
    if args.stage == "cw-compare":
        compare_class_weighting()


if __name__ == "__main__":
    main()
