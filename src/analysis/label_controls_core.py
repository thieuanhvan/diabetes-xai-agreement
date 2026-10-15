"""
Shared helpers for the outcome-definition control analyses (release v32).

Every alternative label is fitted on the SAME people and the SAME train/test
split as the jbi-v1 label axis: the split is stratified on the ORIGINAL
(diagnosis, second label) cell, never on the alternative label, so a
diag-versus-alternative pair differs only in the target.

Functions
    load_base(outcome, cycle)      analytic sample + original labels + strata
    split(df, seed)                the jbi-v1 split for this seed
    fit_labels(...)                fit 3 models x labels x seeds, unweighted
                                   global SHAP and PI vectors (resumable)
    shift_table(vectors, a, b)     corrected resampled t-test on share
                                   differences + BH within the jbi-v1 families
    replicated(shift)              q < 0.05 in both cycles with the same sign
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from scipy.stats import t as t_dist
from sklearn.model_selection import train_test_split

from src.explainability.global_attribution import MODELS, fit_and_attribute

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"
CYCLES = ["2017-2020", "2021-2023"]

FEATURES = {
    "diabetes": [
        "HighBP", "HighChol", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
        "PhysActivity_LTPA", "AnyHealthcare", "GenHlth", "MentHlth",
        "Sex", "Age", "Education", "Income",
    ],
    "hypertension": [
        "Diabetes_self", "HighChol", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
        "PhysActivity_LTPA", "AnyHealthcare", "GenHlth", "MentHlth",
        "Sex", "Age", "Education", "Income",
    ],
}
SECOND = {"diabetes": "lab", "hypertension": "measured"}
V1_DIR = {"diabetes": OUT_DIR / "label_axis",
          "hypertension": OUT_DIR / "label_axis_hypertension_pipeline"}
TIE_SEED = 20261013


def _matched(score: pd.Series, k: int, seed: int = TIE_SEED) -> pd.Series:
    """1 for exactly the k highest scores; ties broken by a fixed random order."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(score)), -score.to_numpy()))
    out = np.zeros(len(score), dtype=int)
    out[order[:k]] = 1
    return pd.Series(out, index=score.index)


def load_base(outcome: str, cycle: str) -> pd.DataFrame:
    feats = FEATURES[outcome]
    if outcome == "diabetes":
        df = pd.read_csv(DATA_DIR / f"cdc_nhanes_diabetes_{cycle}.csv")
        df = df[(df["Diabetes_borderline"] == 0) & df["Diabetes_lab_a1c"].notna()]
        df = df.dropna(subset=feats + ["Diabetes_self", "WTMEC"]).copy().reset_index(drop=True)
        df["diag"] = df["Diabetes_self"].astype(int)
        df["lab"] = df["Diabetes_lab_a1c"].astype(int)
        k = int(round(len(df) * df["diag"].mean()))
        df["lab63"] = (df["HbA1c"] >= 6.3).astype(int)
        df["labM"] = _matched(df["HbA1c"], k)
        df["labG"] = df["Diabetes_lab"].astype(int)
        df["labTx"] = ((df["lab"] == 1) | (df["Diabetes_treated"] == 1)).astype(int)
    else:
        df = pd.read_csv(DATA_DIR / f"cdc_nhanes_hypertension_{cycle}.csv")
        df = df[df["Diabetes_borderline"] == 0]
        df = df.dropna(subset=feats + ["HighBP", "SBP_mean", "DBP_mean", "WTMEC"]).copy().reset_index(drop=True)
        df["diag"] = df["HighBP"].astype(int)
        df["measured"] = ((df["SBP_mean"] >= 140) | (df["DBP_mean"] >= 90)).astype(int)
        k = int(round(len(df) * df["diag"].mean()))
        df["meas130"] = ((df["SBP_mean"] >= 130) | (df["DBP_mean"] >= 80)).astype(int)
        df["measM"] = _matched(np.maximum(df["SBP_mean"] / 140.0, df["DBP_mean"] / 90.0), k)
    df["strata"] = df["diag"].astype(str) + df[SECOND[outcome]].astype(str)
    return df


def split(df: pd.DataFrame, seed: int):
    return train_test_split(df.index, test_size=0.2, random_state=seed, stratify=df["strata"])


def fit_labels(outcome: str, labels: list[str], seeds: list[int], out_csv: Path,
               class_weighting: str = "pipeline", shap_n_rf: int | None = None,
               label_frames: dict | None = None, models=MODELS, cycles=CYCLES) -> pd.DataFrame:
    """Fit and store unweighted global SHAP / PI vectors. Resumable: keys
    already in ``out_csv`` are skipped. ``label_frames`` maps cycle -> DataFrame
    of extra label columns (e.g. null labels) aligned with load_base()."""
    feats = FEATURES[outcome]
    done = set()
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        done = set(map(tuple, old[["cycle", "label", "model", "seed"]].astype(str).to_numpy()))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    for cycle in cycles:
        df = load_base(outcome, cycle)
        if label_frames is not None:
            df = pd.concat([df, label_frames[cycle]], axis=1)
        for seed in seeds:
            tr, te = split(df, seed)
            for label in labels:
                rows = []
                for model in models:
                    if (cycle, label, model, str(seed)) in done:
                        continue
                    t0 = time.time()
                    r = fit_and_attribute(model, df.loc[tr, feats], df.loc[tr, label],
                                          df.loc[te, feats], df.loc[te, label], seed=seed,
                                          shap_n=shap_n_rf if model == "random_forest" else None,
                                          class_weighting=class_weighting, shap_jobs=2)
                    for method, vec in [("SHAP", r.shap_mean_abs), ("PI", r.pi_mean)]:
                        for f, val in vec.items():
                            rows.append(dict(cycle=cycle, label=label, model=model, seed=seed,
                                             method=method, weighting="unweighted", feature=f,
                                             importance=float(val), auc=r.auc,
                                             n_train=len(tr), n_test=len(te)))
                    logging.info("%s %s seed=%d %-8s %-20s AUC=%.3f (%.0fs)", outcome, cycle, seed,
                                 label, model, r.auc, time.time() - t0)
                if rows:
                    pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
    return pd.read_csv(out_csv)


def v1_vectors(outcome: str, labels=("diag",), seeds=None) -> pd.DataFrame:
    """Stored jbi-v1 unweighted vectors (reproduced to 1e-16 in this release)."""
    v = pd.read_csv(V1_DIR[outcome] / "vectors.csv")
    v = v[(v.weighting == "unweighted") & v.label.isin(labels)]
    if seeds is not None:
        v = v[v.seed.isin(seeds)]
    perf = pd.read_csv(V1_DIR[outcome] / "performance.csv")
    v = v.merge(perf[["cycle", "label", "model", "seed", "auc", "n_train", "n_test"]],
                on=["cycle", "label", "model", "seed"], how="left")
    return v[["cycle", "label", "model", "seed", "method", "weighting", "feature",
              "importance", "auc", "n_train", "n_test"]]


def shift_table(v: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Same statistic as run_label_axis.run_shift (unweighted only): per-seed
    difference in normalised share a - b, Nadeau-Bengio corrected t-test, BH
    within cycle x method (42 tests)."""
    v = v[v.label.isin([a, b])].copy()
    v["importance"] = v["importance"].clip(lower=0.0)
    keys = ["cycle", "method", "model", "label", "seed"]
    v["share"] = v["importance"] / v.groupby(keys)["importance"].transform("sum")
    wide = v.pivot_table(index=["cycle", "method", "model", "seed", "feature"],
                         columns="label", values="share").reset_index().dropna(subset=[a, b])
    ratio = (v["n_test"] / v["n_train"]).median()
    rows = []
    wide["d"] = wide[a] - wide[b]
    for key, g in wide.groupby(["cycle", "method", "model", "feature"]):
        d = g["d"].to_numpy()
        J = len(d)
        var = d.var(ddof=1)
        se = np.sqrt((1.0 / J + ratio) * var) if var > 0 else np.nan
        tval = d.mean() / se if se and se > 0 else np.nan
        pval = 2 * t_dist.sf(abs(tval), J - 1) if np.isfinite(tval) else np.nan
        rows.append(dict(zip(["cycle", "method", "model", "feature"], key), label_1=a, label_2=b,
                         n_seeds=J, mean_share_1=g[a].mean(), mean_share_2=g[b].mean(),
                         diff=d.mean(), t_corrected=tval, p_corrected=pval))
    out = pd.DataFrame(rows)
    out["q_bh"] = np.nan
    for _, idx in out.groupby(["cycle", "method"]).groups.items():
        p = out.loc[idx, "p_corrected"]
        ok = p.notna()
        out.loc[p[ok].index, "q_bh"] = false_discovery_control(p[ok].to_numpy(), method="bh")
    return out


def replicated(shift: pd.DataFrame) -> pd.DataFrame:
    """Shifts with BH q < 0.05 in both cycles and the same sign."""
    s = shift[shift.q_bh < 0.05]
    g = s.groupby(["method", "model", "feature"]).agg(
        n_cycles=("cycle", "nunique"), n_pos=("diff", lambda x: int((x > 0).sum())))
    g = g[(g.n_cycles == 2) & g.n_pos.isin([0, 2])].reset_index()
    g["sign"] = np.where(g.n_pos == 2, 1, -1)
    return g[["method", "model", "feature", "sign"]]


def v1_replicated(outcome: str) -> pd.DataFrame:
    """The replicated shifts of jbi-v1 (13 for diabetes, 34 for hypertension)."""
    fs = pd.read_csv(V1_DIR[outcome] / "feature_shift.csv")
    fs = fs[(fs.weighting == "unweighted") & (fs.label_1 == "diag") & (fs.label_2 == SECOND[outcome])]
    rep = replicated(fs)
    d = fs.pivot_table(index=["method", "model", "feature"], columns="cycle", values="diff").reset_index()
    d.columns = ["method", "model", "feature", "diff_v1_1720", "diff_v1_2123"]
    return rep.merge(d, on=["method", "model", "feature"])


def survival(base: pd.DataFrame, shift: pd.DataFrame, tag: str) -> pd.DataFrame:
    """For each base shift: same sign and q < 0.05 in both cycles under ``shift``."""
    s = shift.copy()
    s["ok"] = s.q_bh < 0.05
    out = base.copy()
    flags, d1, d2 = [], [], []
    for _, r in base.iterrows():
        g = s[(s.method == r.method) & (s.model == r.model) & (s.feature == r.feature)].set_index("cycle")
        ok = all(c in g.index and g.loc[c, "ok"] and np.sign(g.loc[c, "diff"]) == r.sign for c in CYCLES)
        flags.append(bool(ok))
        d1.append(g.loc[CYCLES[0], "diff"] if CYCLES[0] in g.index else np.nan)
        d2.append(g.loc[CYCLES[1], "diff"] if CYCLES[1] in g.index else np.nan)
    out[f"survive_{tag}"] = flags
    out[f"diff_{tag}_1720"] = d1
    out[f"diff_{tag}_2123"] = d2
    return out
