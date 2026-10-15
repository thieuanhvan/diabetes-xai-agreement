"""
Random-discordance null for the outcome-definition shifts (v32, Spec A.3).

Question: if a second label disagreed with the diagnosis label for exactly as
many people as the real second label does, but for randomly chosen people,
how large would the attribution-share shifts be?

null_r (r = 1..R), per cycle: start from diag; flip exactly n10 random
diag = 1 people to 0 and n01 random diag = 0 people to 1, where n10 and n01
are the real discordant counts of (diag, second label). null_r therefore has
the same prevalence and the same 2 x 2 table with diag as the real label.
RNG: default_rng(1000 + r) per cycle offset; flipped indices are stored.

Fits: seeds 0-4, three models, unweighted SHAP and PI; random-forest SHAP
on 500 test rows (applied to diag, the real label and every null label alike).

Outputs: outputs/label_null/<outcome>/
    null_flips.csv          r, cycle, row index flipped
    vectors.csv             fitted vectors (diag, real label, null_r)
    null_vs_observed.csv    observed diag - label shift vs the null distribution
    survival.csv            jbi-v1 replicated shifts: exceed the null p95 in both cycles?
Usage:
    python -m src.analysis.run_label_null --outcome diabetes --R 50
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from src.analysis import label_controls_core as C

SEEDS = list(range(5))
SHAP_N_RF = 500


def diag_risk_logit(df: pd.DataFrame, outcome: str) -> np.ndarray:
    """Out-of-fold logit of P(diag = 1) from a logistic regression on the
    predictors (5-fold, stratified, seed 0); used only to place the
    discordance of the risk-structured null, never as a predictor."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    p = cross_val_predict(model, df[C.FEATURES[outcome]], df["diag"], method="predict_proba",
                          cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def null_frames(outcome: str, R: int, dst, gamma: float = 0.0) -> dict:
    """gamma = 0: random discordance (Spec A.3). gamma > 0: risk-structured
    discordance - diag = 1 people are flipped to 0 with weight exp(-gamma *
    logit), diag = 0 people to 1 with weight exp(+gamma * logit), so flips
    preferentially involve diagnosed positives with lower and diagnosed
    negatives with higher predicted diagnosis risk; the 2 x 2 table with diag
    is the same as for the real label."""
    frames, flips = {}, []
    for ci, cycle in enumerate(C.CYCLES):
        df = C.load_base(outcome, cycle)
        second = C.SECOND[outcome]
        n10 = int(((df.diag == 1) & (df[second] == 0)).sum())
        n01 = int(((df.diag == 0) & (df[second] == 1)).sum())
        pos, neg = np.flatnonzero(df.diag == 1), np.flatnonzero(df.diag == 0)
        w_pos = w_neg = None
        if gamma > 0:
            z = diag_risk_logit(df, outcome)
            w_pos = np.exp(-gamma * (z[pos] - z[pos].max()))
            w_neg = np.exp(gamma * (z[neg] - z[neg].max()))
            w_pos, w_neg = w_pos / w_pos.sum(), w_neg / w_neg.sum()
        cols = {}
        for r in range(1, R + 1):
            rng = np.random.default_rng(1000 + r + 100000 * ci)
            y = df.diag.to_numpy().copy()
            f10 = rng.choice(pos, n10, replace=False, p=w_pos)
            f01 = rng.choice(neg, n01, replace=False, p=w_neg)
            y[f10], y[f01] = 0, 1
            cols[f"null{r:03d}"] = y
            flips += [dict(r=r, cycle=cycle, row=int(i), to=0) for i in f10]
            flips += [dict(r=r, cycle=cycle, row=int(i), to=1) for i in f01]
        frames[cycle] = pd.DataFrame(cols, index=df.index)
        logging.info("%s %s: n10=%d n01=%d (null labels have the same 2x2 table with diag)",
                     outcome, cycle, n10, n01)
    dst.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flips).to_csv(dst / "null_flips.csv", index=False)
    return frames


def summarise(outcome: str, v: pd.DataFrame, R: int, dst) -> pd.DataFrame:
    second = C.SECOND[outcome]
    obs = C.shift_table(v, "diag", second)
    nulls = [n for n in sorted(set(v.label)) if n.startswith("null")][:R]
    nd = pd.concat([C.shift_table(v, "diag", n)[["cycle", "method", "model", "feature", "diff"]]
                    .assign(null=n) for n in nulls])
    nd.to_csv(dst / "null_distribution.csv", index=False)
    g = nd.groupby(["cycle", "method", "model", "feature"])["diff"]
    stats = pd.DataFrame({"null_abs_p95": g.apply(lambda x: np.quantile(np.abs(x), 0.95)),
                          "null_mean": g.mean(), "null_sd": g.std(), "n_null": g.size()}).reset_index()
    m = obs.merge(stats, on=["cycle", "method", "model", "feature"])

    def _null(r):
        return nd[(nd.cycle == r.cycle) & (nd.method == r.method) & (nd.model == r.model)
                  & (nd.feature == r.feature)]["diff"].to_numpy()

    # pre-specified (Spec A.3): |observed| against the |null| distribution
    m["p_emp"] = [(1 + (np.abs(_null(r)) >= abs(r["diff"])).sum()) / (r.n_null + 1) for _, r in m.iterrows()]
    m["exceeds_null_p95"] = m["diff"].abs() > m["null_abs_p95"]
    # post hoc (added after the null was found to be biased, i.e. random discordance
    # itself moves shares systematically): one-sided in the observed direction,
    # observed versus the signed null distribution
    m["p_dir"] = [(1 + ((_null(r) <= r["diff"]) if r["diff"] < 0 else (_null(r) >= r["diff"])).sum())
                  / (r.n_null + 1) for _, r in m.iterrows()]
    m["z_vs_null"] = (m["diff"] - m["null_mean"]) / m["null_sd"]
    m.to_csv(dst / "null_vs_observed.csv", index=False)
    base = C.v1_replicated(outcome)
    ex = []
    for _, r in base.iterrows():
        g2 = m[(m.method == r.method) & (m.model == r.model) & (m.feature == r.feature)].set_index("cycle")
        if not all(c in g2.index for c in C.CYCLES):
            ex.append({})
            continue
        ex.append(dict(exceed_null_1720=bool(g2.loc[C.CYCLES[0], "exceeds_null_p95"]),
                       exceed_null_2123=bool(g2.loc[C.CYCLES[1], "exceeds_null_p95"]),
                       p_emp_1720=g2.loc[C.CYCLES[0], "p_emp"], p_emp_2123=g2.loc[C.CYCLES[1], "p_emp"],
                       p_dir_1720=g2.loc[C.CYCLES[0], "p_dir"], p_dir_2123=g2.loc[C.CYCLES[1], "p_dir"],
                       null_mean_1720=g2.loc[C.CYCLES[0], "null_mean"],
                       null_mean_2123=g2.loc[C.CYCLES[1], "null_mean"],
                       z_1720=g2.loc[C.CYCLES[0], "z_vs_null"], z_2123=g2.loc[C.CYCLES[1], "z_vs_null"],
                       null_p95_1720=g2.loc[C.CYCLES[0], "null_abs_p95"],
                       null_p95_2123=g2.loc[C.CYCLES[1], "null_abs_p95"],
                       diff_obs5_1720=g2.loc[C.CYCLES[0], "diff"], diff_obs5_2123=g2.loc[C.CYCLES[1], "diff"]))
    out = pd.concat([base.reset_index(drop=True), pd.DataFrame(ex)], axis=1)
    out["exceed_null_both"] = (out.exceed_null_1720.fillna(False).astype(bool)
                               & out.exceed_null_2123.fillna(False).astype(bool))
    out["beyond_null_dir_both"] = (out.p_dir_1720 < 0.05) & (out.p_dir_2123 < 0.05)
    out.to_csv(dst / "survival.csv", index=False)
    v = v.assign(kind=np.where(v.label.str.startswith("null"), "null_labels", v.label))
    v.groupby(["kind", "cycle"]).auc.mean().unstack("cycle").to_csv(dst / "auc_by_label_kind.csv")
    logging.info("%s: %d/%d jbi-v1 replicated shifts exceed the null p95 in both cycles (R=%d)",
                 outcome, int(out.exceed_null_both.sum()), len(out), len(nulls))
    return out


# Final grids (a coarse grid 0-6 was run first and showed the real-label AUC between 0 and 0.5).
GAMMAS = {"diabetes": [0.0, 0.15, 0.2, 0.25, 0.3, 0.35],
          "hypertension": [0.0, 0.2, 0.3, 0.35, 0.4, 0.5, 0.7]}
GAMMA_USED = {"diabetes": 0.3, "hypertension": 0.35}


def select_gamma(summ: pd.DataFrame) -> float:
    """gamma* = argmin_gamma L(gamma), L(gamma) = mean over the four
    (cycle, model) cells of |mean null AUC(gamma) - mean real-label AUC|,
    means over seeds 0-4 and the R calibration labels; ties go to the smaller
    gamma (none occurred)."""
    real = summ.loc["real"].iloc[0]
    loss = (summ.loc["null"] - real).abs().mean(axis=1)
    return float(loss[loss == loss.min()].index.min())


def calibrate(outcome: str, R: int = 5) -> pd.DataFrame:
    """Mean test AUC (seeds 0-4) of the real second label and of null labels
    for each gamma, for logistic regression and XGBoost (fast models). The
    risk-structured null uses the gamma whose AUC is closest to the real label."""
    from src.explainability.global_attribution import build_model
    from sklearn.metrics import roc_auc_score
    rows = []
    tmp = C.OUT_DIR / "label_null" / f"_calibrate_{outcome}"
    grid = GAMMAS[outcome] if isinstance(GAMMAS, dict) else GAMMAS
    for gamma in grid:
        frames = null_frames(outcome, R, tmp, gamma=gamma)
        for cycle in C.CYCLES:
            df = pd.concat([C.load_base(outcome, cycle), frames[cycle]], axis=1)
            X = df[C.FEATURES[outcome]]
            for model in ["logistic_regression", "xgboost"]:
                for seed in SEEDS:
                    tr, te = C.split(df, seed)
                    labs = [C.SECOND[outcome]] if gamma == grid[0] else []
                    for lab in labs + list(frames[cycle].columns):
                        m = build_model(model, seed).fit(X.loc[tr], df.loc[tr, lab])
                        auc = roc_auc_score(df.loc[te, lab], m.predict_proba(X.loc[te])[:, 1])
                        rows.append(dict(gamma=gamma, cycle=cycle, model=model, seed=seed,
                                         kind="real" if lab == C.SECOND[outcome] else "null", auc=auc))
    out = pd.DataFrame(rows)
    summ = out.groupby(["kind", "gamma", "cycle", "model"]).auc.mean().unstack(["cycle", "model"])
    dst = C.OUT_DIR / "label_null"
    summ.to_csv(dst / f"calibration_{outcome}.csv")
    logging.info("AUC by gamma (null) and real label:\n%s\nselected gamma = %s",
                 summ.round(4).to_string(), select_gamma(summ))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return summ


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outcome", choices=["diabetes", "hypertension"], default="diabetes")
    p.add_argument("--R", type=int, default=50)
    p.add_argument("--stage", choices=["calibrate", "fit", "summary", "all"], default="all")
    p.add_argument("--gamma", type=float, default=0.0,
                   help="0 = random discordance; > 0 = risk-structured (outputs/label_null/<outcome>_risk)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    if a.stage == "calibrate":
        calibrate(a.outcome)
        return
    if a.gamma == 0:
        sub = a.outcome
    elif abs(a.gamma - GAMMA_USED[a.outcome]) < 1e-9:
        sub = f"{a.outcome}_risk"
    else:  # gamma sensitivity runs
        sub = f"{a.outcome}_risk_g{a.gamma:g}"
    dst = C.OUT_DIR / "label_null" / sub
    frames = null_frames(a.outcome, a.R, dst, gamma=a.gamma)
    if a.gamma > 0:
        (dst / "gamma.txt").write_text(f"{a.gamma}\n")
    labels = ["diag", C.SECOND[a.outcome]] + [f"null{r:03d}" for r in range(1, a.R + 1)]
    if a.stage in ("fit", "all"):
        # loop null labels in the outer position so that a partial run is usable
        C.fit_labels(a.outcome, labels[:2], SEEDS, dst / "vectors.csv",
                     shap_n_rf=SHAP_N_RF, label_frames=frames)
        for lab in labels[2:]:
            C.fit_labels(a.outcome, [lab], SEEDS, dst / "vectors.csv",
                         shap_n_rf=SHAP_N_RF, label_frames=frames)
    if a.stage in ("summary", "all"):
        summarise(a.outcome, pd.read_csv(dst / "vectors.csv"), a.R, dst)


if __name__ == "__main__":
    main()
