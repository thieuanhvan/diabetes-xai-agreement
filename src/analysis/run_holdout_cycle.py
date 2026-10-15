"""
Held-out replication in NHANES 2015-2016 (v32).

The replicated and core shifts were defined on 2017-2020 and 2021-2023. Here
they are re-tested, without re-selection, on an independent cycle:

    grid       diagnosis and measured labels, 10 seeds (diabetes) / 5 (hypertension),
               full-test-split SHAP, same models and settings as the main analysis
    shift      corrected resampled t-test, BH within the cycle x method family
    calibrate  gamma for the AUC-matched null, chosen on this cycle alone
    null       30 AUC-matched null labels, seeds 0-4, RF SHAP on 500 rows
    summary    per pre-specified shift: same sign and q < 0.05 ("held-out
               replicated"), and beyond the null (one-sided p < 0.05)

Hypertension in 2015-2016 is measured by auscultation (BPX), not oscillometry.

Outputs: outputs/holdout_2015_2016/<outcome>/
Usage:
    python -m src.analysis.run_holdout_cycle --outcome diabetes --stage all
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from src.analysis import label_controls_core as C
from src.analysis import run_label_null as NL

CYCLE = "2015-2016"
DST = C.OUT_DIR / "holdout_2015_2016"
SEEDS_GRID = {"diabetes": list(range(10)), "hypertension": list(range(5))}
GRID = {"diabetes": [0.0, 0.2, 0.25, 0.3, 0.35, 0.4], "hypertension": [0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]}
R = 30


def frames(outcome: str, gamma: float, R: int) -> dict:
    df = C.load_base(outcome, CYCLE)
    s = C.SECOND[outcome]
    n10 = int(((df.diag == 1) & (df[s] == 0)).sum())
    n01 = int(((df.diag == 0) & (df[s] == 1)).sum())
    pos, neg = np.flatnonzero(df.diag == 1), np.flatnonzero(df.diag == 0)
    wp = wn = None
    if gamma > 0:
        z = NL.diag_risk_logit(df, outcome)
        wp = np.exp(-gamma * (z[pos] - z[pos].max())); wp /= wp.sum()
        wn = np.exp(gamma * (z[neg] - z[neg].max())); wn /= wn.sum()
    cols = {}
    for r in range(1, R + 1):
        rng = np.random.default_rng(7000 + r)
        y = df.diag.to_numpy().copy()
        y[rng.choice(pos, n10, replace=False, p=wp)] = 0
        y[rng.choice(neg, n01, replace=False, p=wn)] = 1
        cols[f"null{r:03d}"] = y
    logging.info("%s %s: n10=%d n01=%d gamma=%s", outcome, CYCLE, n10, n01, gamma)
    return {CYCLE: pd.DataFrame(cols, index=df.index)}


def calibrate(outcome: str) -> float:
    from sklearn.metrics import roc_auc_score
    from src.explainability.global_attribution import build_model
    df = C.load_base(outcome, CYCLE)
    X = df[C.FEATURES[outcome]]
    rows = []
    for g in GRID[outcome]:
        fr = frames(outcome, g, 5)[CYCLE]
        d = pd.concat([df, fr], axis=1)
        for model in ["logistic_regression", "xgboost"]:
            for seed in range(5):
                tr, te = C.split(d, seed)
                labs = list(fr.columns) + ([C.SECOND[outcome]] if g == GRID[outcome][0] else [])
                for lab in labs:
                    m = build_model(model, seed).fit(X.loc[tr], d.loc[tr, lab])
                    rows.append(dict(gamma=g, model=model, kind="real" if lab == C.SECOND[outcome] else "null",
                                     auc=roc_auc_score(d.loc[te, lab], m.predict_proba(X.loc[te])[:, 1])))
    out = pd.DataFrame(rows)
    summ = out.groupby(["kind", "gamma", "model"]).auc.mean().unstack("model")
    (DST / outcome).mkdir(parents=True, exist_ok=True)
    summ.to_csv(DST / outcome / "calibration.csv")
    real = summ.loc["real"].iloc[0]
    nulls = summ.loc["null"]
    gamma = float((nulls - real).abs().mean(axis=1).idxmin())
    (DST / outcome / "gamma.txt").write_text(f"{gamma}\n")
    logging.info("%s calibration:\n%s\nselected gamma = %s", outcome, summ.round(4).to_string(), gamma)
    return gamma


def run(outcome: str, stages: list[str]) -> None:
    d = DST / outcome
    d.mkdir(parents=True, exist_ok=True)
    s = C.SECOND[outcome]
    if "grid" in stages:
        C.fit_labels(outcome, ["diag", s], SEEDS_GRID[outcome], d / "vectors_grid.csv", cycles=[CYCLE])
    if "calibrate" in stages:
        calibrate(outcome)
    if "null" in stages:
        gamma = float((d / "gamma.txt").read_text())
        fr = frames(outcome, gamma, R)
        C.fit_labels(outcome, ["diag", s], NL.SEEDS, d / "vectors_null.csv", shap_n_rf=NL.SHAP_N_RF,
                     label_frames=fr, cycles=[CYCLE])
        for r in range(1, R + 1):
            C.fit_labels(outcome, [f"null{r:03d}"], NL.SEEDS, d / "vectors_null.csv", shap_n_rf=NL.SHAP_N_RF,
                         label_frames=fr, cycles=[CYCLE])
    if "summary" in stages:
        summary(outcome)


def summary(outcome: str) -> pd.DataFrame:
    d = DST / outcome
    s = C.SECOND[outcome]
    vg = pd.read_csv(d / "vectors_grid.csv")
    sh = C.shift_table(vg, "diag", s)
    sh.to_csv(d / "feature_shift.csv", index=False)
    m = pd.read_csv(C.OUT_DIR / "label_null" / f"master_{outcome}.csv")
    rows = []
    vn = pd.read_csv(d / "vectors_null.csv") if (d / "vectors_null.csv").exists() else None
    if vn is not None:
        obs = C.shift_table(vn, "diag", s)
        nulls = sorted(x for x in set(vn.label) if x.startswith("null"))
        nd = pd.concat([C.shift_table(vn, "diag", n)[["method", "model", "feature", "diff"]].assign(null=n)
                        for n in nulls])
    for r in m.itertuples():
        g = sh[(sh.method == r.method) & (sh.model == r.model) & (sh.feature == r.feature)].iloc[0]
        rep = bool(np.sign(g["diff"]) == r.sign and g.q_bh < 0.05)
        row = dict(method=r.method, model=r.model, feature=r.feature, sign=r.sign, core=bool(r.core),
                   diff=g["diff"], q=g.q_bh, same_sign=bool(np.sign(g["diff"]) == r.sign), replicated=rep)
        if vn is not None:
            o = obs[(obs.method == r.method) & (obs.model == r.model) & (obs.feature == r.feature)]["diff"].iloc[0]
            x = nd[(nd.method == r.method) & (nd.model == r.model) & (nd.feature == r.feature)]["diff"].to_numpy()
            p = (1 + ((x <= o) if o < 0 else (x >= o)).sum()) / (len(x) + 1)
            row.update(diff_obs5=o, null_mean=x.mean(), p_dir=p,
                       beyond=bool(p < 0.05 and np.sign(o) == r.sign))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(d / "holdout_summary.csv", index=False)
    if vn is not None:
        a = vn.drop_duplicates(["label", "model", "seed"])
        a = a.assign(kind=np.where(a.label.str.startswith("null"), "null_labels", a.label))
        a.groupby(["kind", "model"]).auc.mean().unstack("model").to_csv(d / "auc_by_label_kind.csv")
    for flag in ["core", None]:
        sub = out[out.core] if flag else out
        msg = f"same sign {int(sub.same_sign.sum())}/{len(sub)}, replicated {int(sub.replicated.sum())}/{len(sub)}"
        if "beyond" in sub:
            msg += f", replicated and beyond null {int((sub.replicated & sub.beyond).sum())}/{len(sub)}"
        logging.info("%s %s shifts: %s", outcome, "core" if flag else "all replicated", msg)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outcome", choices=["diabetes", "hypertension"], default="diabetes")
    p.add_argument("--stage", default="all", help="comma list of grid,calibrate,null,summary or all")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    stages = ["grid", "calibrate", "null", "summary"] if a.stage == "all" else a.stage.split(",")
    run(a.outcome, stages)


if __name__ == "__main__":
    main()
