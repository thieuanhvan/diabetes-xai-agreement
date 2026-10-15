"""
Validation of the outcome-definition shift procedure (v32).

(1) False positives, no refit. Each AUC-matched null label in turn plays the
    role of the measured label: its shifts against the diagnosis are tested
    (a) against zero with the replication criterion only, as a conventional
    analysis would, and (b) with the full procedure, i.e. replicated and
    beyond the remaining R-1 null labels (one-sided p < 0.05 in both cycles).
    A null label carries no outcome information beyond the specified
    disagreement mechanism, so a shift flagged as exceeding the null is a false
    flag under that mechanism (a nonzero shift itself can be real).

(2) Positive control, refit. Planted labels have the same 2x2 table with the
    diagnosis as the real measured label and the same risk-structured
    disagreement as the AUC-matched null, plus a tilt on one predictor f:
    positives are flipped with weight exp(-gamma*z - beta*x_f) and negatives
    with weight exp(gamma*z + beta*x_f), x_f the standardised predictor, so
    people with high f become positive. The procedure should flag a shift
    toward f under the planted label.

Outputs: outputs/null_validation/<outcome>/
Usage:
    python -m src.analysis.run_null_validation --stage all
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from src.analysis import label_controls_core as C
from src.analysis.run_label_null import GAMMA_USED, SEEDS, SHAP_N_RF, diag_risk_logit

DST = C.OUT_DIR / "null_validation"
PLANT = {"diabetes": ["BMI", "Age"], "hypertension": ["BMI", "Age"]}
BETA = 1.0
R_PLANT = 3


def _null_vectors(outcome: str) -> pd.DataFrame:
    return pd.read_csv(C.OUT_DIR / "label_null" / f"{outcome}_risk" / "vectors.csv")


def _flag(obs: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """obs: shift_table of one label vs diag; ref: diffs of reference null labels
    (columns cycle, method, model, feature, diff, null). Adds replicated,
    p_dir per cycle and the full-procedure flag."""
    rep = C.replicated(obs)
    rows = []
    for r in rep.itertuples():
        ok, ps = True, {}
        for cyc in C.CYCLES:
            o = obs[(obs.cycle == cyc) & (obs.method == r.method) & (obs.model == r.model)
                    & (obs.feature == r.feature)]["diff"].iloc[0]
            x = ref[(ref.cycle == cyc) & (ref.method == r.method) & (ref.model == r.model)
                    & (ref.feature == r.feature)]["diff"].to_numpy()
            p = (1 + ((x <= o) if o < 0 else (x >= o)).sum()) / (len(x) + 1)
            ps[cyc] = p
            ok &= p < 0.05
        rows.append(dict(method=r.method, model=r.model, feature=r.feature, sign=r.sign,
                         p_dir_1720=ps[C.CYCLES[0]], p_dir_2123=ps[C.CYCLES[1]], beyond=ok))
    return pd.DataFrame(rows, columns=["method", "model", "feature", "sign", "p_dir_1720",
                                       "p_dir_2123", "beyond"])


def false_positives(outcome: str) -> pd.DataFrame:
    v = _null_vectors(outcome)
    nulls = sorted(n for n in set(v.label) if n.startswith("null"))
    shifts = {n: C.shift_table(v, "diag", n) for n in nulls}
    rows, detail = [], []
    for n in nulls:
        ref = pd.concat([shifts[m][["cycle", "method", "model", "feature", "diff"]].assign(null=m)
                         for m in nulls if m != n])
        f = _flag(shifts[n], ref)
        rows.append(dict(null=n, replicated_vs_zero=len(f), beyond_other_nulls=int(f.beyond.sum())))
        detail.append(f.assign(null=n))
    out = pd.DataFrame(rows)
    d = DST / outcome
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "false_positives.csv", index=False)
    pd.concat(detail).to_csv(d / "false_positives_detail.csv", index=False)
    logging.info("%s: per null label, replicated shifts vs zero: mean %.2f (max %d); "
                 "full procedure: mean %.3f, labels with >=1 flag %d/%d", outcome,
                 out.replicated_vs_zero.mean(), out.replicated_vs_zero.max(),
                 out.beyond_other_nulls.mean(), int((out.beyond_other_nulls > 0).sum()), len(out))
    return out


def planted_frames(outcome: str) -> dict:
    gamma = GAMMA_USED[outcome]
    frames = {}
    for ci, cycle in enumerate(C.CYCLES):
        df = C.load_base(outcome, cycle)
        s = C.SECOND[outcome]
        n10 = int(((df.diag == 1) & (df[s] == 0)).sum())
        n01 = int(((df.diag == 0) & (df[s] == 1)).sum())
        pos, neg = np.flatnonzero(df.diag == 1), np.flatnonzero(df.diag == 0)
        z = diag_risk_logit(df, outcome)
        cols = {}
        for f in PLANT[outcome]:
            x = ((df[f] - df[f].mean()) / df[f].std()).to_numpy()
            lp, ln = -gamma * z[pos] - BETA * x[pos], gamma * z[neg] + BETA * x[neg]
            wp, wn = np.exp(lp - lp.max()), np.exp(ln - ln.max())
            wp, wn = wp / wp.sum(), wn / wn.sum()
            for r in range(1, R_PLANT + 1):
                rng = np.random.default_rng(5000 + r + 100 * PLANT[outcome].index(f) + 100000 * ci)
                y = df.diag.to_numpy().copy()
                y[rng.choice(pos, n10, replace=False, p=wp)] = 0
                y[rng.choice(neg, n01, replace=False, p=wn)] = 1
                cols[f"plant_{f}_{r}"] = y
        frames[cycle] = pd.DataFrame(cols, index=df.index)
    return frames


def positive_control(outcome: str, fit: bool) -> pd.DataFrame:
    d = DST / outcome
    d.mkdir(parents=True, exist_ok=True)
    frames = planted_frames(outcome)
    labels = list(frames[C.CYCLES[0]].columns)
    if fit:
        for lab in labels:
            C.fit_labels(outcome, [lab], SEEDS, d / "planted_vectors.csv",
                         shap_n_rf=SHAP_N_RF, label_frames=frames)
    pv = pd.read_csv(d / "planted_vectors.csv")
    v = pd.concat([_null_vectors(outcome), pv])
    nulls = sorted(n for n in set(v.label) if n.startswith("null"))
    ref = pd.concat([C.shift_table(v, "diag", n)[["cycle", "method", "model", "feature", "diff"]].assign(null=n)
                     for n in nulls])
    rows, detail = [], []
    for lab in labels:
        if lab not in set(pv.label):
            continue
        f = _flag(C.shift_table(v, "diag", lab), ref)
        target = lab.split("_")[1]
        hit = f[(f.feature == target) & (f.sign < 0) & f.beyond]
        rows.append(dict(label=lab, target=target,
                         auc=pv[pv.label == lab].groupby("cycle").auc.mean().round(4).to_dict(),
                         target_combos_flagged=len(hit),
                         other_flagged=int(f[(f.feature != target) & f.beyond].shape[0]),
                         other_features=", ".join(sorted(set(f[(f.feature != target) & f.beyond].feature)))))
        detail.append(f.assign(label=lab))
    out = pd.DataFrame(rows)
    out.to_csv(d / "positive_control.csv", index=False)
    if detail:
        pd.concat(detail).to_csv(d / "positive_control_detail.csv", index=False)
    logging.info("%s positive control:\n%s", outcome, out.to_string(index=False))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["fp", "plant", "plant-summary", "all"], default="all")
    p.add_argument("--outcome", choices=["diabetes", "hypertension", "both"], default="both")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    outs = ["diabetes", "hypertension"] if a.outcome == "both" else [a.outcome]
    for o in outs:
        if a.stage in ("fp", "all"):
            false_positives(o)
    for o in outs:
        if a.stage in ("plant", "all"):
            positive_control(o, fit=True)
        elif a.stage == "plant-summary":
            positive_control(o, fit=False)


if __name__ == "__main__":
    main()
