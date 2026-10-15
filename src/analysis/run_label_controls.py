"""
Are the replicated outcome-definition shifts explained by prevalence, class
weighting or the particular laboratory definition? (v32, Spec A.1, A.2, A.4)

Stages
    variants   fit alternative labels on the jbi-v1 people and splits
               diabetes:     lab63 (HbA1c >= 6.3), labM (prevalence-matched to
                             diag), labG (HbA1c >= 6.5 or FPG >= 126, as stored
                             in Diabetes_lab), labTx (HbA1c >= 6.5 or treated)
               hypertension: meas130 (>= 130/80), measM (prevalence-matched)
               diag vectors are the stored jbi-v1 vectors (reproduced exactly)
    cw         refit diag and the second label with class weighting
               'balanced' (all models) and 'none'
    summary    survival of every jbi-v1 replicated shift under each control

Outputs: outputs/label_controls/
Usage:
    python -m src.analysis.run_label_controls --stage all
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.analysis import label_controls_core as C

DST = C.OUT_DIR / "label_controls"
VARIANTS = {"diabetes": ["lab63", "labM", "labG", "labTx"], "hypertension": ["meas130", "measM"]}
SEEDS = {"diabetes": list(range(10)), "hypertension": list(range(5))}


def stage_variants(outcome: str) -> None:
    C.fit_labels(outcome, VARIANTS[outcome], SEEDS[outcome], DST / f"vectors_{outcome}_variants.csv")


def stage_cw(outcome: str) -> None:
    for cw in ["balanced", "none"]:
        if outcome == "hypertension" and cw == "none":
            continue  # already in outputs/label_axis_hypertension_none (jbi-v1)
        C.fit_labels(outcome, ["diag", C.SECOND[outcome]], SEEDS[outcome],
                     DST / f"vectors_{outcome}_cw_{cw}.csv", class_weighting=cw)


def stage_summary(outcome: str) -> pd.DataFrame:
    base = C.v1_replicated(outcome)
    diag = C.v1_vectors(outcome, labels=("diag",), seeds=SEEDS[outcome])
    out = base
    shifts = []
    var_csv = DST / f"vectors_{outcome}_variants.csv"
    if var_csv.exists():
        vv = pd.concat([diag, pd.read_csv(var_csv)])
        for lab in VARIANTS[outcome]:
            if lab in set(vv.label):
                s = C.shift_table(vv, "diag", lab)
                shifts.append(s.assign(control=lab))
                out = C.survival(out, s, lab)
    for cw in ["balanced", "none"]:
        f = DST / f"vectors_{outcome}_cw_{cw}.csv"
        if outcome == "hypertension" and cw == "none":
            v = pd.read_csv(C.OUT_DIR / "label_axis_hypertension_none" / "vectors.csv")
            perf = pd.read_csv(C.OUT_DIR / "label_axis_hypertension_none" / "performance.csv")
            v = v[v.weighting == "unweighted"].merge(
                perf[["cycle", "label", "model", "seed", "n_train", "n_test"]],
                on=["cycle", "label", "model", "seed"])
        elif f.exists():
            v = pd.read_csv(f)
        else:
            continue
        s = C.shift_table(v, "diag", C.SECOND[outcome])
        shifts.append(s.assign(control=f"cw_{cw}"))
        out = C.survival(out, s, f"cw_{cw}")
    DST.mkdir(parents=True, exist_ok=True)
    if shifts:
        pd.concat(shifts).to_csv(DST / f"feature_shift_{outcome}_controls.csv", index=False)
    out.to_csv(DST / f"survival_{outcome}.csv", index=False)
    surv = [c for c in out.columns if c.startswith("survive_")]
    logging.info("%s: %d jbi-v1 replicated shifts; surviving per control:\n%s", outcome, len(out),
                 out[surv].sum().to_string())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["variants", "cw", "summary", "all"], default="all")
    p.add_argument("--outcome", choices=["diabetes", "hypertension", "both"], default="both")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    outcomes = ["diabetes", "hypertension"] if a.outcome == "both" else [a.outcome]
    for o in outcomes:
        if a.stage in ("variants", "all"):
            stage_variants(o)
    for o in outcomes:
        if a.stage in ("cw", "all"):
            stage_cw(o)
    for o in outcomes:
        if a.stage in ("summary", "all"):
            stage_summary(o)


if __name__ == "__main__":
    main()
