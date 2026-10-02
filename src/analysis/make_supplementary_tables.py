"""
LaTeX fragments for the data-driven supplementary tables of the journal manuscript.
Values are truncated to four decimals (percentages to two), as in the manuscript.

Writes outputs/journal_tables/{seed_reference,variance_sensitivity,label_shift_tests,fairness,income}.tex
Usage:
    python -m src.analysis.make_supplementary_tables
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.manuscript_numbers import pct, t4

OUT = Path(__file__).resolve().parents[2] / "outputs"
DST = OUT / "journal_tables"
METRIC = {"J_A": r"$J_A$", "spearman": r"Spearman", "rbo_ext_p90": r"$\mathrm{RBO}_{\mathrm{ext}}$"}
MODEL = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGBoost"}
DESIGN = {"brfss_pipeline": "BRFSS, conference weighting", "brfss_none": "BRFSS, no weighting",
          "brfss_balanced": "BRFSS, all weighted", "nhanes_2017-2020": "NHANES 2017--2020",
          "nhanes_2021-2023": "NHANES 2021--2023"}


def seed_reference() -> str:
    d = pd.read_csv(OUT / "sensitivity" / "seed_floor_percentiles.csv")
    d = d[d.weighting == "unweighted"]
    rows = []
    for (cyc, met), g in d.groupby(["cycle", "metric"]):
        cells = []
        for q in [0.01, 0.05, 0.10]:
            x = g[g.percentile == q].share_below
            cells.append(f"{pct(x.mean())} ({pct(x.max())})")
        rows.append(f"{cyc.replace('-', '--')} & {METRIC[met]} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def variance_sensitivity() -> str:
    v = pd.read_csv(OUT / "sensitivity" / "variance_specs.csv")
    pf = pd.read_csv(OUT / "sensitivity" / "variance_per_feature.csv")
    pm = pd.read_csv(OUT / "sensitivity" / "variance_permutation.csv")
    rows = []
    for des in DESIGN:
        third = "label" if des.startswith("nhanes") else "year"
        for resp in ["share", "rank", "clr"]:
            rep = v[(v.design == des) & (v.response == resp) & (v.model_spec == "seed as replicate")].set_index("source").eta2
            cro = v[(v.design == des) & (v.response == resp) & (v.model_spec == "seed crossed")].set_index("source").eta2
            med = pf[(pf.design == des) & (pf.response == resp)].set_index("source")["median"]
            perm = pm[(pm.design == des) & (pm.response == resp)].set_index("factor").p_perm
            cells = [t4(rep[k]) for k in ["model", "method", third]]
            cells += [t4(med[k]) for k in ["model", "method", third]]
            cells += [t4(cro["seed"]), t4(cro["seed x factors"])]
            cells += [t4(perm.max()) if len(perm) else "--"]
            rows.append(f"{DESIGN[des]} & {resp.upper() if resp == 'clr' else resp} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def label_shift_tests() -> str:
    s = pd.read_csv(OUT / "sensitivity" / "label_shift_signflip.csv")
    s = s[s.weighting == "unweighted"]
    rep = pd.read_csv(OUT / "sensitivity" / "label_shift_replicated_q_bh.csv")
    rows = []
    for _, r in rep.sort_values(["method", "feature", "model"]).iterrows():
        g = s[(s.method == r.method) & (s.model == r.model) & (s.feature == r.feature)].set_index("cycle")
        cells = []
        for cyc in ["2017-2020", "2021-2023"]:
            x = g.loc[cyc]
            cells += [pct(x["diff"]), t4(x.q_bh), t4(x.q_signflip)]
        rows.append(f"{r.method} & {MODEL[r.model]} & {r.feature} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def fairness() -> str:
    g = pd.read_csv(OUT / "label_fairness" / "gaps_summary.csv")
    g = g[g.truth == "lab"]
    rows = []
    for att, label in [("Education", "Education (high school or less vs.\\ more)"), ("Income", "Income (PIR $<2$ vs.\\ $\\ge 2$)")]:
        for cyc in ["2017-2020", "2021-2023"]:
            x = g[(g.cycle == cyc) & (g.attribute == att)].groupby("train_label").signed_gap_weighted_mean.mean()
            rows.append(f"{cyc.replace('-', '--')} & {label} & {t4(x['diag'])} & {t4(x['lab'])} & {t4(x['total'])} \\\\")
    d = pd.read_csv(OUT / "label_fairness" / "gaps.csv")
    d = d[d.truth == "lab"]
    piv = d.pivot_table(index=["cycle", "seed", "model", "attribute"], columns="train_label", values="tpr_gap_weighted")
    piv["diff"] = piv["diag"] - piv["lab"]
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{@{}l}{Diagnosis-trained minus HbA1c-trained gap: mean [2.5th, 97.5th percentile over 3 models $\times$ 10 seeds]} \\")
    for (cyc, att), x in piv.groupby(["cycle", "attribute"])["diff"]:
        rows.append(f"{cyc.replace('-', '--')} & {att} & \\multicolumn{{3}}{{l}}{{{t4(x.mean())} [{t4(x.quantile(0.025))}, {t4(x.quantile(0.975))}]}} \\\\")
    return "\n".join(rows)


def income() -> str:
    d = pd.read_csv(OUT / "brfss_income_harmonised" / "delta_tpr.csv")
    rows = []
    for mdl in ["xgboost", "random_forest", "logistic_regression"]:
        g = d[d.model == mdl].set_index("year")
        nat = " / ".join(t4(g.loc[y, "delta_tpr_native"]) for y in [2015, 2021, 2023])
        har = " / ".join(t4(g.loc[y, "delta_tpr_harmonised"]) for y in [2015, 2021, 2023])
        rows.append(f"{MODEL[mdl]} & {nat} & {har} \\\\")
    return "\n".join(rows)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name, fn in [("seed_reference", seed_reference), ("variance_sensitivity", variance_sensitivity),
                     ("label_shift_tests", label_shift_tests), ("fairness", fairness), ("income", income)]:
        (DST / f"{name}.tex").write_text(fn() + "\n")
    print("Tables written to", DST)


if __name__ == "__main__":
    main()
