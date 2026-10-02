"""
LaTeX fragments for the data-driven supplementary tables of the journal manuscript.
Values are truncated to four decimals (percentages to two), as in the manuscript.

Writes outputs/journal_tables/{seed_reference,variance_sensitivity,label_shift_tests,fairness,income,hypertension_shifts,dependence,soa_comparison}.tex
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
    nb = pd.read_csv(OUT / "label_axis" / "feature_shift.csv")
    nb = nb[(nb.weighting == "unweighted") & (nb.label_1 == "diag") & (nb.label_2 == "lab")].copy()
    nb["bonf"] = nb.groupby(["cycle", "method"]).p_corrected.transform(lambda p: (p * p.notna().sum()).clip(upper=1))
    bonf_ok = nb.groupby(["method", "model", "feature"]).bonf.max() < 0.05
    rows = []
    for _, r in rep.sort_values(["method", "feature", "model"]).iterrows():
        g = s[(s.method == r.method) & (s.model == r.model) & (s.feature == r.feature)].set_index("cycle")
        cells = []
        for cyc in ["2017-2020", "2021-2023"]:
            x = g.loc[cyc]
            cells += [pct(x["diff"]), t4(x.q_bh), t4(x.q_signflip)]
        mark = r"$^{\dagger}$" if bonf_ok[(r.method, r.model, r.feature)] else ""
        rows.append(f"{r.method} & {MODEL[r.model]} & {r.feature}{mark} & " + " & ".join(cells) + r" \\")
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


def hypertension_shifts() -> str:
    path = OUT / "label_axis_hypertension_pipeline" / "feature_shift.csv"
    if not path.exists():
        return ""
    s = pd.read_csv(path)
    s = s[(s.weighting == "unweighted") & (s.label_1 == "diag") & (s.label_2 == "measured")]
    w = s.pivot_table(index=["method", "model", "feature"], columns="cycle", values=["diff", "q_bh"])
    same = (w["diff"]["2017-2020"] > 0) == (w["diff"]["2021-2023"] > 0)
    rep = (w["q_bh"] < 0.05).all(axis=1) & same
    expected = w.index.get_level_values("feature").isin(["Diabetes_self", "HeartDiseaseorAttack", "HighChol"]) & \
        (w.index.get_level_values("method") == "SHAP")
    keep = w[rep | expected].sort_values([("diff", "2021-2023")], ascending=False)
    rows = []
    for (meth, mdl, feat), r in keep.iterrows():
        mark = r"$^{\ast}$" if rep[(meth, mdl, feat)] else ""
        cells = []
        for cyc in ["2017-2020", "2021-2023"]:
            cells += [pct(r[("diff", cyc)]), t4(r[("q_bh", cyc)])]
        feat_tex = feat.replace("_", r"\_")
        rows.append(f"{meth} & {MODEL[mdl]} & {feat_tex}{mark} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def dependence() -> str:
    d = OUT / "dependence"
    if not (d / "domain_shift_replicated.csv").exists():
        return ""
    rows = [r"\multicolumn{6}{@{}l}{\textit{(a) Predictor pairs with $|\rho|\ge 0.4$ (Spearman)}} \\"]
    c = pd.read_csv(d / "predictor_correlation_strong.csv")
    for _, r in c[c.rho.abs() >= 0.4].iterrows():
        rows.append(f"{r.cohort.replace('-', '--')} & \\multicolumn{{3}}{{l}}{{{r.feature_1.replace('_', chr(92) + '_')} -- {r.feature_2.replace('_', chr(92) + '_')}}} & \\multicolumn{{2}}{{l}}{{{t4(r.rho)}}} \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{@{}l}{\textit{(b) Domain-level diag-minus-lab share shifts replicated in both cycles (percentage points)}} \\")
    rows.append(r"Domain & Method & Model & 2017--2020 & 2021--2023 & \\")
    rep = pd.read_csv(d / "domain_shift_replicated.csv")
    for _, r in rep.sort_values(["domain", "method", "model"]).iterrows():
        rows.append(f"{r.domain} & {r.method} & {MODEL[r.model]} & {pct(r['2017-2020'])} & {pct(r['2021-2023'])} & \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{@{}l}{\textit{(c) RBO persistence $p$: NHANES label pairs below the seed 5th percentile (\%); BRFSS mean RBO}} \\")
    rows.append(r" & & & $p=0.8$ & $p=0.9$ & $p=0.95$ \\")
    rb = pd.read_csv(d / "rbo_persistence_summary.csv")
    for _, r in rb.iterrows():
        f = pct if r.analysis.startswith("NHANES") else t4
        name = {"NHANES label pairs below seed p5": "NHANES label pairs", "BRFSS mean RBO, method axis": "BRFSS SHAP vs.\\ PI",
                "BRFSS mean RBO, model axis": "BRFSS cross-model"}[r.analysis]
        rows.append(f"{name} & {str(r.cycle).replace('-', '--')} & & {f(r['0.8'])} & {f(r['0.9'])} & {f(r['0.95'])} \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{@{}l}{\textit{(d) Replicated PI label shifts by handling of negative PI values (percentage points)}} \\")
    rows.append(r"Handling & Model & Feature & 2017--2020 & 2021--2023 & \\")
    pi = pd.read_csv(d / "pi_negative_handling.csv")
    for h in ["share, negative PI clipped at 0", "share, negative PI as absolute value", "rank of raw PI"]:
        g = pi[pi.handling == h]
        if len(g):
            for _, r in g.iterrows():
                rows.append(f"{h} & {MODEL[r.model]} & {r.feature} & {pct(r['2017-2020'])} & {pct(r['2021-2023'])} & \\\\")
        else:
            rows.append(f"{h} & \\multicolumn{{5}}{{l}}{{none replicated}} \\\\")
    return "\n".join(rows)


def soa_comparison() -> str:
    path = OUT / "soa_comparison" / "summary.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    name = {"feature_agreement": "Feature agreement@5", "rank_agreement": "Rank agreement@5",
            "rank_correlation": "Rank correlation (Spearman)", "pairwise_rank_agreement": "Pairwise rank agreement",
            "J_A": r"$J_A$ (conference metric)", "contradiction_A (triad)": r"Contradiction flag (triad)"}
    rows = [r"\multicolumn{4}{@{}l}{\textit{(a) Conference vectors, 45 pairs: mean and number of pairs below 1 (flag: pairs flagged)}} \\"]
    for _, r in d[d.case.str.startswith("A.")].iterrows():
        rows.append(f"{name[r.quantity]} & {t4(r.value)} & {int(r.pairs_below_1)} & \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{4}{@{}l}{\textit{(b) BRFSS cross-model SHAP pairs (45 with conference weighting, 27 without): mean}} \\")
    rows.append(r" & Conference weighting & No weighting & \\")
    b = d[d.case.str.startswith("B.")]
    for q in ["feature_agreement", "rank_agreement", "rank_correlation", "pairwise_rank_agreement", "J_A"]:
        x = b[b.quantity == q].set_index("case").value
        rows.append(f"{name[q]} & {t4(x['B. BRFSS cross-model SHAP, weighting pipeline'])} & {t4(x['B. BRFSS cross-model SHAP, weighting none'])} & \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{4}{@{}l}{\textit{(c) NHANES label pairs: mean (seed-pair mean) and share of label pairs below the seed 5th percentile}} \\")
    rows.append(r" & 2017--2020 & 2021--2023 & \\")
    c = d[d.case.str.startswith("C.")]
    for q in ["feature_agreement", "rank_agreement", "rank_correlation", "pairwise_rank_agreement", "J_A"]:
        cells = []
        for cyc in ["2017-2020", "2021-2023"]:
            r = c[(c.quantity == q) & (c.case.str.endswith(cyc))].iloc[0]
            cells.append(f"{t4(r.value)} ({t4(r.seed_value)}); {pct(r.label_below_seed_p5)}\\%")
        rows.append(f"{name[q]} & {cells[0]} & {cells[1]} & \\\\")
    return "\n".join(rows)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name, fn in [("seed_reference", seed_reference), ("variance_sensitivity", variance_sensitivity),
                     ("label_shift_tests", label_shift_tests), ("fairness", fairness), ("income", income), ("hypertension_shifts", hypertension_shifts), ("dependence", dependence), ("soa_comparison", soa_comparison)]:
        (DST / f"{name}.tex").write_text(fn() + "\n")
    print("Tables written to", DST)


if __name__ == "__main__":
    main()
