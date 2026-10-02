"""
Every number quoted in the journal manuscript, recomputed from the stored outputs.

Values are truncated (not rounded) to four decimals, percentages to two, so that
each printed number is a prefix of the stored value.

Writes outputs/journal_numbers/numbers.json and numbers.md
Usage:
    python -m src.analysis.manuscript_numbers
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.run_label_axis import load_cycle

OUT = Path(__file__).resolve().parents[2] / "outputs"
DST = OUT / "journal_numbers"
N: dict[str, str] = {}


def t4(x: float) -> str:
    s = "-" if x < 0 else ""
    return f"{s}{math.floor(abs(x) * 10000 + 1e-6) / 10000:.4f}"


def pct(x: float) -> str:
    s = "-" if x < 0 else ""
    return f"{s}{math.floor(abs(x) * 10000 + 1e-6) / 100:.2f}"


def put(key: str, val: str) -> None:
    N[key] = val


def metric_comparison() -> None:
    m = pd.read_csv(OUT / "metric_comparison" / "axis_means.csv").set_index("axis")
    for ax in m.index:
        for c in ["J_A", "OC_A", "contradiction_A", "top5", "top10", "spearman", "kendall_tau_b", "rbo_ext_p90"]:
            put(f"mc.{ax}.{c}", t4(m.loc[ax, c]))
    p = pd.read_csv(OUT / "metric_comparison" / "pairs.csv")
    nested = int(((p.J_A < 1) & (p.contradiction_A == 0)).sum())
    eq = int((abs(p.J_A - p.size_ratio_A) < 1e-9).sum())
    assert (len(p), int((p.J_A < 1).sum()), nested, eq, int(p.contradiction_A.sum())) == (45, 15, 14, 44, 1)
    put("mc.top5_all_one", str(int((p.top5 == 1).sum())))
    r = pd.read_csv(OUT / "metric_comparison" / "metric_rank_corr.csv", index_col=0).loc["J_A"].drop("J_A")
    put("mc.ja_corr_min", t4(r.min())); put("mc.ja_corr_max", t4(r.max()))
    put("mc.ja_corr_min_metric", r.idxmin())
    put("mc.ja_corr_rank_min", t4(r.drop(["top6"]).min())); put("mc.ja_corr_rank_max", t4(r.drop(["top6"]).max()))
    v = pd.read_csv(OUT / "metric_comparison" / "vectors.csv")
    lr = v[(v.model == "logistic_regression") & (v.method == "SHAP")].set_index("year")
    for y in [2015, 2021, 2023]:
        ratio = lr.loc[y, "last_A_over_mean"] if "Sex" == lr.loc[y, "last_A"] else lr.loc[y, "first_B_over_mean"]
        put(f"mc.sex_ratio_{y}", t4(ratio))
    put("mc.ja_jump", t4(1 - 5 / 6))
    b = pd.read_csv(OUT / "metric_comparison" / "bootstrap_group_a.csv")
    put("mc.boot_B", str(int(b.n_boot.iloc[0])))


def brfss() -> None:
    perf = pd.read_csv(OUT / "brfss_multiseed" / "pipeline" / "performance.csv")
    for mdl, g in perf.groupby("model"):
        put(f"br.auc.{mdl}", f"{t4(g.auc.mean())} ± {t4(g.auc.std())}")
    p = pd.read_csv(OUT / "brfss_multiseed" / "pipeline" / "pairs.csv")
    for ax in ["seed", "year", "method"]:
        g = p[p.axis == ax]
        for c in ["J_A", "spearman", "rbo_ext_p90", "contradiction_A"]:
            put(f"br.{ax}.{c}", t4(g[c].mean()))
    for meth in ["SHAP", "PI"]:
        g = p[(p.axis == "model") & (p.method == meth)]
        for c in ["J_A", "spearman", "rbo_ext_p90", "contradiction_A"]:
            put(f"br.model_{meth}.{c}", t4(g[c].mean()))
        s = p[(p.axis == "seed") & (p.method == meth)]
        put(f"br.seed_{meth}.spearman", t4(s.spearman.mean()))
    for cw in ["pipeline", "none", "balanced"]:
        q = pd.read_csv(OUT / "brfss_multiseed" / cw / "pairs.csv")
        q = q[(q.axis != "seed") & q.seed.isin([0, 1, 2])]
        ms = q[(q.axis == "model") & (q.method == "SHAP")]
        mt = q[q.axis == "method"]
        put(f"cw.{cw}.model_SHAP.J_A", t4(ms.J_A.mean()))
        put(f"cw.{cw}.model_SHAP.contradiction", t4(ms.contradiction_A.mean()))
        put(f"cw.{cw}.model_SHAP.spearman", t4(ms.spearman.mean()))
        put(f"cw.{cw}.method.J_A", t4(mt.J_A.mean()))
        put(f"cw.{cw}.method.spearman", t4(mt.spearman.mean()))
        perf = pd.read_csv(OUT / "brfss_multiseed" / cw / "performance.csv")
        perf = perf[perf.seed.isin([0, 1, 2])]
        put(f"cw.{cw}.rf_auc", t4(perf[perf.model == "random_forest"].auc.mean()))
        vec = pd.read_csv(OUT / "brfss_multiseed" / cw / "vectors.csv")
        vec = vec[(vec.model == "random_forest") & (vec.method == "SHAP") & vec.seed.isin([0, 1, 2])]
        sizes = vec.groupby(["year", "seed"]).importance.apply(lambda s: int((s.abs() > s.abs().mean()).sum()))
        put(f"cw.{cw}.rf_shap_A5", f"{int((sizes == 5).sum())}/{len(sizes)}")


def variance() -> None:
    for stem in ["brfss_pipeline", "brfss_none", "brfss_balanced"]:
        for resp in ["share", "rank"]:
            s = pd.read_csv(OUT / "variance_decomposition" / f"{stem}_{resp}_pooled.csv").set_index("source").pooled_eta2
            for k in ["method", "model", "year", "residual (seed)"]:
                put(f"eta.{stem}.{resp}.{k.split()[0]}", t4(s[k]))
    for resp in ["share", "rank"]:
        d = pd.read_csv(OUT / "variance_decomposition" / f"nhanes_{resp}_pooled.csv")
        for cyc, g in d.groupby("cycle"):
            s = g.set_index("source").pooled_eta2
            for k in ["method", "model", "label", "residual (seed)"]:
                put(f"eta.nhanes_{cyc}.{resp}.{k.split()[0]}", t4(s[k]))


def nhanes() -> None:
    for cyc in ["2017-2020", "2021-2023"]:
        df = load_cycle(cyc)
        put(f"nh.{cyc}.n", f"{len(df):,}")
        diag, lab = df["Diabetes_self"], df["Diabetes_lab_a1c"]
        total = ((diag == 1) | (lab == 1)).astype(int)
        for k, v in [("diag", diag), ("lab", lab), ("total", total)]:
            put(f"nh.{cyc}.prev_{k}", t4(v.mean()))
        put(f"nh.{cyc}.discord_pct", pct(float((diag != lab).mean())))
        dn = df[(diag == 1) & (lab == 0)]
        put(f"nh.{cyc}.diag_not_lab_treated_pct", pct(float(dn["Diabetes_treated"].mean())))
    perf = pd.read_csv(OUT / "label_axis" / "performance.csv")
    for (cyc, lab), g in perf.groupby(["cycle", "label"]):
        put(f"nh.auc.{cyc}.{lab}", t4(g.auc.mean()))
        put(f"nh.pos_test.{cyc}.{lab}", f"{int(g.pos_test.min())}-{int(g.pos_test.max())}")
    pw = pd.read_csv(OUT / "label_axis_trainweighted" / "performance.csv")
    put("nh.auc_weighted_range", f"{t4(pw.groupby(['cycle','label','model']).auc.mean().min())}-{t4(pw.groupby(['cycle','label','model']).auc.mean().max())}")
    p = pd.read_csv(OUT / "label_axis" / "pairs.csv")
    p = p[p.weighting == "unweighted"]
    for (cyc, ax), g in p.groupby(["cycle", "axis"]):
        for c in ["J_A", "contradiction_A", "spearman", "rbo_ext_p90"]:
            put(f"nh.pairs.{cyc}.{ax}.{c}", t4(g[c].mean()))
    f = pd.read_csv(OUT / "label_axis" / "label_vs_seed_floor.csv")
    f = f[f.weighting == "unweighted"]
    for (met, cyc), g in f.groupby(["metric", "cycle"]):
        put(f"nh.floor.{cyc}.{met}", pct(g.label_share_below_seed_p5.mean()))
    put("nh.floor.cell_max", pct(f.label_share_below_seed_p5.max()))
    s = pd.read_csv(OUT / "label_axis" / "feature_shift.csv")
    s = s[(s.label_1 == "diag") & (s.label_2 == "lab")]
    u = s[s.weighting == "unweighted"]
    for cyc, g in u.groupby("cycle"):
        put(f"sh.nsig.{cyc}", str(int((g.q_bh < 0.05).sum())))
    sig = u[u.q_bh < 0.05].pivot_table(index=["method", "model", "feature"], columns="cycle", values="diff")
    rep = sig.dropna()
    rep = rep[np.sign(rep["2017-2020"]) == np.sign(rep["2021-2023"])]
    put("sh.n_replicated", str(len(rep)))
    for (meth, mdl, feat), row in rep.iterrows():
        put(f"sh.{meth}.{mdl}.{feat}", f"{pct(row['2017-2020'])} / {pct(row['2021-2023'])}")
    w = pd.read_csv(OUT / "label_axis_trainweighted" / "feature_shift.csv")
    w = w[(w.label_1 == "diag") & (w.label_2 == "lab") & (w.weighting == "unweighted")].set_index(["method", "model", "feature", "cycle"])
    keep = same = sigw = 0
    for (meth, mdl, feat), row in rep.iterrows():
        for cyc in ["2017-2020", "2021-2023"]:
            r = w.loc[(meth, mdl, feat, cyc)]
            keep += 1
            same += int(np.sign(r["diff"]) == np.sign(row[cyc]))
            sigw += int(np.sign(r["diff"]) == np.sign(row[cyc]) and r["q_bh"] < 0.05)
    put("sh.weighted_same_sign", f"{same}/{keep}"); put("sh.weighted_sig", f"{sigw}/{keep}")


def instance() -> None:
    s = pd.read_csv(OUT / "instance_vs_population" / "instance_summary.csv")
    for _, r in s.iterrows():
        k = f"in.{r.dataset}.{r.cohort}.{r.pair}.{r.metric}"
        put(k + ".pop", t4(r.population)); put(k + ".median", t4(r.instance_median))
        put(k + ".q25", t4(r.instance_q25)); put(k + ".q75", t4(r.instance_q75))
        put(k + ".below_pct", pct(r.share_below_population))
    t5 = s[(s.dataset == "BRFSS") & (s.metric == "top5")]
    put("in.brfss_top5_differ_pct", f"{pct(t5.share_below_population.min())}-{pct(t5.share_below_population.max())}")


def fairness() -> None:
    g = pd.read_csv(OUT / "label_fairness" / "gaps_summary.csv")
    g = g[g.truth == "lab"]
    sig = g.pivot_table(index=["cycle", "attribute"], columns="train_label", values="signed_gap_weighted_mean")
    for (cyc, att), row in sig.iterrows():
        for lab in ["diag", "lab", "total"]:
            put(f"fa.signed.{cyc}.{att}.{lab}", t4(row[lab]))
    uns = g.pivot_table(index=["cycle", "attribute"], columns="train_label", values="tpr_gap_weighted_mean")
    for (cyc, att), row in uns.iterrows():
        for lab in ["diag", "lab", "total"]:
            put(f"fa.unsigned.{cyc}.{att}.{lab}", t4(row[lab]))
    d = pd.read_csv(OUT / "label_fairness" / "gaps.csv")
    d = d[d.truth == "lab"]
    piv = d.pivot_table(index=["cycle", "seed", "model", "attribute"], columns="train_label", values="tpr_gap_weighted")
    piv["diff"] = piv["diag"] - piv["lab"]
    summ = piv.groupby(["cycle", "attribute"])["diff"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    summ.columns = ["mean", "lo", "hi"]
    for (cyc, att), r in summ.iterrows():
        put(f"fa.diff.{cyc}.{att}", f"{t4(r['mean'])} [{t4(r.lo)}, {t4(r.hi)}]")
    se = summ.loc[(slice(None), ["Income", "Education"]), :]
    put("fa.diff_inc_edu_range", f"{t4(se['mean'].min())} to {t4(se['mean'].max())}")
    put("fa.diff_inc_edu_ci_contain0", str(bool(((se.lo <= 0) & (se.hi >= 0)).all())))
    a = summ.loc[(slice(None), "Age"), :]
    put("fa.age_ci_contain0", str(bool(((a.lo <= 0) & (a.hi >= 0)).all())))
    mp = d[d.attribute == "Age"].min_group_n_pos
    put("fa.age_min_pos", f"{int(mp.min())}-{int(mp.max())}")


def income() -> None:
    d = pd.read_csv(OUT / "brfss_income_harmonised" / "delta_tpr.csv")
    for _, r in d.iterrows():
        put(f"inc.{r.model}.{r.year}", f"{t4(r.delta_tpr_native)} -> {t4(r.delta_tpr_harmonised)}")


def sensitivity() -> None:
    d = OUT / "sensitivity"
    v = pd.read_csv(d / "variance_specs.csv")
    for _, r in v.iterrows():
        put(f"se.var.{r.design}.{r.response}.{r.model_spec}.{r.source}", t4(r.eta2))
    pf = pd.read_csv(d / "variance_per_feature.csv")
    for _, r in pf.iterrows():
        put(f"se.feat.{r.design}.{r.response}.{r.source}", f"{t4(r['median'])} [{t4(r.q25)}, {t4(r.q75)}]")
    pm = pd.read_csv(d / "variance_permutation.csv")
    put("se.perm.max_p", t4(pm.p_perm.max())); put("se.perm.n", str(int(pm.n_perm.iloc[0]))); put("se.perm.tests", str(len(pm)))
    sf = pd.read_csv(d / "seed_floor_percentiles.csv")
    sf = sf[sf.weighting == "unweighted"].groupby(["percentile", "metric", "cycle"]).share_below.agg(["mean", "max"])
    for (q, met, cyc), r in sf.iterrows():
        put(f"se.floor.p{int(round(q * 100))}.{cyc}.{met}", f"{pct(r['mean'])} (max {pct(r['max'])})")
    for test in ["q_bh", "q_signflip"]:
        rep = pd.read_csv(d / f"label_shift_replicated_{test}.csv")
        put(f"se.shift.replicated.{test}", str(len(rep)))
    sfl = pd.read_csv(d / "label_shift_signflip.csv")
    u = sfl[sfl.weighting == "unweighted"]
    for cyc, g in u.groupby("cycle"):
        put(f"se.shift.nsig_signflip.{cyc}", str(int((g.q_signflip < 0.05).sum())))


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    metric_comparison(); brfss(); variance(); nhanes(); instance(); fairness(); income(); sensitivity()
    (DST / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False))
    (DST / "numbers.md").write_text("\n".join(f"- `{k}`: {v}" for k, v in N.items()) + "\n")
    print(f"{len(N)} values written to {DST}")


if __name__ == "__main__":
    main()
