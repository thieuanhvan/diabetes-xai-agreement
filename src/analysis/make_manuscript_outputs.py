"""
Numbers, tables and figures for manuscript v32 (outcome-definition headline).

Reads only stored outputs (label_axis*, label_controls, label_null, detection_power)
and writes
    outputs/manuscript_numbers/numbers.json   every decimal quoted in manuscript v32
    outputs/manuscript_tables/*.tex           table bodies (main and supplementary)
    outputs/manuscript_figures/fig1_shift_vs_null.pdf
Values are truncated, not rounded (four decimals; percentages and percentage
points to two).

Usage:
    python -m src.analysis.make_manuscript_outputs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import label_controls_core as C
from src.analysis.manuscript_numbers import pct, q4, t4

OUT = C.OUT_DIR
DT, DF, DN = OUT / "manuscript_tables", OUT / "manuscript_figures", OUT / "manuscript_numbers"
N: dict[str, str] = {}
K = ["method", "model", "feature"]
MODEL = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGBoost"}
SECOND = {"diabetes": "lab", "hypertension": "measured"}
PM = {"diabetes": "labM", "hypertension": "measM"}
OUTCOMES = ["diabetes", "hypertension"]


def put(k: str, v: str) -> str:
    N[k] = v
    return v


def pp(x: float) -> str:
    """Share difference (fraction) -> percentage points, truncated to two decimals."""
    return pct(x)


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def write(name: str, rows: list[str]) -> None:
    DT.mkdir(parents=True, exist_ok=True)
    (DT / name).write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------- per-shift master
def master(outcome: str) -> pd.DataFrame:
    c = pd.read_csv(OUT / "label_controls" / f"survival_{outcome}.csv")
    r = pd.read_csv(OUT / "label_null" / f"{outcome}_risk" / "survival.csv")
    n = pd.read_csv(OUT / "label_null" / outcome / "survival.csv")
    keep = ["p_dir_1720", "p_dir_2123", "null_mean_1720", "null_mean_2123", "diff_obs5_1720",
            "diff_obs5_2123", "exceed_null_both", "beyond_null_dir_both"]
    m = c.merge(r[K + keep].add_prefix("auc_").rename(columns={f"auc_{k}": k for k in K}), on=K)
    m = m.merge(n[K + keep].add_prefix("rnd_").rename(columns={f"rnd_{k}": k for k in K}), on=K)
    m["prev"] = m[f"survive_{PM[outcome]}"]
    m["core"] = m["prev"] & m["auc_beyond_null_dir_both"]
    return m


def null_draws(outcome: str, kind: str) -> pd.DataFrame:
    d = OUT / "label_null" / (f"{outcome}_risk" if kind == "auc" else outcome)
    return pd.read_csv(d / "null_distribution.csv")


# ---------------------------------------------------------------- Table 1
def table_cohorts() -> None:
    rows = []
    for o in OUTCOMES:
        perf = pd.read_csv(OUT / "label_null" / f"{o}_risk" / "auc_by_label_kind.csv").set_index("kind")
        rnd = pd.read_csv(OUT / "label_null" / o / "auc_by_label_kind.csv").set_index("kind")
        for i, cyc in enumerate(C.CYCLES):
            df = C.load_base(o, cyc)
            s = SECOND[o]
            disc = ((df.diag != df[s]).mean())
            key = f"{o}_{cyc}"
            vals = [put(f"n_{key}", f"{len(df):,}".replace(",", "{,}")),
                    put(f"prev_diag_{key}", t4(df.diag.mean())), put(f"prev_second_{key}", t4(df[s].mean())),
                    put(f"disc_{key}", pct(disc)),
                    put(f"auc_diag_{key}", t4(perf.loc["diag", cyc])), put(f"auc_second_{key}", t4(perf.loc[s, cyc])),
                    put(f"auc_rndnull_{key}", t4(rnd.loc["null_labels", cyc])),
                    put(f"auc_matchnull_{key}", t4(perf.loc["null_labels", cyc]))]
            lab = ("Diabetes" if o == "diabetes" else "Hypertension") if i == 0 else ""
            rows.append(f"{lab} & {cyc.replace('-', '--')} & " + " & ".join(vals) + r" \\")
        if o == "diabetes":
            rows.append(r"\addlinespace")
    write("cohorts.tex", rows)


# ---------------------------------------------------------------- Table 2
def table_robustness(ms: dict) -> None:
    def cnt(o, col):
        return int(ms[o][col].sum()) if col in ms[o] else None

    spec = [
        ("Replicated shifts (corrected $t$, BH $q<0.05$ in both cycles, same sign)", None, None),
        ("Second label matched to diagnosis prevalence", "survive_labM", "survive_measM"),
        ("Second label at a lower threshold (HbA1c ${\\ge}6.3\\%$; ${\\ge}130/80$ mmHg)", "survive_lab63", "survive_meas130"),
        ("Class weighting in all models", "survive_cw_balanced", "survive_cw_balanced"),
        ("No class weighting", "survive_cw_none", "survive_cw_none"),
        ("HbA1c ${\\ge}6.5\\%$ or fasting glucose ${\\ge}126$ mg/dL", "survive_labG", None),
        ("HbA1c ${\\ge}6.5\\%$ or glucose-lowering treatment", "survive_labTx", None),
        ("Beyond random-discordance null", "rnd_beyond_null_dir_both", "rnd_beyond_null_dir_both"),
        ("Beyond AUC-matched discordance null", "auc_beyond_null_dir_both", "auc_beyond_null_dir_both"),
        ("Core: prevalence-matched and beyond AUC-matched null", "core", "core"),
    ]
    rows = []
    for i, (lab, cd, ch) in enumerate(spec):
        vals = []
        for o, col in [("diabetes", cd), ("hypertension", ch)]:
            n = len(ms[o])
            if i == 0:
                vals.append(str(n))
            elif col is None:
                vals.append("--")
            else:
                vals.append(f"{cnt(o, col)} / {n}")
        if i == len(spec) - 1:
            rows.append(r"\midrule")
            lab = r"\textbf{" + lab + "}"
            vals = [r"\textbf{" + v + "}" for v in vals]
        rows.append(f"{lab} & {vals[0]} & {vals[1]} " + r"\\")
    write("robustness.tex", rows)
    # sensitivity: |obs| > p95 |null| rule
    for o in OUTCOMES:
        m = ms[o]
        put(f"{o}_rnd_abs", str(int(m.rnd_exceed_null_both.sum())))
        put(f"{o}_auc_abs", str(int(m.auc_exceed_null_both.sum())))
        put(f"{o}_core_abs", str(int((m.prev & m.auc_exceed_null_both).sum())))


# ---------------------------------------------------------------- Table 3
def holdout(o: str):
    f = OUT / "holdout_2015_2016" / o / "holdout_summary.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)[[c for c in ["method", "model", "feature", "replicated", "beyond", "diff", "q",
                                       "p_dir", "same_sign", "diff_obs5", "null_mean"]
                          if c in pd.read_csv(f, nrows=1).columns]]


def holdout_tables(ms: dict) -> None:
    rows = []
    for o in OUTCOMES:
        h = holdout(o)
        if h is None:
            continue
        m = ms[o][["method", "model", "feature", "sign", "core"]].merge(h, on=["method", "model", "feature"])
        has_null = "beyond" in m
        ok = m.replicated & (m.beyond if has_null else True)
        put(f"ho_core_rep_{o}", str(int(m[m.core].replicated.sum())))
        put(f"ho_core_n_{o}", str(int(m.core.sum())))
        put(f"ho_all_rep_{o}", str(int(m.replicated.sum())))
        put(f"ho_all_n_{o}", str(len(m)))
        put(f"ho_all_sign_{o}", str(int(m.same_sign.sum())))
        if has_null:
            put(f"ho_core_full_{o}", str(int(ok[m.core].sum())))
            put(f"ho_all_full_{o}", str(int(ok.sum())))
        g = (OUT / "holdout_2015_2016" / o / "gamma.txt")
        if g.exists():
            put(f"ho_gamma_{o}", f"{float(g.read_text()):.2f}")
        a = OUT / "holdout_2015_2016" / o / "auc_by_label_kind.csv"
        if a.exists():
            aa = pd.read_csv(a).set_index("kind")
            put(f"ho_auc_second_{o}", t4(aa.loc[SECOND[o]].mean()))
            put(f"ho_auc_null_{o}", t4(aa.loc["null_labels"].mean()))
            put(f"ho_auc_diag_{o}", t4(aa.loc["diag"].mean()))
        df = C.load_base(o, "2015-2016")
        put(f"ho_n_{o}", f"{len(df):,}".replace(",", "{,}"))
        put(f"ho_prev_diag_{o}", t4(df.diag.mean())); put(f"ho_prev_second_{o}", t4(df[SECOND[o]].mean()))
        put(f"ho_disc_{o}", pct((df.diag != df[SECOND[o]]).mean()))
        for r in m.sort_values(["sign", "feature", "method", "model"]).itertuples():
            lab = "Diabetes" if o == "diabetes" else "Hypertension"
            larger = "diagnosis" if r.sign > 0 else ("HbA1c" if o == "diabetes" else "measured")
            pdir = t4(r.p_dir) if has_null else "--"
            beyond = mark(r.beyond) if has_null else "--"
            rows.append(f"{lab} & {larger} & {r.method} & {MODEL[r.model]} & {esc(r.feature)} & {mark(r.core)} & "
                        f"{pp(r.diff)} & {q4(r.q)} & {mark(r.replicated)} & {pdir} & {beyond} " + r"\\")
            N[f"ho_{o}_{r.method}_{r.model}_{r.feature}"] = pp(r.diff)
            if has_null:
                N[f"hop_{o}_{r.method}_{r.model}_{r.feature}"] = t4(r.p_dir)
    write("holdout.tex", rows)


GAMMA_STEP = {"diabetes": 0.4, "hypertension": 0.25}


def robust_subset(ms: dict, o: str) -> set:
    """Reference-robust subset: core under the main AUC-matched null, under the
    single gamma one grid step away, under cycle-specific gamma, and replicated
    and beyond the null in the held-out cycle. Robustness to the evaluated
    risk-structured nulls only, not to every label control."""
    m = ms[o]
    keys = set(map(tuple, m[m.core][K].to_numpy()))
    s = pd.read_csv(OUT / "label_null" / f"{o}_risk_g{GAMMA_STEP[o]:g}" / "survival.csv")
    s = m[K + ["prev"]].merge(s[K + ["beyond_null_dir_both"]], on=K)
    keys &= set(map(tuple, s[s.prev & s.beyond_null_dir_both][K].to_numpy()))
    cm = pd.read_csv(OUT / "reviewer_checks" / "cycle_matched_detail.csv")
    keys &= set(map(tuple, cm[(cm.outcome == o) & cm.core_cm][K].to_numpy()))
    h = holdout(o)
    keys &= set(map(tuple, h[h.replicated & h.beyond][K].to_numpy()))
    return keys


def table_core(ms: dict) -> None:
    rows = []
    for o in OUTCOMES:
        m = ms[o].copy()
        m["larger"] = np.where(m.sign > 0, "diagnosis", "HbA1c" if o == "diabetes" else "measured")
        first = True
        for larger in ["diagnosis", "HbA1c" if o == "diabetes" else "measured"]:
            g = m[m.larger == larger]
            for f, gf in sorted(g.groupby("feature"), key=lambda x: (-x[1].core.sum(), x[0])):
                core = gf[gf.core]
                if len(core):
                    v = np.concatenate([core.diff_v1_1720.to_numpy(), core.diff_v1_2123.to_numpy()])
                    lo, hi = v[np.argmin(np.abs(v))], v[np.argmax(np.abs(v))]
                    rng = f"{pp(lo)} to {pp(hi)}"
                    combos = ", ".join(f"{r.method}-{MODEL[r.model]}" for r in core.itertuples())
                else:
                    rng, combos = "--", "--"
                lab = ("Diabetes" if o == "diabetes" else "Hypertension") if first else ""
                first = False
                ho = holdout(o)
                if ho is not None and len(core):
                    h = core.merge(ho, on=["method", "model", "feature"])
                    ok = h.replicated & (h.beyond if "beyond" in h else True)
                    hcol = f"{int(ok.sum())} / {len(core)}"
                else:
                    hcol = "--"
                rob = robust_subset(ms, o)
                nrob = sum((r.method, r.model, r.feature) in rob for r in core.itertuples())
                rcol = f"{nrob} / {len(core)}" if len(core) else "--"
                rows.append(f"{lab} & {larger} & {esc(f)} & {int(gf.core.sum())} / {len(gf)} & {rng} & {hcol} & {rcol} & {combos} " + r"\\")
        if o == "diabetes":
            rows.append(r"\addlinespace")
    write("core.tex", rows)
    for o in OUTCOMES:
        put(f"robust_n_{o}", str(len(robust_subset(ms, o))))


# ---------------------------------------------------------------- Table 4
METRICS = [("J_A", r"$J_A$"), ("OC_A", r"$\mathrm{OC}_A$"), ("size_ratio_A", "Size ratio"),
           ("top5", "Top-5 overlap"), ("top10", "Top-10 overlap"),
           ("feature_agreement", "Feature agreement$^{a}$"), ("contradiction_A", "Contradiction flag$^{b}$"),
           ("rbo_ext_p90", r"$\mathrm{RBO}_{\mathrm{ext}}$"), ("spearman", r"Spearman $\rho$"),
           ("kendall_tau_b", r"Kendall $\tau_b$"), ("pairwise_rank_agreement", "Pairwise rank agreement$^{a}$"),
           ("weighted_tau", r"Weighted $\tau$")]


def table_detection() -> None:
    b = {o: pd.read_csv(OUT / "detection_power" / o / "detection_config.csv") for o in OUTCOMES}
    lam = pd.read_csv(OUT / "detection_power" / "diabetes" / "mds_joint_lambda.csv")
    lam = lam[lam.positive]
    vis = pd.read_csv(OUT / "detection_power" / "diabetes" / "feature_visibility.csv")
    vis_h = pd.read_csv(OUT / "detection_power" / "hypertension" / "feature_visibility.csv")
    put("vis_A_changed", pct(vis.share_A_membership_changed.mean()))
    put("vis_top5_changed", pct(vis.share_top5_membership_changed.mean()))
    put("vis_median_abs_pp_diab", pct(vis.share_diff_pp.abs().median() / 100))
    put("vis_median_abs_pp_htn", pct(vis_h.share_diff_pp.abs().median() / 100))
    rows = []
    d, h = b["diabetes"], b["hypertension"]
    npos, nneg, nh = d[d.positive].groupby(["cycle", "method", "model"]).ngroups, \
        d[~d.positive].groupby(["cycle", "method", "model"]).ngroups, h.groupby(["cycle", "method", "model"]).ngroups
    put("n_pos_cfg", str(npos)), put("n_neg_cfg", str(nneg)), put("n_cfg_htn", str(nh))
    for key, lab in METRICS:
        dp = int(d[(d.metric == key) & d.positive].flagged.sum())
        dn = int(d[(d.metric == key) & ~d.positive].flagged.sum())
        hh = int(h[(h.metric == key)].flagged.sum())
        lk = lam[lam.metric == key].lambda_star
        le1 = int((lk <= 1.0).sum())
        med = np.median(lk)
        meds = r"$>3$" if not np.isfinite(med) else f"{med:.1f}"
        rows.append(f"{lab} & {dp} / {npos} & {dn} / {nneg} & {hh} / {nh} & {le1} / {npos} & {meds} " + r"\\")
        N[f"det_{key}"] = f"{dp}/{npos}"
        if np.isfinite(med):
            N[f"lam_{key}"] = f"{med:.1f}"
    write("detection.tex", rows)


# ---------------------------------------------------------------- supplementary per-shift tables
def mark(b) -> str:
    return r"\checkmark" if bool(b) else r"$\times$"


def ci(outcome: str, method: str, model: str, feature: str, cycle: str):
    """Diagnosis-minus-measured shift and its 95% interval from the corrected
    resampled t statistic (se = diff / t; t quantile with J - 1 df)."""
    from scipy.stats import t as t_dist
    fs = pd.read_csv(C.V1_DIR[outcome] / "feature_shift.csv")
    r = fs[(fs.weighting == "unweighted") & (fs.label_1 == "diag") & (fs.label_2 == SECOND[outcome])
           & (fs.cycle == cycle) & (fs.method == method) & (fs.model == model) & (fs.feature == feature)].iloc[0]
    se = abs(r["diff"] / r.t_corrected)
    q = t_dist.ppf(0.975, r.n_seeds - 1)
    lo, hi = r["diff"] - q * se, r["diff"] + q * se
    key = f"ci_{outcome}_{method}_{model}_{feature}_{cycle}"
    N[key] = f"{pp(r['diff'])} [{pp(lo)}, {pp(hi)}]"
    return r["diff"], lo, hi


def table_validation() -> None:
    rows = []
    for o in OUTCOMES:
        d = OUT / "null_validation" / o
        if not (d / "false_positives.csv").exists():
            continue
        fp = pd.read_csv(d / "false_positives.csv")
        put(f"fp_zero_mean_{o}", f"{fp.replicated_vs_zero.mean():.2f}")
        put(f"fp_zero_max_{o}", str(int(fp.replicated_vs_zero.max())))
        put(f"fp_zero_any_{o}", str(int((fp.replicated_vs_zero > 0).sum())))
        put(f"fp_full_any_{o}", str(int((fp.beyond_other_nulls > 0).sum())))
        put(f"fp_n_{o}", str(len(fp)))
        lab = "Diabetes" if o == "diabetes" else "Hypertension"
        rows.append(f"{lab} & AUC-matched null labels ({len(fp)}) & "
                    f"{fp.replicated_vs_zero.mean():.2f} (max {int(fp.replicated_vs_zero.max())}); "
                    f"{int((fp.replicated_vs_zero > 0).sum())} of {len(fp)} labels & "
                    f"{int((fp.beyond_other_nulls > 0).sum())} of {len(fp)} labels " + r"\\")
        if (d / "positive_control_detail.csv").exists():
            det = pd.read_csv(d / "positive_control_detail.csv")
            det["target"] = det.label.str.split("_").str[1]
            for f, g in det.groupby("target"):
                nlab = g.label.nunique()
                hit = int(g[(g.feature == f) & (g.sign < 0)].beyond.sum())
                other = g[(g.feature != f) & g.beyond]
                same, comp = int((other.sign < 0).sum()), int((other.sign > 0).sum())
                put(f"pc_hit_{o}_{f}", f"{hit}/{nlab * 6}")
                put(f"pc_same_{o}_{f}", str(same))
                put(f"pc_comp_{o}_{f}", str(comp))
                rows.append(f" & Labels planted on {f} ({nlab}) & -- & {f} flagged in {hit} of {nlab * 6}; "
                            f"other features toward the planted label {same}, away (compensation) {comp} " + r"\\")
        if o == "diabetes":
            rows.append(r"\addlinespace")
    write("validation.tex", rows)


def compensation_numbers() -> None:
    """Mean SHAP-LR shifts (both cycles, seeds 0-4) under the real hypertension
    measured label and under the labels planted on age, for the compensation
    comparison in the text."""
    o = "hypertension"
    pv = OUT / "null_validation" / o / "planted_vectors.csv"
    if not pv.exists():
        return
    v = pd.concat([pd.read_csv(OUT / "label_null" / f"{o}_risk" / "vectors.csv"), pd.read_csv(pv)])
    def mean_shift(lab, feat):
        s = C.shift_table(v, "diag", lab)
        s = s[(s.method == "SHAP") & (s.model == "logistic_regression") & (s.feature == feat)]
        return s["diff"].mean()
    planted = [f"plant_Age_{r}" for r in range(1, 4)]
    for feat in ["Age", "HighChol", "GenHlth", "Diabetes_self"]:
        put(f"comp_real_{feat}", pp(mean_shift("measured", feat)))
        put(f"comp_plantAge_{feat}", pp(np.mean([mean_shift(l, feat) for l in planted])))


def reviewer_tables() -> None:
    """Supplementary tables for the null and detection checks (run_reviewer_checks)."""
    d = OUT / "reviewer_checks"
    if not d.exists():
        return
    # AUC match per cycle x model
    a = pd.read_csv(d / "auc_match.csv")
    rows = []
    for r in a.itertuples():
        lab = ("Diabetes" if r.outcome == "diabetes" else "Hypertension")
        rows.append(f"{lab} & {r.cycle.replace('-', '--')} & {MODEL[r.model]} & {t4(r.diag)} & {t4(r.measured)} & "
                    f"{t4(r.null_mean)} ({t4(r.null_sd)}) & {t4(r.null_min)}--{t4(r.null_max)} & {t4(r.diff)} " + r"\\")
        N[f"aucm_{r.outcome}_{r.cycle}_{r.model}"] = t4(r.diff)
    write("auc_match.tex", rows)
    put("aucm_maxabs", t4(a["diff"].abs().max()))
    h = a[(a.outcome == "hypertension") & (a.cycle == "2021-2023")]["diff"]
    put("aucm_htn2123_min", t4(h.min())); put("aucm_htn2123_max", t4(h.max()))
    h = a[(a.outcome == "hypertension") & (a.cycle == "2017-2020")]["diff"]
    put("aucm_htn1720_min", t4(h.min())); put("aucm_htn1720_max", t4(h.max()))
    dd = a[(a.outcome == "diabetes")]["diff"]
    put("aucm_diab_min", t4(dd.min())); put("aucm_diab_max", t4(dd.max()))
    # disjoint-split calibration
    rows = []
    for o in OUTCOMES:
        f = d / f"calibration_disjoint_{o}.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f, header=[0, 1], index_col=[0, 1])
        first = True
        for (kind, gamma), r in c.iterrows():
            lab = ("Diabetes" if o == "diabetes" else "Hypertension") if first else ""
            first = False
            g = "real label" if kind == "real" else f"null, $\\gamma={float(gamma):g}$"
            rows.append(f"{lab} & {g} & " + " & ".join(t4(x) for x in r.to_numpy()) + r" \\")
        if o == "diabetes":
            rows.append(r"\addlinespace")
    write("calibration_disjoint.tex", rows)
    # feature-level false positives
    fp = pd.read_csv(d / "fp_feature.csv")
    for r in fp.itertuples():
        put(f"fpfeat_zero_{r.outcome}", pct(r.rate_vs_zero))
        put(f"fpfeat_full_{r.outcome}", pct(r.rate_full))
        put(f"fpfeat_n_zero_{r.outcome}", str(r.replicated_vs_zero))
    # detection variants
    da = pd.read_csv(d / "detection_alt.csv")
    rows = []
    names = dict(METRICS)
    for o in OUTCOMES:
        first = True
        for key, lab in METRICS:
            g = da[(da.outcome == o) & (da.metric == key)].set_index("reference")
            a1, a2 = g.loc["diag+measured"], g.loc["diag+measured+composite"]
            ol = ("Diabetes" if o == "diabetes" else "Hypertension") if first else ""
            first = False
            rows.append(f"{ol} & {lab} & {a2.rep_pos} & {a1.rep_pos} & {a2.core_pos} & {a2.core_neg} & {a1.core_pos} & {a1.core_neg} " + r"\\")
        if o == "diabetes":
            rows.append(r"\addlinespace")
    write("detection_alt.tex", rows)
    # CLR
    c = pd.read_csv(d / "clr_core.csv")
    for o in OUTCOMES:
        g = c[c.outcome == o]
        put(f"clr_kept_{o}", str(int(g.clr_replicated.sum())))
    cp = pd.read_csv(d / "clr_pseudocount.csv")
    for o in OUTCOMES:
        g = cp[cp.outcome == o]
        for pc in ["1e-05", "0.001"]:
            put(f"clr_kept_{o}_{pc}", str(int(g[f"clr_replicated_{pc}"].sum())))
    rows = [f"{('Diabetes' if r.outcome == 'diabetes' else 'Hypertension')} & {r.method} & {MODEL[r.model]} & {esc(r.feature)} & "
            f"{'diagnosis' if r.sign > 0 else ('HbA1c' if r.outcome == 'diabetes' else 'measured')} & "
            f"{mark(getattr(r, 'clr_replicated_1e_05'))} & {mark(getattr(r, 'clr_replicated_0_0001'))} & "
            f"{mark(getattr(r, 'clr_replicated_0_001'))} " + r"\\"
            for r in cp.rename(columns=lambda x: x.replace("-", "_").replace(".", "_")).itertuples()]
    write("clr_core.tex", rows)
    # gamma sensitivity
    f = d / "gamma_sensitivity.csv"
    if f.exists():
        gs = pd.read_csv(f)
        rows = []
        for _, r in gs.iterrows():
            lab = "Diabetes" if r.outcome == "diabetes" else "Hypertension"
            rows.append(f"{lab} & {r.gamma:g} & {t4(r['auc_null_2017-2020'])} & {t4(r['auc_null_2021-2023'])} & "
                        f"{r.beyond} & {r.core} & {r.core_kept} / {r.core_main} " + r"\\")
            put(f"gs_core_{r.outcome}_{r.gamma:g}", str(r.core)); put(f"gs_kept_{r.outcome}_{r.gamma:g}", str(r.core_kept))
            put(f"gs_auc_{r.outcome}_{r.gamma:g}_1720", t4(r['auc_null_2017-2020'])); put(f"gs_auc_{r.outcome}_{r.gamma:g}_2123", t4(r['auc_null_2021-2023']))
        f = d / "cycle_matched.csv"
        if f.exists():
            cm = pd.read_csv(f)
            gam = {"diabetes": "0.30 / 0.40", "hypertension": "0.35 / 0.25"}
            for r in cm.itertuples():
                lab = "Diabetes" if r.outcome == "diabetes" else "Hypertension"
                rows.append(f"{lab} & {gam[r.outcome]} (per cycle) & -- & -- & {r.beyond} & {r.core} & {r.core_kept} / {r.core_main} " + r"\\")
                put(f"cm_core_{r.outcome}", str(r.core)); put(f"cm_kept_{r.outcome}", str(r.core_kept))
        write("gamma_sensitivity.tex", rows)


def table_shift_controls(ms: dict) -> None:
    for o in OUTCOMES:
        m = ms[o].sort_values(["sign", "feature", "method", "model"])
        cols = ["survive_labM", "survive_lab63", "survive_labG", "survive_labTx", "survive_cw_balanced",
                "survive_cw_none"] if o == "diabetes" else ["survive_measM", "survive_meas130",
                                                            "survive_cw_balanced", "survive_cw_none"]
        rows = []
        for r in m.itertuples():
            larger = "diagnosis" if r.sign > 0 else SECOND[o].replace("lab", "HbA1c")
            vals = [larger, r.method, MODEL[r.model], esc(r.feature)]
            for cyc in C.CYCLES:
                d, lo, hi = ci(o, r.method, r.model, r.feature, cyc)
                vals.append(f"{pp(d)} [{pp(lo)}, {pp(hi)}]")
            vals += [mark(getattr(r, c)) for c in cols]
            vals += [mark(r.rnd_beyond_null_dir_both), mark(r.auc_beyond_null_dir_both), mark(r.core)]
            rows.append(" & ".join(vals) + r" \\")
        write(f"shift_controls_{o}.tex", rows)


def table_shift_null(ms: dict) -> None:
    for o in OUTCOMES:
        m = ms[o].sort_values(["sign", "feature", "method", "model"])
        rows = []
        for r in m.itertuples():
            vals = [r.method, MODEL[r.model], esc(r.feature)]
            for cyc in ["1720", "2123"]:
                vals += [pp(getattr(r, f"auc_diff_obs5_{cyc}")), pp(getattr(r, f"rnd_null_mean_{cyc}")),
                         t4(getattr(r, f"rnd_p_dir_{cyc}")), pp(getattr(r, f"auc_null_mean_{cyc}")),
                         t4(getattr(r, f"auc_p_dir_{cyc}"))]
            rows.append(" & ".join(vals) + r" \\")
        write(f"shift_null_{o}.tex", rows)


def table_calibration() -> None:
    rows = []
    for o in OUTCOMES:
        c = pd.read_csv(OUT / "label_null" / f"calibration_{o}.csv", header=[0, 1], index_col=[0, 1])
        first = True
        for (kind, gamma), r in c.iterrows():
            lab = ("Diabetes" if o == "diabetes" else "Hypertension") if first else ""
            first = False
            g = "real label" if kind == "real" else f"null, $\\gamma={float(gamma):g}$"
            rows.append(f"{lab} & {g} & " + " & ".join(t4(x) for x in r.to_numpy()) + r" \\")
        if o == "diabetes":
            rows.append(r"\addlinespace")
    write("null_calibration.tex", rows)


# ---------------------------------------------------------------- figure 1
def figure(ms: dict) -> None:
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7,
        "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
    col = {"2017-2020": "#2a6f97", "2021-2023": "#c8553d"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 6.6), gridspec_kw={"width_ratios": [1, 1]})
    for ax, o in zip(axes, OUTCOMES):
        m = ms[o].copy()
        m["absmean"] = (m.diff_v1_1720 + m.diff_v1_2123) / 2
        m = m.sort_values("absmean").reset_index(drop=True)
        nd = null_draws(o, "auc")
        for i, r in m.iterrows():
            for j, cyc in enumerate(C.CYCLES):
                y = i + (0.18 if j == 0 else -0.18)
                x = nd[(nd.cycle == cyc) & (nd.method == r.method) & (nd.model == r.model)
                       & (nd.feature == r.feature)]["diff"].to_numpy() * 100
                lo, hi = np.quantile(x, [0.025, 0.975])
                ax.plot([lo, hi], [y, y], color="#b9b7b0", lw=2.2, solid_capstyle="butt", zorder=1)
                obs = r[f"auc_diff_obs5_{'1720' if j == 0 else '2123'}"] * 100
                beyond = r[f"auc_p_dir_{'1720' if j == 0 else '2123'}"] < 0.05
                ax.scatter([obs], [y], s=14, color=col[cyc] if beyond else "white",
                           edgecolor=col[cyc], linewidth=0.9, zorder=3)
        lab = [f"{'★ ' if r.core else ''}{r.method}-{MODEL[r.model]} {r.feature}" for r in m.itertuples()]
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(lab, fontsize=6)
        ax.axvline(0, color="#52514e", lw=0.6)
        second = "HbA1c" if o == "diabetes" else "BP"
        ax.set_xlabel(f"Share shift, diagnosis − {second} label (pp)")
        ax.set_title("Diabetes" if o == "diabetes" else "Hypertension", loc="left", fontweight="bold")
        ax.set_ylim(-0.7, len(m) - 0.3)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=col[c], label=c.replace("-", "–")) for c in C.CYCLES]
    h += [Line2D([], [], marker="o", ls="", mfc="white", mec="#52514e", label="within null (one-sided p ≥ 0.05)"),
          Line2D([], [], color="#b9b7b0", lw=2.2, label="AUC-matched null, 2.5–97.5%")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    DF.mkdir(parents=True, exist_ok=True)
    fig.savefig(DF / "fig1_shift_vs_null.pdf", metadata={"CreationDate": None})
    plt.close(fig)


# ---------------------------------------------------------------- Group A rule sensitivity
def table_cabc_rule() -> None:
    """Set metrics with the BreakEven-only (above-mean) rule used throughout
    versus the published cABC A|B limit (reviewer_checks.cabc_std)."""
    base = {"BreakEven (above mean)": OUT / "detection_power",
            "Published cABC": OUT / "reviewer_checks" / "cabc_standard"}
    if not (base["Published cABC"] / "diabetes" / "detection_config.csv").exists():
        return
    names = [("J_A", r"$\JA$"), ("OC_A", r"$\OC$"), ("size_ratio_A", "Size ratio"),
             ("contradiction_A", "Contradiction flag")]
    rows = []
    for rule, b in base.items():
        dd = pd.read_csv(b / "diabetes" / "detection_config.csv")
        dl = pd.read_csv(b / "diabetes" / "mds_joint_lambda.csv")
        hd = pd.read_csv(b / "hypertension" / "detection_config.csv")
        first = True
        for k, lab in names:
            x = dd[dd.metric == k]; pos, neg = x[x.positive], x[~x.positive]
            lp = dl[(dl.metric == k) & dl.positive]
            h = hd[hd.metric == k]
            key = "be" if rule.startswith("Break") else "std"
            fp = put(f"cabc_{key}_{k}_pos", f"{int(pos.flagged.sum())}/{len(pos)}")
            rows.append(f"{rule if first else ''} & {lab} & {fp.replace('/', ' / ')} & "
                        f"{int(neg.flagged.sum())} / {len(neg)} & {int((lp.lambda_star <= 1).sum())} / {len(lp)} & "
                        f"{int(h.flagged.sum())} / {len(h)} " + r"\\")
            first = False
        v = pd.read_csv(b / "diabetes" / "feature_visibility.csv")
        put(f"cabc_{'be' if rule.startswith('Break') else 'std'}_Achanged", pct(v.share_A_membership_changed.mean()))
        rows.append(r"\addlinespace")
    write("cabc_rule.tex", rows[:-1])
    g = pd.read_csv(OUT / "reviewer_checks" / "cabc_standard" / "group_a_agreement.csv")
    for r in g.itertuples():
        if r.set.startswith("NHANES"):
            o = r.set.split()[1]
            put(f"cabc_identical_{o}", f"{r.identical_group_a} of {r.n}")
        elif r.set == "conference vectors":
            put("cabc_identical_conf", f"{r.identical_group_a} of {r.n}")
        else:
            a, nst, c = str(r.identical_group_a).split("/")
            put("cabc_conf_pairs_std", f"{a}/{nst}/{c}")


def coding_numbers() -> None:
    f = OUT / "reviewer_checks" / "menthlth_ordinal.csv"
    if f.exists():
        r = pd.read_csv(f)
        for o, g in r.groupby("outcome"):
            put(f"ment_surv_{o}", f"{int(g.survive_ordinal.sum())} of {len(g)}")
        rows = []
        for o in OUTCOMES:
            g = r[r.outcome == o]
            for i, x in enumerate(g.itertuples()):
                lab = ("Diabetes" if o == "diabetes" else "Hypertension") if i == 0 else ""
                larger = "diagnosis" if x.sign > 0 else ("HbA1c" if o == "diabetes" else "measured")
                rows.append(f"{lab} & {x.method} & {esc(x.feature)} & {larger} & {mark(bool(x.core))} & "
                            f"{pp(x.diff_v1_1720)} & {pp(x.diff_v1_2123)} & {pp(x.diff_ordinal_1720)} & "
                            f"{pp(x.diff_ordinal_2123)} & {mark(bool(x.survive_ordinal))} " + r"\\")
            if o == "diabetes":
                rows.append(r"\addlinespace")
        write("menthlth_ordinal.tex", rows)


# ---------------------------------------------------------------- graphical abstract
GA_ROWS = [("diabetes", "SHAP", "logistic_regression", "BMI", "Diabetes · BMI"),
           ("diabetes", "SHAP", "logistic_regression", "HighBP", "Diabetes · high BP (self)"),
           ("hypertension", "SHAP", "logistic_regression", "Age", "Hypertension · age"),
           ("hypertension", "SHAP", "logistic_regression", "HighChol", "Hypertension · high chol. (self)"),
           ("hypertension", "SHAP", "logistic_regression", "Diabetes_self", "Hypertension · diabetes (self)")]
GA_METRICS = [("J_A", "Group-A Jaccard"), ("OC_A", "Group-A overlap"), ("top5", "Top-5 overlap"),
              ("top10", "Top-10 overlap"), ("spearman", "Spearman ρ"), ("weighted_tau", "Weighted τ")]


def graphical_abstract(ms: dict) -> None:
    """Three panels, 2.5:1, readable at 13 x 5.3 cm (JBI: >= 1328 x 531 px)."""
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
    CYC = [("2017-2020", "2017–20", "#2a6f97", "o"), ("2021-2023", "2021–23", "#c8553d", "o"),
           ("2015-2016", "2015–16 held out", "#1b9e77", "D")]
    rc = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7.5,
          "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
          "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
        # 1 example shifts, measured minus diagnosis (pp), three cycles
        ax = axes[0]
        for i, (o, me, mo, f, lab) in enumerate(GA_ROWS):
            r = ms[o].set_index(K).loc[(me, mo, f)]
            h = holdout(o).set_index(K).loc[(me, mo, f)]
            vals = [-r.diff_v1_1720, -r.diff_v1_2123, -h["diff"]]
            y = len(GA_ROWS) - 1 - i
            ax.plot([min(vals) * 100, max(vals) * 100], [y, y], color=GRID, lw=2.0, zorder=1)
            for (cyc, name, col, mk), v, dy in zip(CYC, vals, [0.2, 0.0, -0.2]):
                ax.scatter([v * 100], [y + dy], s=22 if mk == "o" else 18, marker=mk, color=col,
                           edgecolor="white", linewidth=0.6, zorder=3, label=name if i == 0 else None)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(range(len(GA_ROWS)), [r[4] for r in GA_ROWS][::-1], fontsize=6.6)
        ax.set_xlabel("SHAP share, measured − diagnosis label (pp)\nlogistic regression, same adults and splits",
                      fontsize=6.6)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title("1  The outcome definition moves\n    the explanation", loc="left", color=INK)
        ax.legend(loc="upper center", ncol=3, fontsize=5.8, frameon=False, handletextpad=0.1, columnspacing=0.8,
                  borderaxespad=0.1)
        ax.set_ylim(-0.6, len(GA_ROWS) + 0.15)
        # 2 what survives
        ax = axes[1]
        stages = ["Replicated in\n2 cycles", "+ prevalence\nmatched", "+ beyond AUC-\nmatched null",
                  "Held-out 2015–16:\nreplicated & beyond"]
        cnt = {}
        for o in OUTCOMES:
            m = ms[o]
            h = holdout(o).merge(m[K + ["core"]], on=K)
            hc = h[h.core]
            cnt[o] = [len(m), int(m[f"survive_{PM[o]}"].sum()), int(m.core.sum()),
                      int((hc.replicated & hc.beyond).sum())]
        assert cnt == {"diabetes": [13, 11, 5, 5], "hypertension": [34, 26, 18, 15]}, cnt
        y = np.arange(len(stages))[::-1]
        for k, (o, col) in enumerate([("diabetes", "#2a6f97"), ("hypertension", "#c8553d")]):
            ax.barh(y + (0.19 if k == 0 else -0.19), cnt[o], height=0.36, color=col,
                    label=o.capitalize())
            for yy, v in zip(y, cnt[o]):
                txt = f"{v}  {o}" if yy == y[0] else str(v)
                ax.text(v + 0.6, yy + (0.19 if k == 0 else -0.19), txt, va="center", fontsize=6.4,
                        color=col if yy == y[0] else INK)
        ax.set_yticks(y, stages, fontsize=6.4)
        ax.set_xlim(0, 58); ax.set_xticks([0, 10, 20, 30, 40]); ax.set_xlabel("Number of shifts", fontsize=6.6)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title("2  Core shifts survive controls,\n    nulls and a held-out cycle", loc="left", color=INK)
        # 3 detection by agreement metrics (diabetes)
        ax = axes[2]
        d = pd.read_csv(OUT / "detection_power" / "diabetes" / "detection_config.csv")
        d = d[d.positive]
        n_cfg = d.drop_duplicates(["cycle", "method", "model"]).shape[0]
        v = [int(d[d.metric == k].flagged.sum()) for k, _ in GA_METRICS]
        assert n_cfg == 8 and v[0] == 1 and v[2] == 0 and v[-1] == 7, (n_cfg, v)
        yy = np.arange(len(v))[::-1]
        ax.barh(yy, v, height=0.6, color=["#b9b7b0" if x <= 1 else "#2a6f97" for x in v])
        for a, b in zip(yy, v):
            ax.text(b + 0.12, a, f"{b}/{n_cfg}", va="center", fontsize=6.4, color=INK)
        ax.set_yticks(yy, [lab for _, lab in GA_METRICS], fontsize=6.4)
        ax.set_xlim(0, 8.8); ax.set_xticks(range(0, 9, 2))
        ax.set_xlabel("Configurations flagged, of 8\n(diabetes, with replicated shifts)", fontsize=6.6)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title("3  Overlap and top-K\n    metrics miss it", loc="left", color=INK)
        fig.tight_layout(w_pad=1.2)
        DF.mkdir(parents=True, exist_ok=True)
        fig.savefig(DF / "graphical_abstract.pdf", metadata={"CreationDate": None}, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(DF / "graphical_abstract.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(DF / "graphical_abstract.tiff", dpi=300, bbox_inches="tight", pad_inches=0.06, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)


# ---------------------------------------------------------------- text numbers
def text_numbers(ms: dict) -> None:
    for o in OUTCOMES:
        m = ms[o]
        put(f"{o}_n_rep", str(len(m)))
        put(f"{o}_n_core", str(int(m.core.sum())))
        for _, r in m.iterrows():
            for cyc in ["1720", "2123"]:
                N[f"{o}_{r.method}_{r.model}_{r.feature}_{cyc}_v1"] = pp(r[f"diff_v1_{cyc}"])
                N[f"{o}_{r.method}_{r.model}_{r.feature}_{cyc}_obs5"] = pp(r[f"auc_diff_obs5_{cyc}"])
                N[f"{o}_{r.method}_{r.model}_{r.feature}_{cyc}_nullm"] = pp(r[f"auc_null_mean_{cyc}"])
                N[f"{o}_{r.method}_{r.model}_{r.feature}_{cyc}_rndm"] = pp(r[f"rnd_null_mean_{cyc}"])
                N[f"{o}_{r.method}_{r.model}_{r.feature}_{cyc}_pdir"] = t4(r[f"auc_p_dir_{cyc}"])
                for c in [x for x in m.columns if x.startswith("diff_") and x.endswith(cyc)]:
                    N[f"{o}_{r.method}_{r.model}_{r.feature}_{c}"] = pp(r[c])
    from src.analysis.run_label_null import GAMMA_USED
    for o in OUTCOMES:
        put(f"gamma_{o}", f"{GAMMA_USED[o]:.2f}")
    put("min_p_dir_R30", t4(1 / 31))
    put("min_p_dir_R50", t4(1 / 51))
    # discordance composition and treatment
    for o in OUTCOMES:
        for cyc in C.CYCLES:
            df = C.load_base(o, cyc)
            s = SECOND[o]
            put(f"n10_{o}_{cyc}", str(int(((df.diag == 1) & (df[s] == 0)).sum())))
            put(f"n01_{o}_{cyc}", str(int(((df.diag == 0) & (df[s] == 1)).sum())))
            if o == "diabetes":
                d = df[(df.diag == 1) & (df.lab == 0)]
                put(f"treated_share_{cyc}", pct((d.Diabetes_treated == 1).mean()))
                for lab in ["lab63", "labM", "labG", "labTx"]:
                    put(f"prev_{lab}_{cyc}", t4(df[lab].mean()))
            else:
                for lab in ["meas130", "measM"]:
                    put(f"prev_{lab}_{cyc}", t4(df[lab].mean()))


def main() -> None:
    ms = {o: master(o) for o in OUTCOMES}
    for o in OUTCOMES:
        ms[o].to_csv(OUT / "label_null" / f"master_{o}.csv", index=False)
    table_cohorts()
    table_robustness(ms)
    table_core(ms)
    holdout_tables(ms)
    table_detection()
    table_shift_controls(ms)
    table_shift_null(ms)
    table_calibration()
    table_validation()
    compensation_numbers()
    reviewer_tables()
    table_cabc_rule()
    coding_numbers()
    figure(ms)
    graphical_abstract(ms)
    text_numbers(ms)
    DN.mkdir(parents=True, exist_ok=True)
    (DN / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False))
    print(f"{len(N)} numbers; tables in {DT}; figure in {DF}")


if __name__ == "__main__":
    main()
