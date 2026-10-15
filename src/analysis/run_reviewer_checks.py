"""
Additional checks of the null and the detection analysis (v32).

    auc_match        AUC of the measured label and of the AUC-matched null labels
                     per cycle x model (mean, SD and range across null labels)
    calib_disjoint   gamma calibration repeated on a separate set of random
                     splits of the same participants (seeds 100-104; not
                     participant-disjoint; name kept for output compatibility)
    gamma_sens       core shifts under neighbouring gamma values (if run)
    fp_feature       feature-level false-positive rate of the pseudo-outcome check
    detection_alt    detection with the seed reference built from the diagnosis
                     and measured labels only, and with positives defined by core
                     shifts
    clr              core shifts re-tested on centred log-ratio shares

Outputs: outputs/reviewer_checks/
Usage:
    python -m src.analysis.run_reviewer_checks
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from src.analysis import label_controls_core as C
from src.analysis import run_label_null as NL
from src.analysis.run_detection_power import flag, k_needed, load, metrics, seed_floor

DST = C.OUT_DIR / "reviewer_checks"
OUTS = ["diabetes", "hypertension"]
GAMMA_SENS = {"diabetes": [0.4], "hypertension": [0.25]}  # one grid step away; 30 null labels each


def master(o):
    return pd.read_csv(C.OUT_DIR / "label_null" / f"master_{o}.csv")


def auc_match() -> pd.DataFrame:
    rows = []
    for o in OUTS:
        v = pd.read_csv(C.OUT_DIR / "label_null" / f"{o}_risk" / "vectors.csv")
        a = v.drop_duplicates(["cycle", "label", "model", "seed"])
        for (cyc, model), g in a.groupby(["cycle", "model"]):
            real = g[g.label == C.SECOND[o]].auc.mean()
            diag = g[g.label == "diag"].auc.mean()
            nl = g[g.label.str.startswith("null")].groupby("label").auc.mean()
            rows.append(dict(outcome=o, cycle=cyc, model=model, diag=diag, measured=real,
                             null_mean=nl.mean(), null_sd=nl.std(), null_min=nl.min(), null_max=nl.max(),
                             diff=nl.mean() - real))
    out = pd.DataFrame(rows)
    out.to_csv(DST / "auc_match.csv", index=False)
    logging.info("AUC match:\n%s", out.round(4).to_string(index=False))
    return out


def calib_disjoint() -> None:
    old = NL.SEEDS
    NL.SEEDS = list(range(100, 105))
    for o in OUTS:
        s = NL.calibrate(o)
        s.to_csv(DST / f"calibration_disjoint_{o}.csv")
    NL.SEEDS = old
    # calibrate() also overwrote label_null/calibration_<o>.csv; restore with seeds 0-4
    for o in OUTS:
        NL.calibrate(o)


def gamma_sens() -> pd.DataFrame:
    rows = []
    for o in OUTS:
        m = master(o)
        base = m[["method", "model", "feature", "prev", "core"]]
        for g in GAMMA_SENS[o]:
            d = C.OUT_DIR / "label_null" / f"{o}_risk_g{g:g}" / "survival.csv"
            if not d.exists():
                continue
            s = pd.read_csv(d)[["method", "model", "feature", "beyond_null_dir_both"]]
            x = base.merge(s, on=["method", "model", "feature"])
            core_g = x.prev & x.beyond_null_dir_both
            rows.append(dict(outcome=o, gamma=g, beyond=int(x.beyond_null_dir_both.sum()),
                             core=int(core_g.sum()), core_kept=int((core_g & x.core).sum()),
                             core_main=int(x.core.sum())))
            auc = pd.read_csv(C.OUT_DIR / "label_null" / f"{o}_risk_g{g:g}" / "auc_by_label_kind.csv")
            rows[-1].update({f"auc_null_{c}": v for c, v in auc.set_index("kind").loc["null_labels"].items()})
    out = pd.DataFrame(rows)
    out.to_csv(DST / "gamma_sensitivity.csv", index=False)
    logging.info("gamma sensitivity:\n%s", out.to_string(index=False))
    return out


CYCLE_GAMMA = {"diabetes": {"2017-2020": 0.3, "2021-2023": 0.4},
               "hypertension": {"2017-2020": 0.35, "2021-2023": 0.25}}


def cycle_matched() -> pd.DataFrame:
    """Null with a cycle-specific gamma chosen by the calibration tables (closest
    LR and XGBoost AUC to the measured label in each cycle)."""
    rows, detail = [], []
    for o in OUTS:
        m = master(o)
        draws = {}
        for cyc, g in CYCLE_GAMMA[o].items():
            sub = f"{o}_risk" if abs(g - NL.GAMMA_USED[o]) < 1e-9 else f"{o}_risk_g{g:g}"
            f = C.OUT_DIR / "label_null" / sub / "null_distribution.csv"
            if not f.exists():
                break
            nd = pd.read_csv(f)
            draws[cyc] = nd[nd.cycle == cyc]
        if len(draws) < 2:
            continue
        for r in m.itertuples():
            ok = True
            ps = {}
            for cyc, key in zip(C.CYCLES, ["1720", "2123"]):
                obs = getattr(r, f"auc_diff_obs5_{key}")
                x = draws[cyc]
                x = x[(x.method == r.method) & (x.model == r.model) & (x.feature == r.feature)]["diff"].to_numpy()
                p = (1 + ((x <= obs) if obs < 0 else (x >= obs)).sum()) / (len(x) + 1)
                ps[cyc] = p
                ok &= p < 0.05
            detail.append(dict(outcome=o, method=r.method, model=r.model, feature=r.feature,
                               p_1720=ps[C.CYCLES[0]], p_2123=ps[C.CYCLES[1]], beyond=ok,
                               core_cm=bool(ok and r.prev), core_main=bool(r.core)))
        dd = pd.DataFrame([x for x in detail if x["outcome"] == o])
        rows.append(dict(outcome=o, beyond=int(dd.beyond.sum()), core=int(dd.core_cm.sum()),
                         core_kept=int((dd.core_cm & dd.core_main).sum()), core_main=int(dd.core_main.sum())))
    out = pd.DataFrame(rows)
    out.to_csv(DST / "cycle_matched.csv", index=False)
    pd.DataFrame(detail).to_csv(DST / "cycle_matched_detail.csv", index=False)
    logging.info("cycle-matched gamma:\n%s", out.to_string(index=False))
    return out


def fp_feature() -> pd.DataFrame:
    rows = []
    for o in OUTS:
        d = pd.read_csv(C.OUT_DIR / "null_validation" / o / "false_positives_detail.csv")
        fp = pd.read_csv(C.OUT_DIR / "null_validation" / o / "false_positives.csv")
        n_tests = 14 * 3 * 2
        rows.append(dict(outcome=o, null_labels=len(fp), tests_per_label=n_tests,
                         replicated_vs_zero=int(fp.replicated_vs_zero.sum()),
                         flagged=int(d.beyond.sum()),
                         rate_vs_zero=fp.replicated_vs_zero.sum() / (len(fp) * n_tests),
                         rate_full=d.beyond.sum() / (len(fp) * n_tests)))
    out = pd.DataFrame(rows)
    out.to_csv(DST / "fp_feature.csv", index=False)
    logging.info("feature-level false positives:\n%s", out.to_string(index=False))
    return out


METRICS = ["J_A", "OC_A", "size_ratio_A", "top5", "top10", "feature_agreement", "contradiction_A",
           "rbo_ext_p90", "spearman", "kendall_tau_b", "pairwise_rank_agreement", "weighted_tau"]


def detection_alt() -> pd.DataFrame:
    rows = []
    for o in OUTS:
        vec, seeds = load(o)
        second = C.SECOND[o]
        k = k_needed(len(seeds))
        m = master(o)
        pos_rep = set(map(tuple, m[["method", "model"]].to_numpy()))
        pos_core = set(map(tuple, m[m.core][["method", "model"]].to_numpy()))
        for ref_name, labels in [("diag+measured", ["diag", second]), ("diag+measured+composite", ["diag", second, "total"])]:
            for cfg in sorted({key[:3] for key in vec}):
                cyc, method, model = cfg
                floor = seed_floor(vec, seeds, cfg, labels)
                pairs = pd.DataFrame([metrics(vec[cfg + ("diag", s)], vec[cfg + (second, s)]) for s in seeds])
                for met, (_, _, fl) in flag(pairs, floor, k).items():
                    rows.append(dict(outcome=o, reference=ref_name, cycle=cyc, method=method, model=model,
                                     metric=met, flagged=bool(fl),
                                     positive_replicated=(method, model) in pos_rep,
                                     positive_core=(method, model) in pos_core))
    d = pd.DataFrame(rows)
    d.to_csv(DST / "detection_alt_detail.csv", index=False)
    summ = []
    for (o, ref), g in d.groupby(["outcome", "reference"]):
        for met in METRICS:
            gm = g[g.metric == met]
            summ.append(dict(outcome=o, reference=ref, metric=met,
                             rep_pos=f"{int(gm[gm.positive_replicated].flagged.sum())}/{int(gm.positive_replicated.sum())}",
                             rep_neg=f"{int(gm[~gm.positive_replicated].flagged.sum())}/{int((~gm.positive_replicated).sum())}",
                             core_pos=f"{int(gm[gm.positive_core].flagged.sum())}/{int(gm.positive_core.sum())}",
                             core_neg=f"{int(gm[~gm.positive_core].flagged.sum())}/{int((~gm.positive_core).sum())}"))
    s = pd.DataFrame(summ)
    s.to_csv(DST / "detection_alt.csv", index=False)
    logging.info("detection variants:\n%s", s.to_string(index=False))
    return s


CLR_PSEUDO = [1e-5, 1e-4, 1e-3]


def clr() -> pd.DataFrame:
    """Core shifts re-tested on centred log-ratio shares; main pseudocount 1e-4,
    sensitivity 1e-5 and 1e-3 (clr_pseudocount.csv)."""
    outs = {pc: _clr(pc) for pc in CLR_PSEUDO}
    out = outs[1e-4]
    out.to_csv(DST / "clr_core.csv", index=False)
    wide = out[["outcome", "method", "model", "feature", "sign"]].copy()
    for pc, o in outs.items():
        wide[f"clr_replicated_{pc:g}"] = o.clr_replicated.to_numpy()
    wide.to_csv(DST / "clr_pseudocount.csv", index=False)
    logging.info("CLR re-test of core shifts by pseudocount:\n%s",
                 wide.groupby("outcome")[[c for c in wide if c.startswith("clr_")]].sum())
    return out


def _clr(pc: float) -> pd.DataFrame:
    from scipy.stats import false_discovery_control
    from scipy.stats import t as t_dist
    rows = []
    for o in OUTS:
        v = C.v1_vectors(o, labels=("diag", C.SECOND[o]))
        v["importance"] = v.importance.clip(lower=0)
        keys = ["cycle", "method", "model", "label", "seed"]
        v["share"] = v.importance / v.groupby(keys).importance.transform("sum")
        v["lg"] = np.log(v.share + pc)
        v["clr"] = v.lg - v.groupby(keys).lg.transform("mean")
        w = v.pivot_table(index=["cycle", "method", "model", "seed", "feature"], columns="label",
                          values="clr").reset_index()
        w["d"] = w["diag"] - w[C.SECOND[o]]
        ratio = (v.n_test / v.n_train).median()
        res = []
        for key, g in w.groupby(["cycle", "method", "model", "feature"]):
            d = g.d.to_numpy(); J = len(d); var = d.var(ddof=1)
            se = np.sqrt((1 / J + ratio) * var) if var > 0 else np.nan
            tv = d.mean() / se if se and se > 0 else np.nan
            p = 2 * t_dist.sf(abs(tv), J - 1) if np.isfinite(tv) else np.nan
            res.append(dict(zip(["cycle", "method", "model", "feature"], key), diff=d.mean(), p=p))
        res = pd.DataFrame(res)
        res["q"] = np.nan
        for _, idx in res.groupby(["cycle", "method"]).groups.items():
            pp = res.loc[idx, "p"]; ok = pp.notna()
            res.loc[pp[ok].index, "q"] = false_discovery_control(pp[ok].to_numpy(), method="bh")
        m = master(o)
        for r in m[m.core].itertuples():
            g = res[(res.method == r.method) & (res.model == r.model) & (res.feature == r.feature)].set_index("cycle")
            ok = all(g.loc[c, "q"] < 0.05 and np.sign(g.loc[c, "diff"]) == r.sign for c in C.CYCLES)
            rows.append(dict(outcome=o, method=r.method, model=r.model, feature=r.feature, sign=r.sign,
                             clr_replicated=bool(ok)))
    return pd.DataFrame(rows)


def cabc_std() -> pd.DataFrame:
    """Set metrics recomputed with the published cABC A|B limit (Ultsch and
    Lotsch 2015: the smaller-effort of the Pareto and BreakEven points; see
    agreement_metrics.cabc_group_a_standard) instead of the BreakEven-only
    (above-mean) rule used in the conference pipeline. Re-runs the detection
    analysis (B1-B3) into reviewer_checks/cabc_standard/<outcome>/ and recounts
    the 45 conference pairs."""
    from src.evaluation import agreement_metrics as AM
    from src.analysis import run_detection_power as DP
    from src.analysis import run_metric_comparison as MC
    orig = AM.cabc_partition

    def std_partition(importance):
        a = AM.cabc_group_a_standard(importance)
        rest = [f for f in importance.abs().sort_values(ascending=False).index if f not in a]
        return AM.CABCPartition(a, rest, [], float(importance.abs().mean()))

    AM.cabc_partition = std_partition
    DP.cabc_partition = std_partition
    try:
        for o in OUTS:
            DP.run(o, dst=DST / "cabc_standard" / o)
        pairs = MC.build_pairs(MC.load_all())
    finally:
        AM.cabc_partition = orig
        DP.cabc_partition = orig
    pairs.to_csv(DST / "cabc_standard" / "conference_pairs.csv", index=False)
    rows = []
    for o in OUTS:
        vec, _ = DP.load(o)
        same = sum(set(orig(v).group_a) == set(AM.cabc_group_a_standard(v)) for v in vec.values())
        rows.append(dict(set=f"NHANES {o} label vectors", n=len(vec), identical_group_a=same))
    v = MC.load_all()
    rows.append(dict(set="conference vectors", n=len(v),
                     identical_group_a=sum(set(orig(s).group_a) == set(AM.cabc_group_a_standard(s)) for s in v.values())))
    nested = int(((pairs.contradiction_A == 0) & (pairs.J_A < 1)).sum())
    rows.append(dict(set="conference pairs: J_A<1 / nested / contradiction", n=len(pairs),
                     identical_group_a=f"{int((pairs.J_A < 1).sum())}/{nested}/{int(pairs.contradiction_A.sum())}"))
    out = pd.DataFrame(rows)
    out.to_csv(DST / "cabc_standard" / "group_a_agreement.csv", index=False)
    logging.info("standard cABC:\n%s", out.to_string(index=False))
    return out


MENT_ORDINAL = {0: 0, 5: 1, 10: 2, 20: 3, 30: 4}


def menthlth_ordinal() -> pd.DataFrame:
    """Predictor-coding sensitivity: MentHlth (PHQ-9 bands coded 0, 5, 10, 20,
    30 on the BRFSS days scale) recoded to equally spaced 0-4. Tree models
    split on order and are unaffected by a monotone recoding, so only LR is
    refitted; RF and XGBoost vectors are the stored ones, keeping the BH
    family of 42 tests per cycle x method. Survival of every replicated LR
    shift is reported."""
    orig = C.load_base

    def load_recoded(outcome, cycle):
        df = orig(outcome, cycle)
        df["MentHlth"] = df["MentHlth"].map(MENT_ORDINAL)
        assert df["MentHlth"].notna().all()
        return df

    rows = []
    C.load_base = load_recoded
    try:
        for o in OUTS:
            seeds = list(range(10)) if o == "diabetes" else list(range(5))
            f = DST / f"menthlth_ordinal_{o}.csv"
            new = C.fit_labels(o, ["diag", C.SECOND[o]], seeds, f, models=["logistic_regression"])
            old = C.v1_vectors(o, labels=("diag", C.SECOND[o]), seeds=seeds)
            v = pd.concat([old[old.model != "logistic_regression"], new[old.columns.intersection(new.columns)]])
            sh = C.shift_table(v, "diag", C.SECOND[o])
            base = C.v1_replicated(o)
            base = base[base.model == "logistic_regression"]
            out = C.survival(base, sh, "ordinal")
            m = master(o)[["method", "model", "feature", "core"]]
            out = out.merge(m, on=["method", "model", "feature"], how="left")
            rows.append(out.assign(outcome=o))
    finally:
        C.load_base = orig
    res = pd.concat(rows)
    res.to_csv(DST / "menthlth_ordinal.csv", index=False)
    logging.info("MentHlth 0-4 recoding, LR shifts surviving:\n%s",
                 res.groupby("outcome")[["survive_ordinal", "core"]].agg(["sum", "size"]))
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    DST.mkdir(parents=True, exist_ok=True)
    stages = sys.argv[1:] or ["auc_match", "fp_feature", "detection_alt", "clr", "calib_disjoint", "gamma_sens", "cycle_matched", "cabc_std", "menthlth_ordinal"]
    for s in stages:
        globals()[s]()


if __name__ == "__main__":
    main()
