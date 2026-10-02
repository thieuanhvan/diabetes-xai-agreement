"""
Sensitivity checks for the statistical choices of the journal analyses.

1. Variance decomposition
   a. seed as a crossed factor (seed main effect separated from the residual),
   b. centred log-ratio (CLR) transform of attribution shares (compositional data),
   c. distribution of per-feature eta^2 (heterogeneity behind the pooled value),
   d. permutation test of each pooled main effect: whole attribution vectors are
      shuffled across the levels of one factor within strata of the other factors
      and the seed, which keeps the within-vector dependence intact.
2. Seed reference distribution: share of label pairs below the 1st, 5th and
   10th percentile of the seed pairs.
3. Feature-level label shifts: exact paired sign-flip test over seeds as an
   alternative to the Nadeau-Bengio corrected t-test, with the same
   Benjamini-Hochberg families.

Outputs: outputs/sensitivity/
Usage:
    python -m src.analysis.run_sensitivity_checks [--n-perm 999]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from src.analysis.run_variance_decomposition import balanced_anova

OUT = Path(__file__).resolve().parents[2] / "outputs"
DST = OUT / "sensitivity"
PSEUDO = 1e-4  # added to every share before the CLR transform (PI clipped at 0)


def _response(v: pd.DataFrame, keys: list[str], kind: str) -> pd.Series:
    imp = v["importance"].clip(lower=0.0)
    share = imp / imp.groupby([v[k] for k in keys]).transform("sum")
    if kind == "share":
        return share
    if kind == "rank":
        return imp.groupby([v[k] for k in keys]).rank(ascending=False, method="average")
    if kind == "clr":
        lg = np.log(share + PSEUDO)
        return lg - lg.groupby([v[k] for k in keys]).transform("mean")
    raise ValueError(kind)


def pooled_eta2(v: pd.DataFrame, factors: list[str], resp: str, per_feature: bool = False):
    """Sum of squares per source within each feature, pooled over features:
    eta2_s = sum_f SS_s,f / sum_f SS_total,f."""
    rows = []
    for feat, g in v.groupby("feature"):
        ss = balanced_anova(g, factors, resp)
        for s, x in ss.items():
            rows.append((feat, s, x))
    d = pd.DataFrame(rows, columns=["feature", "source", "ss"])
    tot = d[d.source == "total"].set_index("feature").ss
    d = d[d.source != "total"]
    pooled = d.groupby("source").ss.sum() / tot.sum()
    if not per_feature:
        return pooled
    d["eta2"] = d.ss / d.feature.map(tot)
    return pooled, d


def variance_checks(n_perm: int, rng: np.random.Generator) -> None:
    designs = []
    nh = pd.read_csv(OUT / "label_axis" / "vectors.csv")
    nh = nh[nh.weighting == "unweighted"]
    for cyc, g in nh.groupby("cycle"):
        designs.append((f"nhanes_{cyc}", g.copy(), ["model", "method", "label"]))
    for cw in ["pipeline", "none", "balanced"]:
        b = pd.read_csv(OUT / "brfss_multiseed" / cw / "vectors.csv", dtype={"year": str})
        designs.append((f"brfss_{cw}", b, ["model", "method", "year"]))

    main_rows, feat_rows, perm_rows = [], [], []
    for name, v, factors in designs:
        keys = factors + ["seed"]
        for kind in ["share", "rank", "clr"]:
            v["y"] = _response(v, keys, kind)
            base, by_feat = pooled_eta2(v, factors, "y", per_feature=True)
            crossed = pooled_eta2(v.assign(seed=v.seed.astype(str)), factors + ["seed"], "y")
            for s, x in base.items():
                main_rows.append(dict(design=name, response=kind, model_spec="seed as replicate", source=s, eta2=x))
            seed_inter = sum(x for s, x in crossed.items() if "seed" in s and s != "seed")
            for s, x in crossed.items():
                if "seed" in s and s != "seed":
                    continue
                main_rows.append(dict(design=name, response=kind, model_spec="seed crossed", source=s, eta2=x))
            main_rows.append(dict(design=name, response=kind, model_spec="seed crossed",
                                  source="seed x factors", eta2=seed_inter))
            for s, g in by_feat.groupby("source"):
                feat_rows.append(dict(design=name, response=kind, source=s, median=g.eta2.median(),
                                      q25=g.eta2.quantile(0.25), q75=g.eta2.quantile(0.75),
                                      pooled=base[s]))
            if kind == "clr":
                continue
            # Permutation test of each main effect: shuffle whole vectors across the
            # levels of one factor within every stratum of the other factors x seed.
            # In a balanced design the main-effect SS equals its marginal SS, and the
            # total SS is invariant under this shuffle, so only the factor means move.
            wide = v.pivot_table(index=keys, columns="feature", values="y").reset_index()
            Y = wide[[c for c in wide.columns if c not in keys]].to_numpy()
            grand = Y.mean(axis=0)
            total = float(((Y - grand) ** 2).sum())
            for fac in factors:
                others = [k for k in keys if k != fac]
                codes = pd.factorize(wide[fac])[0]
                strata = wide.groupby(others).indices.values()

                def ss_main(order: np.ndarray) -> float:
                    Yp = Y[order]
                    ss = 0.0
                    for c in np.unique(codes):
                        m = codes == c
                        ss += m.sum() * float(((Yp[m].mean(axis=0) - grand) ** 2).sum())
                    return ss / total

                obs = ss_main(np.arange(len(Y)))
                assert abs(obs - base[fac]) < 1e-9, (name, kind, fac, obs, base[fac])
                count = 0
                for _ in range(n_perm):
                    order = np.arange(len(Y))
                    for idx in strata:
                        order[idx] = rng.permutation(idx)
                    count += int(ss_main(order) >= obs - 1e-12)
                pval = (count + 1) / (n_perm + 1)
                perm_rows.append(dict(design=name, response=kind, factor=fac, eta2=obs, n_perm=n_perm, p_perm=pval))
                logging.info("%s %s %s eta2=%.4f p=%.4f", name, kind, fac, obs, pval)
    pd.DataFrame(main_rows).to_csv(DST / "variance_specs.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(DST / "variance_per_feature.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(DST / "variance_permutation.csv", index=False)


def seed_floor_checks() -> None:
    p = pd.read_csv(OUT / "label_axis" / "pairs.csv")
    rows = []
    for q in [0.01, 0.05, 0.10]:
        for metric in ["J_A", "spearman", "rbo_ext_p90"]:
            for (cyc, w, meth, mdl), seed_g in p[p.axis == "seed"].groupby(["cycle", "weighting", "method", "model"]):
                lab = p[(p.axis == "label") & (p.cycle == cyc) & (p.weighting == w)
                        & (p.method == meth) & (p.model == mdl)]
                floor = seed_g[metric].quantile(q)
                rows.append(dict(percentile=q, metric=metric, cycle=cyc, weighting=w, method=meth, model=mdl,
                                 share_below=float((lab[metric] < floor).mean())))
    d = pd.DataFrame(rows)
    d.to_csv(DST / "seed_floor_percentiles.csv", index=False)
    s = d[d.weighting == "unweighted"].groupby(["percentile", "metric", "cycle"]).share_below.agg(["mean", "max"])
    s.to_csv(DST / "seed_floor_percentiles_summary.csv")
    logging.info("Seed reference distribution, label pairs below percentile:\n%s", s.round(4).to_string())


def sign_flip_checks() -> None:
    v = pd.read_csv(OUT / "label_axis" / "vectors.csv")
    v["importance"] = v["importance"].clip(lower=0.0)
    keys = ["cycle", "weighting", "method", "model", "label", "seed"]
    v["share"] = v["importance"] / v.groupby(keys)["importance"].transform("sum")
    wide = v.pivot_table(index=["cycle", "weighting", "method", "model", "seed", "feature"],
                         columns="label", values="share").reset_index()
    wide["d"] = wide["diag"] - wide["lab"]
    rows = []
    for key, g in wide.groupby(["cycle", "weighting", "method", "model", "feature"]):
        d = g["d"].to_numpy()
        signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
        null = np.abs((signs * d).mean(axis=1))
        p = float((null >= abs(d.mean()) - 1e-15).mean())
        rows.append(dict(zip(["cycle", "weighting", "method", "model", "feature"], key), diff=d.mean(), p_signflip=p))
    out = pd.DataFrame(rows)
    out["q_signflip"] = np.nan
    for _, idx in out.groupby(["cycle", "weighting", "method"]).groups.items():
        out.loc[idx, "q_signflip"] = false_discovery_control(out.loc[idx, "p_signflip"].to_numpy(), method="bh")
    nb = pd.read_csv(OUT / "label_axis" / "feature_shift.csv")
    nb = nb[(nb.label_1 == "diag") & (nb.label_2 == "lab")][["cycle", "weighting", "method", "model", "feature", "q_bh"]]
    out = out.merge(nb, on=["cycle", "weighting", "method", "model", "feature"])
    out.to_csv(DST / "label_shift_signflip.csv", index=False)
    u = out[out.weighting == "unweighted"]
    for test in ["q_bh", "q_signflip"]:
        sig = u[u[test] < 0.05].pivot_table(index=["method", "model", "feature"], columns="cycle", values="diff").dropna()
        rep = sig[np.sign(sig["2017-2020"]) == np.sign(sig["2021-2023"])]
        logging.info("%s: significant per cycle %s; replicated %d", test,
                     u[u[test] < 0.05].groupby("cycle").size().to_dict(), len(rep))
        rep.reset_index().to_csv(DST / f"label_shift_replicated_{test}.csv", index=False)


def patient_direction_checks() -> None:
    """Patient-level agreement with signed SHAP (direction kept) next to |SHAP|,
    XGBoost vs LR on the stored BRFSS test patients."""
    from scipy.stats import spearmanr
    rows = []
    for year in [2015, 2021, 2023]:
        d = OUT / "boundary_analysis" / "shap_per_patient" / f"cdc_brfss_diabetes_{year}"
        A = np.load(d / "xgboost_shap_values.npy").astype(float)
        B = np.load(d / "logistic_regression_shap_values.npy").astype(float)
        rho_abs = np.array([spearmanr(np.abs(a), np.abs(b))[0] for a, b in zip(A, B)])
        rho_sign = np.array([spearmanr(a, b)[0] for a, b in zip(A, B)])
        agree = []
        for a, b in zip(A, B):
            top = set(np.argsort(-np.abs(a))[:5]) | set(np.argsort(-np.abs(b))[:5])
            idx = np.array(sorted(top))
            agree.append(float((np.sign(a[idx]) == np.sign(b[idx])).mean()))
        agree = np.array(agree)
        rows.append(dict(year=year, n=len(A), median_spearman_abs=np.median(rho_abs),
                         median_spearman_signed=np.median(rho_sign),
                         median_sign_agreement_top5=np.median(agree),
                         share_patients_any_sign_conflict_top5=float((agree < 1).mean())))
    out = pd.DataFrame(rows)
    out.to_csv(DST / "patient_direction.csv", index=False)
    logging.info("Patient-level agreement, signed vs absolute SHAP:\n%s", out.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--skip-permutation", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    DST.mkdir(parents=True, exist_ok=True)
    seed_floor_checks()
    patient_direction_checks()
    sign_flip_checks()
    variance_checks(0 if args.skip_permutation else args.n_perm, np.random.default_rng(20261002))


if __name__ == "__main__":
    main()
