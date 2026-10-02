"""
Checks of choices that can shape attribution comparisons independently of the
six audited sources: feature dependence, the RBO persistence parameter and the
handling of negative permutation importance.

1. Feature dependence
   a. Spearman correlation between predictors (NHANES analytic samples, BRFSS
      cohorts): pairs with |rho| >= 0.3.
   b. Domain-level label shifts: attribution shares summed within clinically
      defined predictor domains, then the same corrected resampled t-test,
      Benjamini-Hochberg family (cycle x weighting x method) and cross-cycle
      replication rule as the feature-level analysis. If feature-level shifts
      were only substitution between correlated predictors of one domain, they
      would cancel at the domain level.
2. RBO persistence p in {0.8, 0.9, 0.95}: share of NHANES label pairs below the
   5th percentile of the seed pairs, and BRFSS mean RBO by axis.
3. Negative permutation importance: replicated PI label shifts with negative
   values clipped at 0 (main analysis), with absolute values, and on the rank
   scale of the raw values.

Outputs: outputs/dependence/
Usage:
    python -m src.analysis.run_dependence_checks
"""

from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from scipy.stats import t as t_dist

from src.analysis.run_label_axis import CYCLES, FEATURES, load_cycle
from src.evaluation.agreement_metrics import compare

OUT = Path(__file__).resolve().parents[2] / "outputs"
DST = OUT / "dependence"
DATA = Path(__file__).resolve().parents[2] / "data"

DOMAINS = {
    "Self-reported comorbidity diagnoses": ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack"],
    "Socioeconomic and access": ["Education", "Income", "AnyHealthcare"],
    "Health status and activity": ["GenHlth", "MentHlth", "PhysActivity_LTPA"],
    "Body size": ["BMI"],
    "Smoking": ["Smoker"],
    "Demographic": ["Age", "Sex"],
}
BRFSS_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]


def correlations() -> None:
    rows = []
    for cyc in CYCLES:
        df = load_cycle(cyc)[FEATURES]
        rows.append(_pairs(df.corr(method="spearman"), f"NHANES {cyc}"))
    for year in [2015, 2021, 2023]:
        cols = pd.read_csv(DATA / f"cdc_brfss_diabetes_{year}.csv", nrows=1).columns
        feats = [f for f in BRFSS_FEATURES if f in cols]
        df = pd.read_csv(DATA / f"cdc_brfss_diabetes_{year}.csv", usecols=feats)
        rows.append(_pairs(df.corr(method="spearman"), f"BRFSS {year}"))
    out = pd.concat(rows)
    out.to_csv(DST / "predictor_correlation.csv", index=False)
    strong = out[out.rho.abs() >= 0.3].assign(abs_rho=lambda d: d.rho.abs())
    strong = strong.sort_values(["cohort", "abs_rho"], ascending=[True, False]).drop(columns="abs_rho")
    strong.to_csv(DST / "predictor_correlation_strong.csv", index=False)
    logging.info("Predictor pairs with |rho| >= 0.3:\n%s", strong.round(4).to_string(index=False))


def _pairs(c: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = [dict(cohort=cohort, feature_1=a, feature_2=b, rho=c.loc[a, b])
            for a, b in combinations(c.columns, 2)]
    return pd.DataFrame(rows)


def _shares(v: pd.DataFrame, value: str = "clip") -> pd.DataFrame:
    v = v.copy()
    if value == "clip":
        v["x"] = v["importance"].clip(lower=0.0)
    elif value == "abs":
        v["x"] = v["importance"].abs()
    else:
        raise ValueError(value)
    keys = ["cycle", "weighting", "method", "model", "label", "seed"]
    v["share"] = v["x"] / v.groupby(keys)["x"].transform("sum")
    return v


def _shift_tests(wide: pd.DataFrame, unit: str, ratio: float) -> pd.DataFrame:
    """Corrected resampled t-test of diag minus lab, BH within cycle x weighting x method."""
    rows = []
    for key, g in wide.groupby(["cycle", "weighting", "method", "model", unit]):
        d = (g["diag"] - g["lab"]).to_numpy()
        J = len(d)
        var = d.var(ddof=1)
        se = np.sqrt((1.0 / J + ratio) * var) if var > 0 else np.nan
        tval = d.mean() / se if se and se > 0 else np.nan
        p = 2 * t_dist.sf(abs(tval), J - 1) if np.isfinite(tval) else np.nan
        rows.append(dict(zip(["cycle", "weighting", "method", "model", unit], key), diff=d.mean(), p=p))
    out = pd.DataFrame(rows)
    out["q_bh"] = np.nan
    for _, idx in out.groupby(["cycle", "weighting", "method"]).groups.items():
        p = out.loc[idx, "p"]
        ok = p.notna()
        out.loc[p[ok].index, "q_bh"] = false_discovery_control(p[ok].to_numpy(), method="bh")
    return out


def _replicated(t: pd.DataFrame, unit: str) -> pd.DataFrame:
    u = t[t.weighting == "unweighted"]
    w = u.pivot_table(index=["method", "model", unit], columns="cycle", values=["diff", "q_bh"])
    ok = (w["q_bh"] < 0.05).all(axis=1) & (np.sign(w["diff"][CYCLES[0]]) == np.sign(w["diff"][CYCLES[1]]))
    rep = w[ok]["diff"].reset_index()
    rep.columns.name = None
    return rep


def domain_shifts(ratio: float) -> None:
    v = _shares(pd.read_csv(OUT / "label_axis" / "vectors.csv"))
    dom = {f: d for d, fs in DOMAINS.items() for f in fs}
    v["domain"] = v["feature"].map(dom)
    g = v.groupby(["cycle", "weighting", "method", "model", "label", "seed", "domain"]).share.sum().reset_index()
    wide = g.pivot_table(index=["cycle", "weighting", "method", "model", "seed", "domain"],
                         columns="label", values="share").reset_index()
    t = _shift_tests(wide, "domain", ratio)
    t.to_csv(DST / "domain_shift.csv", index=False)
    rep = _replicated(t, "domain")
    rep.to_csv(DST / "domain_shift_replicated.csv", index=False)
    logging.info("Domain-level shifts replicated in both cycles (diag minus lab share):\n%s",
                 rep.round(4).to_string(index=False))


def pi_handling(ratio: float) -> None:
    v = pd.read_csv(OUT / "label_axis" / "vectors.csv")
    v = v[v.method == "PI"]
    rows = []
    for value in ["clip", "abs"]:
        s = _shares(v, value)
        wide = s.pivot_table(index=["cycle", "weighting", "method", "model", "seed", "feature"],
                             columns="label", values="share").reset_index()
        rep = _replicated(_shift_tests(wide, "feature", ratio), "feature")
        rows.append(rep.assign(handling=f"share, negative PI {'clipped at 0' if value == 'clip' else 'as absolute value'}"))
    keys = ["cycle", "weighting", "method", "model", "label", "seed"]
    r = v.copy()
    r["rank"] = r.groupby(keys)["importance"].rank(ascending=False, method="average")
    # positive diff = higher importance (smaller rank) under diag, as for shares
    wide = r.pivot_table(index=["cycle", "weighting", "method", "model", "seed", "feature"],
                         columns="label", values="rank").reset_index()
    wide["diag"], wide["lab"] = -wide["diag"], -wide["lab"]
    rows.append(_replicated(_shift_tests(wide, "feature", ratio), "feature").assign(handling="rank of raw PI"))
    out = pd.concat(rows)
    out.to_csv(DST / "pi_negative_handling.csv", index=False)
    neg = (v.importance < 0).mean()
    pos = v.groupby(keys).importance.apply(lambda x: x.clip(lower=0).sum())
    negm = v.groupby(keys).importance.apply(lambda x: (-x.clip(upper=0)).sum())
    pd.DataFrame([dict(share_negative_values=neg, neg_mass_over_pos_mass_median=(negm / pos).median(),
                       neg_mass_over_pos_mass_max=(negm / pos).max())]).to_csv(DST / "pi_negative_summary.csv", index=False)
    logging.info("PI replicated label shifts by handling of negative values:\n%s", out.round(4).to_string(index=False))


def _rbo(a: pd.Series, b: pd.Series, p: float) -> float:
    """RBO_ext with the ranking rule of agreement_metrics.compare (same tie handling)."""
    return compare(a, b, rbo_ps=(p,), top_ks=())[f"rbo_ext_p{int(round(p * 100))}"]


def rbo_persistence() -> None:
    rows = []
    v = pd.read_csv(OUT / "label_axis" / "vectors.csv")
    v = v[v.weighting == "unweighted"]
    vec = {k: g.set_index("feature")["importance"].clip(lower=0.0)
           for k, g in v.groupby(["cycle", "method", "model", "label", "seed"])}
    seeds = sorted(v.seed.unique())
    for p in [0.8, 0.9, 0.95]:
        for cyc in CYCLES:
            for meth in ["SHAP", "PI"]:
                for mdl in sorted(v.model.unique()):
                    sp = [_rbo(vec[(cyc, meth, mdl, lab, s1)], vec[(cyc, meth, mdl, lab, s2)], p)
                          for lab in ["diag", "lab", "total"] for s1, s2 in combinations(seeds, 2)]
                    lp = [_rbo(vec[(cyc, meth, mdl, a, s)], vec[(cyc, meth, mdl, b, s)], p)
                          for a, b in combinations(["diag", "lab", "total"], 2) for s in seeds]
                    floor = np.quantile(sp, 0.05)
                    rows.append(dict(analysis="NHANES label pairs below seed p5", p=p, cycle=cyc, method=meth,
                                     model=mdl, value=float(np.mean(np.array(lp) < floor))))
    b = pd.read_csv(OUT / "brfss_multiseed" / "pipeline" / "vectors.csv", dtype={"year": str})
    bvec = {k: g.set_index("feature")["importance"].clip(lower=0.0)
            for k, g in b.groupby(["year", "seed", "method", "model"])}
    models = sorted(b.model.unique())
    for p in [0.8, 0.9, 0.95]:
        for (year, seed) in sorted({k[:2] for k in bvec}):
            for mdl in models:
                rows.append(dict(analysis="BRFSS mean RBO, method axis", p=p, cycle=year, method="", model=mdl,
                                 value=_rbo(bvec[(year, seed, "SHAP", mdl)], bvec[(year, seed, "PI", mdl)], p)))
            for meth in ["SHAP", "PI"]:
                for a, c in combinations(models, 2):
                    rows.append(dict(analysis="BRFSS mean RBO, model axis", p=p, cycle=year, method=meth, model="",
                                     value=_rbo(bvec[(year, seed, meth, a)], bvec[(year, seed, meth, c)], p)))
    out = pd.DataFrame(rows)
    out.to_csv(DST / "rbo_persistence.csv", index=False)
    summ = out.groupby(["analysis", "p", "cycle"]).value.mean().unstack("p")
    summ.to_csv(DST / "rbo_persistence_summary.csv")
    logging.info("RBO persistence sensitivity:\n%s", summ.round(4).to_string())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    DST.mkdir(parents=True, exist_ok=True)
    perf = pd.read_csv(OUT / "label_axis" / "performance.csv")
    ratio = float((perf["n_test"] / perf["n_train"]).median())
    correlations()
    domain_shifts(ratio)
    pi_handling(ratio)
    rbo_persistence()


if __name__ == "__main__":
    main()
