"""
Comparison with the disagreement metrics of Krishna et al. (TMLR 2024), as
implemented in OpenXAI: feature agreement and rank agreement among the top-k
features, Spearman rank correlation and pairwise rank agreement over all
features. Sign agreement is omitted because population-level importances are
non-negative.

The metrics are computed on the same stored vectors as the audit, for three
cases in which the single-metric baseline reached a conclusion:
  A. the 45 pairs of the 18 conference vectors (depth versus contradiction);
  B. cross-model SHAP pairs in BRFSS with and without class weighting;
  C. NHANES label pairs against seed pairs (outcome definition).

Outputs: outputs/soa_comparison/
Usage:
    python -m src.analysis.run_soa_comparison
"""

from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.run_metric_comparison import load_all
from src.evaluation.agreement_metrics import align, cabc_partition, is_contradiction, jaccard

OUT = Path(__file__).resolve().parents[2] / "outputs"
DST = OUT / "soa_comparison"
K = 5


def krishna(a: pd.Series, b: pd.Series, k: int = K) -> dict:
    a, b = align(a, b)
    ra = list(a.sort_values(ascending=False).index)
    rb = list(b.sort_values(ascending=False).index)
    fa = len(set(ra[:k]) & set(rb[:k])) / k
    rank_ag = sum(1 for i in range(k) if ra[i] == rb[i] and ra[i] in rb[:k]) / k
    pos_a = {f: i for i, f in enumerate(ra)}
    pos_b = {f: i for i, f in enumerate(rb)}
    feats = list(a.index)
    pairs = list(combinations(feats, 2))
    pra = sum((pos_a[i] < pos_a[j]) == (pos_b[i] < pos_b[j]) for i, j in pairs) / len(pairs)
    pa, pb = cabc_partition(a), cabc_partition(b)
    return dict(feature_agreement=fa, rank_agreement=rank_ag,
                rank_correlation=float(spearmanr(a.values, b.values).statistic),
                pairwise_rank_agreement=pra, J_A=jaccard(pa.group_a, pb.group_a),
                contradiction_A=int(is_contradiction(pa.group_a, pb.group_a)))


METRICS = ["feature_agreement", "rank_agreement", "rank_correlation", "pairwise_rank_agreement", "J_A"]


def case_conference() -> pd.DataFrame:
    v = load_all()
    p = pd.read_csv(OUT / "metric_comparison" / "pairs.csv")
    rows = []
    for _, r in p.iterrows():
        k1 = tuple(r.run_1.split("/"))
        k2 = tuple(r.run_2.split("/"))
        rows.append(dict(axis=r.axis, pair=r.pair, **krishna(v[k1], v[k2])))
    return pd.DataFrame(rows)


def _vectors(path: Path, keys: list[str]) -> dict:
    d = pd.read_csv(path, dtype={"year": str})
    return {k: g.set_index("feature")["importance"].clip(lower=0.0) for k, g in d.groupby(keys)}


def case_weighting() -> pd.DataFrame:
    rows = []
    for cw in ["pipeline", "none"]:
        vec = _vectors(OUT / "brfss_multiseed" / cw / "vectors.csv", ["year", "seed", "method", "model"])
        models = sorted({k[3] for k in vec})
        for (year, seed) in sorted({k[:2] for k in vec}):
            for a, b in combinations(models, 2):
                rows.append(dict(weighting=cw, year=year, seed=seed, pair=f"{a}-{b}",
                                 **krishna(vec[(year, seed, "SHAP", a)], vec[(year, seed, "SHAP", b)])))
    return pd.DataFrame(rows)


def case_label() -> pd.DataFrame:
    d = pd.read_csv(OUT / "label_axis" / "vectors.csv")
    d = d[d.weighting == "unweighted"]
    vec = {k: g.set_index("feature")["importance"].clip(lower=0.0)
           for k, g in d.groupby(["cycle", "method", "model", "label", "seed"])}
    seeds = sorted(d.seed.unique())
    rows = []
    for cyc in sorted(d.cycle.unique()):
        for meth in ["SHAP", "PI"]:
            for mdl in sorted(d.model.unique()):
                for lab in ["diag", "lab", "total"]:
                    for s1, s2 in combinations(seeds, 2):
                        rows.append(dict(axis="seed", cycle=cyc, method=meth, model=mdl,
                                         **krishna(vec[(cyc, meth, mdl, lab, s1)], vec[(cyc, meth, mdl, lab, s2)])))
                for a, b in combinations(["diag", "lab", "total"], 2):
                    for s in seeds:
                        rows.append(dict(axis="label", cycle=cyc, method=meth, model=mdl,
                                         **krishna(vec[(cyc, meth, mdl, a, s)], vec[(cyc, meth, mdl, b, s)])))
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    DST.mkdir(parents=True, exist_ok=True)
    rows = []

    a = case_conference()
    a.to_csv(DST / "conference_pairs.csv", index=False)
    for m in METRICS:
        rows.append(dict(case="A. Conference vectors (45 pairs)", quantity=m, value=a[m].mean(),
                         pairs_below_1=int((a[m] < 1 - 1e-12).sum())))
    rows.append(dict(case="A. Conference vectors (45 pairs)", quantity="contradiction_A (triad)",
                     value=a.contradiction_A.mean(), pairs_below_1=int(a.contradiction_A.sum())))

    b = case_weighting()
    b.to_csv(DST / "weighting_pairs.csv", index=False)
    for cw, g in b.groupby("weighting"):
        for m in METRICS:
            rows.append(dict(case=f"B. BRFSS cross-model SHAP, weighting {cw}", quantity=m, value=g[m].mean(),
                             pairs_below_1=int((g[m] < 1 - 1e-12).sum())))

    c = case_label()
    c.to_csv(DST / "label_pairs.csv", index=False)
    for cyc, g in c.groupby("cycle"):
        for m in METRICS:
            lab, seed = g[g.axis == "label"], g[g.axis == "seed"]
            below = []
            for _, gg in g.groupby(["method", "model"]):
                floor = gg[gg.axis == "seed"][m].quantile(0.05)
                below.append((gg[gg.axis == "label"][m] < floor).mean())
            rows.append(dict(case=f"C. NHANES label pairs {cyc}", quantity=m, value=lab[m].mean(),
                             seed_value=seed[m].mean(), label_below_seed_p5=float(np.mean(below))))
    out = pd.DataFrame(rows)
    out.to_csv(DST / "summary.csv", index=False)
    logging.info("Comparison with Krishna et al. disagreement metrics:\n%s", out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
