"""
Quantify the sources of variation in global attributions.

Response: the attribution SHARE of each feature (importance / sum of the
vector; PI clipped at 0), so that SHAP and PI are on the same scale.
For every feature, a balanced full-factorial ANOVA decomposes the variance
of the share into main effects and interactions of the design factors; the
seed (split + model randomness) is the replicate, i.e. the residual.
Sums of squares are then pooled over features, giving the share of all
attribution variation attributable to each source.

    NHANES (label axis):  model x method x label        (per cycle; 10 seeds)
    BRFSS  (MAPR axis):   model x method x year         (5 seeds)

This is the quantitative version of "three sources of variation: model,
XAI method, label" (thầy Nghiệp, email 16/09/2026).

Outputs (outputs/variance_decomposition/):
    <dataset>_<response>_by_feature.csv   SS and eta^2 per feature and source
    <dataset>_<response>_pooled.csv       pooled eta^2 per source
with response = share (attribution share) or rank (within-vector rank).

Usage:
    python -m src.analysis.run_variance_decomposition
"""

from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs"
RESULT_DIR = OUT / "variance_decomposition"


def balanced_anova(df: pd.DataFrame, factors: list[str], response: str) -> dict[str, float]:
    """Sums of squares for every main effect and interaction of ``factors``
    in a balanced design; the remainder (replicates) is the residual."""
    grand = df[response].mean()
    total = float(((df[response] - grand) ** 2).sum())
    marginal: dict[tuple, float] = {}
    for k in range(1, len(factors) + 1):
        for subset in combinations(factors, k):
            g = df.groupby(list(subset))[response]
            marginal[subset] = float((g.transform("mean") - grand).pow(2).sum())
    effect: dict[tuple, float] = {}
    for k in range(1, len(factors) + 1):
        for subset in combinations(factors, k):
            lower = sum(effect[t] for j in range(1, k) for t in combinations(subset, j))
            effect[subset] = marginal[subset] - lower
    out = {" x ".join(s): v for s, v in effect.items()}
    out["residual (seed)"] = total - sum(effect.values())
    out["total"] = total
    return out


def decompose(vec: pd.DataFrame, factors: list[str], name: str, group: list[str] | None = None,
              response: str = "share") -> None:
    """response = "share" (importance / vector sum) or "rank" (1 = most
    important). Shares carry concentration differences between SHAP and PI;
    ranks remove them, isolating disagreement about ORDER."""
    vec = vec.copy()
    vec["importance"] = vec["importance"].clip(lower=0.0)
    keys = factors + ["seed"] + (group or [])
    if response == "share":
        vec["share"] = vec["importance"] / vec.groupby(keys)["importance"].transform("sum")
    else:
        vec["share"] = vec.groupby(keys)["importance"].rank(ascending=False, method="average")
    name = f"{name}_{response}"
    rows = []
    for gkey, gdf in vec.groupby((group or []) + ["feature"]):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        cells = gdf.groupby(factors).size()
        if cells.nunique() != 1:
            raise ValueError(f"Unbalanced design for {gkey}: {cells.unique()}")
        ss = balanced_anova(gdf, factors, "share")
        for source, v in ss.items():
            rows.append({**dict(zip((group or []) + ["feature"], gkey)), "source": source, "ss": v})
    by_feat = pd.DataFrame(rows)
    idx = (group or []) + ["feature"]
    tot = by_feat[by_feat.source == "total"].set_index(idx)["ss"].rename("total_ss")
    by_feat = by_feat.join(tot, on=idx)
    by_feat["eta2"] = by_feat["ss"] / by_feat["total_ss"]
    by_feat.round(6).to_csv(RESULT_DIR / f"{name}_by_feature.csv", index=False)

    g = list(group or [])
    pooled = by_feat[by_feat.source != "total"].groupby(g + ["source"])["ss"].sum()
    pooled = pooled / (pooled.groupby(level=g).transform("sum") if g else pooled.sum())
    pooled = pooled.rename("pooled_eta2").reset_index()
    pooled.round(4).to_csv(RESULT_DIR / f"{name}_pooled.csv", index=False)
    logging.info("%s pooled eta^2:\n%s", name, pooled.round(3).to_string(index=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    nh = OUT / "label_axis" / "vectors.csv"
    if nh.exists():
        v = pd.read_csv(nh)
        v = v[v.weighting == "unweighted"]
        for resp in ("share", "rank"):
            decompose(v, ["model", "method", "label"], "nhanes", group=["cycle"], response=resp)

    for cw in ["pipeline", "none", "balanced"]:
        br = OUT / "brfss_multiseed" / cw / "vectors.csv"
        if not br.exists():
            continue
        v = pd.read_csv(br, dtype={"year": str})
        n_models = v.model.nunique()
        if n_models < 3:
            logging.warning("BRFSS %s: only %d models present, skipped", cw, n_models)
            continue
        for resp in ("share", "rank"):
            decompose(v, ["model", "method", "year"], f"brfss_{cw}", response=resp)


if __name__ == "__main__":
    main()
