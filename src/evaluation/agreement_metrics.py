"""
Agreement metrics for comparing two global feature-importance vectors.

Single source of truth for the post-MAPR metric comparison
(src/analysis/run_metric_comparison.py). The MAPR 2026 outputs under
outputs/xai_agreement/ were produced by src/analysis/run_xai_agreement.py and
are frozen at git tag ``mapr2026-v1.0``; this module does not alter them.

Metric families
---------------
1. cABC group partition (Group A / B / C) and per-group Jaccard J_A, J_B, J_C.
   The partition rule is identical to ``build_cabc`` in run_xai_agreement.py:
   the A|B boundary is ``argmax(y - x)`` on the ABC curve, where x is the
   cumulative fraction of features and y the cumulative fraction of importance.

   Property (proved in ``cabc_partition`` docstring and checked by
   tests/test_agreement_metrics.py): for a descending-sorted vector this
   boundary is the last feature whose importance exceeds the MEAN importance.
   The rule is therefore scale-free and data-adaptive, but it is NOT
   threshold-free: the threshold is the mean (the "break-even" point of the
   ABC curve, where the curve's slope equals 1).

2. Containment / contradiction decomposition of Group-A agreement:
   - overlap coefficient OC_A = |A1 & A2| / min(|A1|, |A2|)
     (1.0 when one Group A is nested in the other)
   - size ratio = min(|A1|, |A2|) / max(|A1|, |A2|)  (depth agreement)
   - contradiction flag: each side has at least one Group-A feature the other
     side lacks.
   When there is no contradiction, J_A equals the size ratio exactly, so J_A
   alone cannot distinguish "different essential features" from "same
   essential features, different depth".

3. Baselines requested by MAPR reviewers: top-K overlap, Spearman rho,
   Kendall tau-b, weighted Kendall tau (hyperbolic, top-weighted), and
   extrapolated Rank-Biased Overlap (RBO_ext, Webber et al. 2010, eq. 32).

References
----------
Ultsch, A. & Lötsch, J. (2015). Computed ABC analysis for rational selection
    of most informative variables in multivariate data. PLoS ONE 10(6):e0129767.
Webber, W., Moffat, A. & Zobel, J. (2010). A similarity measure for indefinite
    rankings. ACM Transactions on Information Systems 28(4):20.
Vigna, S. (2015). A weighted correlation index for rankings with ties.
    Proceedings of WWW 2015, 1166-1176.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, weightedtau


# ------------------------------------------------------------------ #
# cABC partition                                                      #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class CABCPartition:
    group_a: List[str]
    group_b: List[str]
    group_c: List[str]
    mean_importance: float

    @property
    def size_a(self) -> int:
        return len(self.group_a)


def _argmax_gap(values: np.ndarray) -> int:
    """0-based index of the last item kept by argmax(y - x) on the ABC curve."""
    n = len(values)
    y = np.cumsum(values) / values.sum()
    x = np.arange(1, n + 1) / n
    return int(np.argmax(y - x))


def cabc_partition(importance: pd.Series) -> CABCPartition:
    """
    Two-pass cABC partition, identical to ``build_cabc`` in run_xai_agreement.py.

    Pass 1 on all features gives A vs (B + C); pass 2 on the remainder gives
    B vs C.

    Why Group A = {features with importance > mean}:
    with values v_1 >= ... >= v_n and total T, the gap D(k) = y_k - k/n
    changes by D(k) - D(k-1) = v_k / T - 1 / n, which is positive exactly
    when v_k > T / n (the mean). Because v is sorted, D rises while
    v_k > mean and falls afterwards, so argmax D is the last k with
    v_k > mean. Ties at exactly the mean are resolved by numpy argmax
    (first maximum), i.e. a feature equal to the mean is excluded.
    """
    s = importance.abs().sort_values(ascending=False)
    if len(s) == 0 or s.sum() <= 0:
        raise ValueError("Importance vector is empty or sums to zero.")
    feats = list(s.index)
    v = s.to_numpy(dtype=float)

    ab = _argmax_gap(v)
    rest = v[ab + 1:]
    if len(rest) == 0 or rest.sum() <= 0:
        return CABCPartition(feats[:ab + 1], feats[ab + 1:], [], float(v.mean()))
    bc = ab + 1 + _argmax_gap(rest)
    return CABCPartition(feats[:ab + 1], feats[ab + 1:bc + 1], feats[bc + 1:], float(v.mean()))


# ------------------------------------------------------------------ #
# Set-based metrics                                                   #
# ------------------------------------------------------------------ #

def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return 1.0 if not union else len(sa & sb) / len(union)


def overlap_coefficient(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 1.0 if sa == sb else 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def is_contradiction(a: Sequence[str], b: Sequence[str]) -> bool:
    """True when each set has at least one member the other lacks."""
    sa, sb = set(a), set(b)
    return bool(sa - sb) and bool(sb - sa)


def topk_overlap(rank_a: Sequence[str], rank_b: Sequence[str], k: int) -> float:
    """|top-k(a) & top-k(b)| / k (same definition as compute_topk_overlap)."""
    if k <= 0:
        raise ValueError("k must be positive.")
    return len(set(rank_a[:k]) & set(rank_b[:k])) / k


# ------------------------------------------------------------------ #
# Rank-based metrics                                                  #
# ------------------------------------------------------------------ #

def rbo_ext(rank_a: Sequence[str], rank_b: Sequence[str], p: float = 0.9) -> float:
    """
    Extrapolated Rank-Biased Overlap (Webber et al. 2010, eq. 32) for two
    conjoint rankings of equal length k:

        RBO_ext = (X_k / k) * p^k + (1 - p) / p * sum_{d=1..k} (X_d / d) * p^d

    where X_d = |top-d(a) & top-d(b)|. Identical rankings give exactly 1.0.

    Note: the finite sum used by ``rbo_score`` in run_xai_agreement.py omits
    the extrapolation term and is bounded above by 1 - p^k (0.8332 for
    k = 17, p = 0.9), so its values are not on the [0, 1] scale.
    """
    if len(rank_a) != len(rank_b):
        raise ValueError("rbo_ext expects two rankings of the same length.")
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1).")
    k = len(rank_a)
    if k == 0:
        return 1.0
    seen_a, seen_b = set(), set()
    total = 0.0
    x_d = 0
    for d in range(1, k + 1):
        seen_a.add(rank_a[d - 1])
        seen_b.add(rank_b[d - 1])
        x_d = len(seen_a & seen_b)
        total += (x_d / d) * p ** d
    return (x_d / k) * p ** k + (1.0 - p) / p * total


# ------------------------------------------------------------------ #
# Pairwise comparison                                                 #
# ------------------------------------------------------------------ #

def align(imp_a: pd.Series, imp_b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Restrict both vectors to their common features (same order)."""
    common = sorted(set(imp_a.index) & set(imp_b.index))
    if not common:
        raise ValueError("No common features between the two vectors.")
    return imp_a.loc[common].abs(), imp_b.loc[common].abs()


def compare(imp_a: pd.Series, imp_b: pd.Series,
            rbo_ps: Sequence[float] = (0.9, 0.8, 0.5),
            top_ks: Sequence[int] = (5, 6, 7, 10)) -> Dict[str, object]:
    """All agreement metrics for one pair of importance vectors."""
    a, b = align(imp_a, imp_b)
    rank_a = list(a.sort_values(ascending=False).index)
    rank_b = list(b.sort_values(ascending=False).index)
    pa, pb = cabc_partition(a), cabc_partition(b)

    out: Dict[str, object] = {
        "n_features": len(a),
        "size_A1": pa.size_a,
        "size_A2": pb.size_a,
        "J_A": jaccard(pa.group_a, pb.group_a),
        "J_B": jaccard(pa.group_b, pb.group_b),
        "J_C": jaccard(pa.group_c, pb.group_c),
        "OC_A": overlap_coefficient(pa.group_a, pb.group_a),
        "size_ratio_A": min(pa.size_a, pb.size_a) / max(pa.size_a, pb.size_a),
        "contradiction_A": int(is_contradiction(pa.group_a, pb.group_a)),
        "A_only_1": ",".join(sorted(set(pa.group_a) - set(pb.group_a))),
        "A_only_2": ",".join(sorted(set(pb.group_a) - set(pa.group_a))),
        "spearman": float(spearmanr(a.values, b.values).statistic),
        "kendall_tau_b": float(kendalltau(a.values, b.values).statistic),
        "weighted_tau": float(weightedtau(a.values, b.values).statistic),
        "top5_order_identical": int(rank_a[:5] == rank_b[:5]),
    }
    for k in top_ks:
        out[f"top{k}"] = topk_overlap(rank_a, rank_b, k)
    for p in rbo_ps:
        out[f"rbo_ext_p{int(round(p * 100)):02d}"] = rbo_ext(rank_a, rank_b, p)
    return out
