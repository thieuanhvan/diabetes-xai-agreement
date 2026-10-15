"""
Do aggregate agreement metrics detect replicated outcome-definition shifts?
(v32, Spec B). Uses stored jbi-v1 vectors only; nothing is refitted.

Configuration = (cycle, method, model), unweighted importance.
Seed reference = all seed pairs of the configuration pooled over the three
labels (as in run_label_axis.run_pairs). A metric FLAGS a configuration when
the number of diag-vs-second-label pairs (one per seed) below the seed 5th
percentile reaches k, the smallest count with binomial tail P(X >= k | n,
0.05) < 0.05 (n = 10 seeds -> k = 3; n = 5 -> k = 2). contradiction_A is
binary: flagged when its rate in label pairs exceeds the seed rate with
one-sided Fisher p < 0.05.

B1  detection_config.csv     per configuration x metric: pairs below p5, flag
B2  feature_visibility.csv   per replicated shift: Group A / top-5 membership
                             changes and rank change across seeds
B3  mds.csv                  minimum detectable single-feature shift (share
                             points) by injection into the diag vectors
    mds_joint_lambda.csv     smallest multiple lambda of the observed mean
                             shift vector that is flagged (lambda* > 1: the
                             observed shift is below the detection threshold)
B4  the same for hypertension (positive control)

Outputs: outputs/detection_power/<outcome>/
Usage:
    python -m src.analysis.run_detection_power
"""

from __future__ import annotations

import logging
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binom, fisher_exact

from src.analysis import label_controls_core as C
from src.analysis.run_soa_comparison import krishna
from src.evaluation.agreement_metrics import cabc_partition, compare

METRICS = ["J_A", "OC_A", "size_ratio_A", "top5", "top10", "spearman", "kendall_tau_b",
           "weighted_tau", "rbo_ext_p90", "feature_agreement", "rank_agreement",
           "pairwise_rank_agreement"]
LABELS3 = {"diabetes": ["diag", "lab", "total"], "hypertension": ["diag", "measured", "total"]}
DELTAS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15]
LAMBDAS = [round(x, 2) for x in np.arange(0.0, 3.01, 0.1)]


def k_needed(n: int, p: float = 0.05, alpha: float = 0.05) -> int:
    for k in range(n + 1):
        if binom.sf(k - 1, n, p) < alpha:
            return k
    return n + 1


def metrics(a: pd.Series, b: pd.Series) -> dict:
    m = compare(a, b)
    m.update({k: v for k, v in krishna(a, b).items() if k not in m})
    return {k: m[k] for k in METRICS + ["contradiction_A"]}


def load(outcome: str):
    v = pd.read_csv(C.V1_DIR[outcome] / "vectors.csv")
    v = v[v.weighting == "unweighted"].copy()
    v["importance"] = v["importance"].clip(lower=0.0)
    vec = {k: g.set_index("feature")["importance"] / g["importance"].sum()
           for k, g in v.groupby(["cycle", "method", "model", "label", "seed"])}
    return vec, sorted(v.seed.unique())


def seed_floor(vec, seeds, cfg, labels):
    rows = [metrics(vec[cfg + (lab, s1)], vec[cfg + (lab, s2)])
            for lab in labels for s1, s2 in combinations(seeds, 2)]
    return pd.DataFrame(rows)


def flag(pairs: pd.DataFrame, floor: pd.DataFrame, k: int) -> dict:
    out = {}
    for m in METRICS:
        p5 = floor[m].quantile(0.05)
        n_below = int((pairs[m] < p5 - 1e-12).sum())
        out[m] = (n_below, p5, n_below >= k)
    c_lab, c_seed = int(pairs.contradiction_A.sum()), int(floor.contradiction_A.sum())
    p = fisher_exact([[c_lab, len(pairs) - c_lab], [c_seed, len(floor) - c_seed]],
                     alternative="greater").pvalue
    out["contradiction_A"] = (c_lab, c_seed / len(floor), p < 0.05)
    return out


def inject(p: pd.Series, f: str, delta: float) -> pd.Series:
    q = p.copy()
    new_f = max(q[f] + delta, 0.0)
    rest = q.drop(f)
    rest = rest * (1.0 - new_f) / rest.sum() if rest.sum() > 0 else rest
    q.loc[rest.index] = rest
    q[f] = new_f
    return q


def run(outcome: str, dst=None) -> None:
    dst = dst or C.OUT_DIR / "detection_power" / outcome
    dst.mkdir(parents=True, exist_ok=True)
    vec, seeds = load(outcome)
    second, labels = C.SECOND[outcome], LABELS3[outcome]
    k = k_needed(len(seeds))
    rep = C.v1_replicated(outcome)
    pos = set(map(tuple, rep[["method", "model"]].to_numpy()))
    cfgs = sorted({key[:3] for key in vec})
    b1, b3, lam_rows = [], [], []
    for cfg in cfgs:
        cycle, method, model = cfg
        floor = seed_floor(vec, seeds, cfg, labels)
        pairs = pd.DataFrame([metrics(vec[cfg + ("diag", s)], vec[cfg + (second, s)]) for s in seeds])
        n_shift = int(((rep.method == method) & (rep.model == model)).sum())
        for m, (nb, ref, fl) in flag(pairs, floor, k).items():
            b1.append(dict(cycle=cycle, method=method, model=model, n_replicated_shifts=n_shift,
                           positive=(method, model) in pos, metric=m, n_pairs=len(pairs),
                           n_below_or_count=nb, seed_p5_or_rate=ref, k_needed=k, flagged=fl,
                           label_mean=pairs[m].mean(), seed_mean=floor[m].mean()))
        # B3 single-feature injection into diag vectors, paired by seed
        base = {s: vec[cfg + ("diag", s)] for s in seeds}
        for f in base[seeds[0]].index:
            for sign in (+1, -1):
                mds = {m: np.inf for m in METRICS + ["contradiction_A"]}
                for d in DELTAS:
                    pr = pd.DataFrame([metrics(base[s], inject(base[s], f, sign * d)) for s in seeds])
                    for m, (_, _, fl) in flag(pr, floor, k).items():
                        if fl and mds[m] == np.inf:
                            mds[m] = d
                    if all(np.isfinite(x) for x in mds.values()):
                        break
                for m, x in mds.items():
                    b3.append(dict(cycle=cycle, method=method, model=model, feature=f, sign=sign,
                                   metric=m, mds=x))
        # joint lambda: p' = p_diag - lambda * mean(p_diag - p_second), clipped and renormalised
        dvec = pd.concat([vec[cfg + ("diag", s)] - vec[cfg + (second, s)] for s in seeds], axis=1).mean(axis=1)
        lam_star = {m: np.inf for m in METRICS + ["contradiction_A"]}
        for lam in LAMBDAS[1:]:
            prs = []
            for s in seeds:
                q = (base[s] - lam * dvec).clip(lower=0.0)
                prs.append(metrics(base[s], q / q.sum()))
            for m, (_, _, fl) in flag(pd.DataFrame(prs), floor, k).items():
                if fl and lam_star[m] == np.inf:
                    lam_star[m] = lam
        for m, x in lam_star.items():
            lam_rows.append(dict(cycle=cycle, method=method, model=model,
                                 positive=(method, model) in pos, metric=m, lambda_star=x,
                                 l1_mean_shift=float(dvec.abs().sum())))
        logging.info("%s %s %s %s done", outcome, cycle, method, model)

    b1 = pd.DataFrame(b1)
    b1.to_csv(dst / "detection_config.csv", index=False)
    b1.pivot_table(index="metric", columns=["cycle", "method", "model"], values="n_below_or_count") \
      .to_csv(dst / "detection_matrix.csv")
    pd.DataFrame(b3).to_csv(dst / "mds.csv", index=False)
    pd.DataFrame(lam_rows).to_csv(dst / "mds_joint_lambda.csv", index=False)

    # B2 feature visibility for the replicated shifts
    vis = []
    for _, r in rep.iterrows():
        for cycle in C.CYCLES:
            cfg = (cycle, r.method, r.model)
            ch_a = ch_t5 = 0
            rk = []
            for s in seeds:
                a, b = vec[cfg + ("diag", s)], vec[cfg + (second, s)]
                ga, gb = set(cabc_partition(a).group_a), set(cabc_partition(b).group_a)
                ta = set(a.sort_values(ascending=False).index[:5])
                tb = set(b.sort_values(ascending=False).index[:5])
                ch_a += (r.feature in ga) != (r.feature in gb)
                ch_t5 += (r.feature in ta) != (r.feature in tb)
                ra = list(a.sort_values(ascending=False).index).index(r.feature) + 1
                rb = list(b.sort_values(ascending=False).index).index(r.feature) + 1
                rk.append((ra, rb))
            vis.append(dict(method=r.method, model=r.model, feature=r.feature, cycle=cycle,
                            share_diff_pp=100 * (r.diff_v1_1720 if cycle == C.CYCLES[0] else r.diff_v1_2123),
                            share_A_membership_changed=ch_a / len(seeds),
                            share_top5_membership_changed=ch_t5 / len(seeds),
                            median_rank_diag=float(np.median([x[0] for x in rk])),
                            median_rank_second=float(np.median([x[1] for x in rk]))))
    pd.DataFrame(vis).to_csv(dst / "feature_visibility.csv", index=False)
    logging.info("%s B1 flags (rows metric, columns config):\n%s", outcome,
                 b1.pivot_table(index="metric", columns=["cycle", "method", "model"],
                                values="flagged").astype(int).to_string())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for outcome in ["diabetes", "hypertension"]:
        run(outcome)


if __name__ == "__main__":
    main()
