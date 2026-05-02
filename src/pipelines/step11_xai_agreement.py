"""
Step 11 — XAI Agreement Analysis (cross-dataset, cross-model)

This step is CROSS-RUN: it consumes SHAP and FI CSVs that were produced by
Step 05 across multiple ACTIVE_DATASET × ACTIVE_MODEL combinations.

Prerequisite: 12 files must exist under outputs/ (after running run_pipeline_all_combos.py
with the updated Step 05, which uses Permutation Importance as the unified FI method) :
    outputs/cdc_brfss_{year}_full/shap/{model}_shap_feature_importance.csv
    outputs/cdc_brfss_{year}_full/fi/{model}_feature_importance.csv
where year ∈ {2015, 2021}, model ∈ {xgboost, random_forest, logistic_regression}.

FI method: **Permutation Importance** (unified across all 3 models) — replacing
the previous mix of Gain (XGB) / MDI (RF) / none-for-LR. This makes SHAP-vs-FI
agreement mathematically comparable across architectures. See step05 for details.

PRIMARY METRIC — Computed ABC (cABC) Analysis [Ultsch & Lötsch 2015, PLoS ONE]:
Partitions features into 3 groups A/B/C via Lorenz curve geometry — breakpoints
are data-driven (argmax of deviation from diagonal), NOT arbitrary thresholds.
This addresses boundary sensitivity of fixed thresholds (e.g., "feature at 79%
vs 81% cumulative coverage") and is the established method for feature-importance
categorization in biomedical ML [Lötsch & Ultsch 2020, Springer; 2023, Sci Rep].

Agreement measured via per-group Jaccard (J_A, J_B, J_C) — reflects
"are the essential features the same?" rather than "is the exact rank the same?".

LEGACY METRICS (kept for backward compatibility and multi-metric reporting):
    (1) Coverage-based Jaccard at thresholds 70%, 80%, 90% cumulative contribution
    (2) Rank-Biased Overlap (RBO) with p = 0.9 (top-heavy weighting)
    (3) Cosine similarity on L1-normalized importance vectors

Writes CSVs into outputs/xai_agreement/ :
    within_model_agreement.csv         (legacy metrics)
    cross_model_shap_agreement.csv     (legacy metrics)
    temporal_stability.csv             (legacy metrics)
    within_model_cabc.csv              (NEW: cABC group agreement)
    cross_model_shap_cabc.csv          (NEW: cABC group agreement)
    temporal_stability_cabc.csv        (NEW: cABC group agreement)
    cabc_groups.csv                    (NEW: per-run A/B/C membership)

Usage (standalone, after all 6 combos have finished):
    python -m src.pipelines.step11_xai_agreement

Or call run_step11_xai_agreement() from run_pipeline_all_combos.py.

References:
    Ultsch, A. & Lötsch, J. (2015). Computed ABC analysis for rational selection
        of most informative variables in multivariate data. PLoS ONE 10(6): e0129767.
    Lötsch, J. & Ultsch, A. (2020). Random Forests Followed by Computed ABC
        Analysis as a Feature Selection Method for Machine Learning in Biomedical
        Data. In: Advanced Studies in Classification and Data Science, Springer.
    Lötsch, J. & Ultsch, A. (2023). Recursive computed ABC (cABC) analysis as a
        precise method for reducing machine learning based feature sets to their
        minimum informative size. Scientific Reports 13, 5470.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.project_paths import OUTPUT_DIR


# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

# Datasets and models to aggregate across.
# Three-phase temporal study: pre-pandemic (2015), mid-pandemic (2021), post-acute (2023).
# Datasets follow the Teboul recoding convention rebuilt from raw CDC XPT.
YEARS = ["2015", "2021", "2023"]
DATASET_SLUGS = [f"cdc_brfss_diabetes_{y}" for y in YEARS]
MODELS = ["xgboost", "random_forest", "logistic_regression"]

# Strip ColumnTransformer prefixes so SHAP ("GenHlth") and FI ("num__GenHlth") align.
FEATURE_PREFIX_RE = re.compile(r"^(num|cat|ord|txt|bin)__")

COVERAGE_THRESHOLDS = [0.70, 0.80, 0.90]
RBO_P = 0.9
RBO_DEPTH_CAP = 200


# ------------------------------------------------------------------ #
# I/O                                                                 #
# ------------------------------------------------------------------ #

def _normalize_feature_name(name: str) -> str:
    """Strip 'num__', 'cat__', ... so SHAP and FI rows align by feature."""
    return FEATURE_PREFIX_RE.sub("", str(name))


def _read_importance_csv(path: Path) -> pd.Series:
    """Read a 2-column CSV → Series[feature → |importance|], sorted desc."""
    df = pd.read_csv(path)
    feat_col = "feature" if "feature" in df.columns else df.columns[0]
    val_col = "importance" if "importance" in df.columns else df.columns[1]

    feats = df[feat_col].astype(str).map(_normalize_feature_name)
    vals = pd.to_numeric(df[val_col], errors="coerce").abs()

    s = pd.Series(vals.values, index=feats.values, dtype=float)
    s = s[~s.index.duplicated(keep="first")]
    return s.sort_values(ascending=False)


def _load_all_importances() -> Dict[Tuple[str, str, str], pd.Series]:
    """Scan outputs/ for all importance CSVs. Returns dict[(year, model, method)] = Series."""
    imp: Dict[Tuple[str, str, str], pd.Series] = {}
    for year, slug in zip(YEARS, DATASET_SLUGS):
        ds_dir = OUTPUT_DIR / slug
        if not ds_dir.exists():
            logging.warning(f"Dataset output dir not found: {ds_dir}")
            continue
        for model in MODELS:
            shap_path = ds_dir / "shap" / f"{model}_shap_feature_importance.csv"
            fi_path = ds_dir / "fi" / f"{model}_feature_importance.csv"

            if shap_path.exists():
                imp[(year, model, "shap")] = _read_importance_csv(shap_path)
                logging.info(f"  loaded SHAP: {year}/{model} (n={len(imp[(year, model, 'shap')])})")
            else:
                logging.warning(f"  missing SHAP: {shap_path}")

            if fi_path.exists():
                imp[(year, model, "fi")] = _read_importance_csv(fi_path)
                logging.info(f"  loaded FI:   {year}/{model} (n={len(imp[(year, model, 'fi')])})")
            else:
                logging.warning(f"  missing FI:  {fi_path}")
    return imp


# ------------------------------------------------------------------ #
# Metric 1 — Coverage-based Jaccard                                   #
# ------------------------------------------------------------------ #

def _coverage_set(importance: pd.Series, alpha: float) -> set:
    """Smallest feature set whose cumulative |importance| ≥ α·total."""
    s = importance.abs().sort_values(ascending=False)
    total = s.sum()
    if total == 0:
        return set()
    cum = s.cumsum() / total
    k = int((cum.values >= alpha).argmax()) + 1
    return set(s.index[:k])


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return 0.0 if not u else len(a & b) / len(u)


def _coverage_jaccard(imp_a: pd.Series, imp_b: pd.Series, alpha: float) -> Dict[str, float]:
    sa = _coverage_set(imp_a, alpha)
    sb = _coverage_set(imp_b, alpha)
    pct = int(alpha * 100)
    return {
        f"J@{pct}":    round(_jaccard(sa, sb), 4),
        f"|Sa|@{pct}": len(sa),
        f"|Sb|@{pct}": len(sb),
    }


# ------------------------------------------------------------------ #
# Metric 2 — Rank-Biased Overlap                                      #
# ------------------------------------------------------------------ #

def _rbo(list_a: List[str], list_b: List[str],
         p: float = RBO_P, depth_cap: int = RBO_DEPTH_CAP) -> float:
    """RBO = (1-p) * Σ_{d=1}^{k} p^{d-1} * A_d."""
    k = min(len(list_a), len(list_b), depth_cap)
    if k == 0:
        return 0.0
    seen_a, seen_b = set(), set()
    overlap = 0
    s = 0.0
    for d in range(1, k + 1):
        a_d, b_d = list_a[d - 1], list_b[d - 1]
        if a_d == b_d:
            overlap += 1
        else:
            if a_d in seen_b:
                overlap += 1
            if b_d in seen_a:
                overlap += 1
        seen_a.add(a_d)
        seen_b.add(b_d)
        s += (p ** (d - 1)) * (overlap / d)
    return (1.0 - p) * s


# ------------------------------------------------------------------ #
# Metric 3 — Cosine on L1-normalized vectors                          #
# ------------------------------------------------------------------ #

def _cosine(imp_a: pd.Series, imp_b: pd.Series) -> float:
    feats = sorted(set(imp_a.index) | set(imp_b.index))
    va = imp_a.reindex(feats, fill_value=0.0).abs().to_numpy(dtype=float)
    vb = imp_b.reindex(feats, fill_value=0.0).abs().to_numpy(dtype=float)
    sa, sb = va.sum(), vb.sum()
    if sa == 0 or sb == 0:
        return 0.0
    va, vb = va / sa, vb / sb
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# ------------------------------------------------------------------ #
# Metric 4 — Computed ABC (cABC) analysis                             #
# Ultsch & Lötsch (2015), PLoS ONE 10(6):e0129767                      #
# ------------------------------------------------------------------ #

def _cabc_split(imp: pd.Series) -> Tuple[List[str], List[str], List[str]]:
    """
    Partition features into A/B/C groups via computed ABC analysis.

    Breakpoints are computed geometrically from the Lorenz-like ABC curve:
      - x = cumulative fraction of items (sorted by importance descending)
      - y = cumulative fraction of total importance
      - A/B breakpoint: argmax of (y - x) → "maximum disproportion" point.
        This is the point where marginal contribution equals the mean,
        beyond which additional features contribute less than average.
      - B/C breakpoint: apply the same rule recursively to items after A,
        i.e., find argmax of (y - x) on the remaining curve.

    Unlike a fixed α-threshold (e.g., 80%), these breakpoints are data-driven
    — a feature near the boundary (say cumulative 79% vs 81%) will fall on
    the same side of the break as long as the curve shape does not change,
    because the break is determined by the curve's geometry, not by α.

    Returns:
        (A, B, C) — three lists of feature names (sorted by importance desc).
        If the series is empty or all-zero, returns (features, [], []).
    """
    if len(imp) == 0:
        return [], [], []
    srt = imp.abs().sort_values(ascending=False)
    values = srt.to_numpy(dtype=float)
    feats = list(srt.index)
    n = len(values)
    total = values.sum()
    if total <= 0:
        return feats, [], []

    # Primary split: A vs (B+C)
    x = np.arange(1, n + 1) / n
    y = np.cumsum(values) / total
    ab_idx = int(np.argmax(y - x))  # 0-based index of last item in A

    # Recursive split on remaining items: B vs C
    remaining = values[ab_idx + 1:]
    if len(remaining) == 0:
        return feats[:ab_idx + 1], [], []
    rem_total = remaining.sum()
    if rem_total <= 0:
        return feats[:ab_idx + 1], feats[ab_idx + 1:], []

    rx = np.arange(1, len(remaining) + 1) / len(remaining)
    ry = np.cumsum(remaining) / rem_total
    bc_local = int(np.argmax(ry - rx))
    bc_idx = ab_idx + 1 + bc_local

    A = feats[:ab_idx + 1]
    B = feats[ab_idx + 1:bc_idx + 1]
    C = feats[bc_idx + 1:]
    return A, B, C


def _jaccard(s1: List[str], s2: List[str]) -> float:
    """Jaccard similarity of two sets (given as lists)."""
    if not s1 and not s2:
        return 1.0
    S1, S2 = set(s1), set(s2)
    union = S1 | S2
    if not union:
        return 0.0
    return len(S1 & S2) / len(union)


def _cabc_agreement(
    imp_a: pd.Series, imp_b: pd.Series
) -> Dict[str, object]:
    """Compare two feature rankings via cABC group memberships."""
    A1, B1, C1 = _cabc_split(imp_a)
    A2, B2, C2 = _cabc_split(imp_b)
    return {
        "|A_1|": len(A1),
        "|A_2|": len(A2),
        "|B_1|": len(B1),
        "|B_2|": len(B2),
        "|C_1|": len(C1),
        "|C_2|": len(C2),
        "J_A": round(_jaccard(A1, A2), 4),
        "J_B": round(_jaccard(B1, B2), 4),
        "J_C": round(_jaccard(C1, C2), 4),
        "A_1": ",".join(A1),
        "A_2": ",".join(A2),
        "B_1": ",".join(B1),
        "B_2": ",".join(B2),
        "C_1": ",".join(C1),
        "C_2": ",".join(C2),
    }


def _intersect_features(imp_a: pd.Series, imp_b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Restrict two importance Series to the intersection of their feature sets.

    Used for cross-temporal comparisons where some features may be absent
    in one year due to BRFSS schema changes (e.g. Fruits, Veggies,
    AnyHealthcare, HvyAlcoholConsump removed from BRFSS 2023).

    Without this, schema-removed features get treated as "present with zero
    importance" in the year that lacks them — a schema artifact that biases
    Group C / J@α metrics.
    """
    common = sorted(set(imp_a.index) & set(imp_b.index))
    a = imp_a.loc[common].sort_values(ascending=False)
    b = imp_b.loc[common].sort_values(ascending=False)
    return a, b


# ------------------------------------------------------------------ #
# Aggregate                                                           #
# ------------------------------------------------------------------ #

def _compute_all_metrics(imp_a: pd.Series, imp_b: pd.Series) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for alpha in COVERAGE_THRESHOLDS:
        row.update(_coverage_jaccard(imp_a, imp_b, alpha))
    row["RBO"] = round(_rbo(list(imp_a.index), list(imp_b.index), p=RBO_P), 4)
    row["cosine"] = round(_cosine(imp_a, imp_b), 4)
    return row


# ------------------------------------------------------------------ #
# Three analyses                                                      #
# ------------------------------------------------------------------ #

def _within_model(imp: Dict) -> pd.DataFrame:
    """SHAP vs native FI, for each (model, year). LR skipped (no FI)."""
    rows = []
    for year in YEARS:
        for model in MODELS:
            k_shap, k_fi = (year, model, "shap"), (year, model, "fi")
            if k_shap not in imp or k_fi not in imp:
                rows.append({"year": year, "model": model,
                             "method_A": "SHAP", "method_B": "native_FI",
                             "note": "SKIPPED (no native FI for this model)"})
                continue
            r = {"year": year, "model": model,
                 "method_A": "SHAP", "method_B": "native_FI", "note": ""}
            r.update(_compute_all_metrics(imp[k_shap], imp[k_fi]))
            rows.append(r)
    return pd.DataFrame(rows)


def _cross_model_shap(imp: Dict) -> pd.DataFrame:
    pairs = [("xgboost", "random_forest"),
             ("xgboost", "logistic_regression"),
             ("random_forest", "logistic_regression")]
    rows = []
    for year in YEARS:
        for m1, m2 in pairs:
            k1, k2 = (year, m1, "shap"), (year, m2, "shap")
            if k1 not in imp or k2 not in imp:
                continue
            r = {"year": year, "model_A": m1, "model_B": m2, "method": "SHAP"}
            r.update(_compute_all_metrics(imp[k1], imp[k2]))
            rows.append(r)
    return pd.DataFrame(rows)


def _temporal_stability(imp: Dict) -> pd.DataFrame:
    """Cross-temporal stability across all year pairs.

    For 3 years (2015, 2021, 2023) generates 3 pairs:
        2015 vs 2021, 2015 vs 2023, 2021 vs 2023

    Uses intersection of features for each pair (schema-robust):
        2015 vs 2021: full 21 features (same schema)
        2015 vs 2023: 17 common features (4 schema-removed in 2023)
        2021 vs 2023: 17 common features
    """
    from itertools import combinations
    rows = []
    for model in MODELS:
        for method in ["shap", "fi"]:
            for year_a, year_b in combinations(YEARS, 2):
                k_a = (year_a, model, method)
                k_b = (year_b, model, method)
                if k_a not in imp or k_b not in imp:
                    continue
                # Intersection-only for cross-temporal
                ia, ib = _intersect_features(imp[k_a], imp[k_b])
                r = {"model": model, "method": method.upper(),
                     "year_A": year_a, "year_B": year_b,
                     "n_common_features": len(ia)}
                r.update(_compute_all_metrics(ia, ib))
                rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# cABC analysis runs (parallel to legacy metric analyses)             #
# ------------------------------------------------------------------ #

def _within_model_cabc(imp: Dict) -> pd.DataFrame:
    """SHAP vs PI per (year, model), compared via cABC group memberships."""
    rows = []
    for year in YEARS:
        for model in MODELS:
            k_shap, k_fi = (year, model, "shap"), (year, model, "fi")
            if k_shap not in imp or k_fi not in imp:
                continue
            r = {"year": year, "model": model,
                 "method_A": "SHAP", "method_B": "PI"}
            r.update(_cabc_agreement(imp[k_shap], imp[k_fi]))
            rows.append(r)
    return pd.DataFrame(rows)


def _cross_model_shap_cabc(imp: Dict) -> pd.DataFrame:
    pairs = [("xgboost", "random_forest"),
             ("xgboost", "logistic_regression"),
             ("random_forest", "logistic_regression")]
    rows = []
    for year in YEARS:
        for m1, m2 in pairs:
            k1, k2 = (year, m1, "shap"), (year, m2, "shap")
            if k1 not in imp or k2 not in imp:
                continue
            r = {"year": year, "model_A": m1, "model_B": m2, "method": "SHAP"}
            r.update(_cabc_agreement(imp[k1], imp[k2]))
            rows.append(r)
    return pd.DataFrame(rows)


def _temporal_stability_cabc(imp: Dict) -> pd.DataFrame:
    """Cross-temporal cABC group agreement across all year pairs.

    For 3 years generates 3 pairs. Uses intersection of features (schema-robust)
    so that 4 features removed from BRFSS 2023 (Fruits, Veggies, AnyHealthcare,
    HvyAlcoholConsump) are not artifact-classified as marginal Group C.
    """
    from itertools import combinations
    rows = []
    for model in MODELS:
        for method in ["shap", "fi"]:
            for year_a, year_b in combinations(YEARS, 2):
                k_a = (year_a, model, method)
                k_b = (year_b, model, method)
                if k_a not in imp or k_b not in imp:
                    continue
                ia, ib = _intersect_features(imp[k_a], imp[k_b])
                r = {"model": model, "method": method.upper(),
                     "year_A": year_a, "year_B": year_b,
                     "n_common_features": len(ia)}
                r.update(_cabc_agreement(ia, ib))
                rows.append(r)
    return pd.DataFrame(rows)


def _cabc_groups_per_run(imp: Dict) -> pd.DataFrame:
    """Emit A/B/C group membership for every (year, model, method) run."""
    rows = []
    for (year, model, method), series in imp.items():
        A, B, C = _cabc_split(series)
        rows.append({
            "year": year,
            "model": model,
            "method": method.upper(),
            "|A|": len(A),
            "|B|": len(B),
            "|C|": len(C),
            "A": ",".join(A),
            "B": ",".join(B),
            "C": ",".join(C),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Public entry point                                                  #
# ------------------------------------------------------------------ #

def run_step11_xai_agreement() -> Dict[str, pd.DataFrame]:
    """
    Cross-run XAI agreement analysis.

    Returns a dict with three DataFrames:
        {"within": ..., "cross": ..., "temporal": ...}
    Also writes three CSVs under outputs/xai_agreement/.
    """
    logging.info("============================================================")
    logging.info("STEP 11 - XAI AGREEMENT (cross-run)")
    logging.info("============================================================")

    out_dir = OUTPUT_DIR / "xai_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("[1/4] Loading importance CSVs from all dataset × model combinations ...")
    imp = _load_all_importances()
    if not imp:
        logging.error("No importance files found. Run the 6 combos (3 models × 2 years) first.")
        return {}

    logging.info("[2/4] Computing within-model agreement (SHAP vs native FI) ...")
    df_within = _within_model(imp)
    df_within.to_csv(out_dir / "within_model_agreement.csv", index=False)
    logging.info(f"\n{df_within.to_string(index=False)}")

    logging.info("[3/4] Computing cross-model SHAP agreement ...")
    df_cross = _cross_model_shap(imp)
    df_cross.to_csv(out_dir / "cross_model_shap_agreement.csv", index=False)
    logging.info(f"\n{df_cross.to_string(index=False)}")

    logging.info("[4/4] Computing temporal stability (2015 vs 2021) ...")
    df_time = _temporal_stability(imp)
    df_time.to_csv(out_dir / "temporal_stability.csv", index=False)
    logging.info(f"\n{df_time.to_string(index=False)}")

    # ------------------- cABC analyses (primary method) -------------- #
    logging.info("[cABC 1/4] Within-model cABC group agreement (SHAP vs PI) ...")
    df_within_cabc = _within_model_cabc(imp)
    df_within_cabc.to_csv(out_dir / "within_model_cabc.csv", index=False)

    logging.info("[cABC 2/4] Cross-model SHAP cABC group agreement ...")
    df_cross_cabc = _cross_model_shap_cabc(imp)
    df_cross_cabc.to_csv(out_dir / "cross_model_shap_cabc.csv", index=False)

    logging.info("[cABC 3/4] Temporal stability via cABC (2015 vs 2021) ...")
    df_time_cabc = _temporal_stability_cabc(imp)
    df_time_cabc.to_csv(out_dir / "temporal_stability_cabc.csv", index=False)

    logging.info("[cABC 4/4] Per-run A/B/C group membership ...")
    df_groups = _cabc_groups_per_run(imp)
    df_groups.to_csv(out_dir / "cabc_groups.csv", index=False)

    logging.info(f"STEP 11 COMPLETED. CSVs in: {out_dir}")
    return {
        "within": df_within,
        "cross": df_cross,
        "temporal": df_time,
        "within_cabc": df_within_cabc,
        "cross_cabc": df_cross_cabc,
        "temporal_cabc": df_time_cabc,
        "cabc_groups": df_groups,
    }


if __name__ == "__main__":
    # Allow standalone invocation: `python -m src.pipelines.step11_xai_agreement`
    from src.utils.logging_config import configure_logging
    configure_logging()
    run_step11_xai_agreement()