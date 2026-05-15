"""
Fairness Analysis Module — Equalized Odds & Demographic Parity
===============================================================

Tính các fairness metrics chính thức:
- TPR (True Positive Rate / Sensitivity) per group
- FPR (False Positive Rate) per group
- Equalized Odds violation = max(dTPR, dFPR)
- Demographic Parity difference = ΔPPV (predicted positive rate)

Demographic attributes trong BRFSS:
- Age    : 1–13 (1=18-24, 2=25-29, ..., 13=80+)
- Sex    : 0=Female, 1=Male
- Income : schema-dependent (BRFSS 2015: 1–8, BRFSS 2021: 1–11, BRFSS 2023: 1–7)

Schema heterogeneity for Income across years is automatically handled:
- BRFSS 2015 (INCOME2): 8 brackets, capped at $75K+
- BRFSS 2021 (INCOME3): 11 brackets, with high-income brackets up to >$200K
- BRFSS 2023 (_INCOMG1): 7 grouped brackets, capped at $200K+

Auto-detection:
    Looks at unique values of Income column. Picks the appropriate label map.

Cross-temporal interpretation warning:
    Raw δTPR for Income across years is NOT directly comparable due to schema
    heterogeneity. For cross-temporal analysis, restrict to common brackets
    (1-7 for 2015 vs 2023, 1-8 for 2015 vs 2021).
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path


# ── Label maps ────────────────────────────────────────────────────────────────
AGE_LABELS = {
    1.0: "18-24", 2.0: "25-29", 3.0: "30-34", 4.0: "35-39",
    5.0: "40-44", 6.0: "45-49", 7.0: "50-54", 8.0: "55-59",
    9.0: "60-64", 10.0: "65-69", 11.0: "70-74", 12.0: "75-79", 13.0: "80+",
}
SEX_LABELS = {0.0: "Female", 1.0: "Male"}

# BRFSS 2015 INCOME2 schema (1-8)
INCOME_LABELS_2015 = {
    1.0: "<$10K",   2.0: "$10-15K", 3.0: "$15-20K", 4.0: "$20-25K",
    5.0: "$25-35K", 6.0: "$35-50K", 7.0: "$50-75K", 8.0: "$75K+",
}

# BRFSS 2021 INCOME3 schema (1-11)
INCOME_LABELS_2021 = {
    1.0: "<$10K",     2.0: "$10-15K",   3.0: "$15-20K",  4.0: "$20-25K",
    5.0: "$25-35K",   6.0: "$35-50K",   7.0: "$50-75K",  8.0: "$75-100K",
    9.0: "$100-150K", 10.0: "$150-200K", 11.0: ">$200K",
}

# BRFSS 2023 _INCOMG1 schema (1-7, calculated grouping)
INCOME_LABELS_2023 = {
    1.0: "<$15K",   2.0: "$15-25K",   3.0: "$25-35K", 4.0: "$35-50K",
    5.0: "$50-100K", 6.0: "$100-200K", 7.0: "$200K+",
}

# Backward-compat alias (defaults to 2021 schema = most granular)
INCOME_LABELS = INCOME_LABELS_2021


def _detect_income_schema(income_values):
    """
    Auto-detect which BRFSS year's Income schema is in use.

    Decision rule (based on max valid bracket code):
        max == 8   -> BRFSS 2015 (INCOME2)
        max == 11  -> BRFSS 2021 (INCOME3)
        max == 7   -> BRFSS 2023 (_INCOMG1)
        otherwise  -> default to 2021 (most granular)
    """
    valid = pd.Series(income_values).dropna()
    if len(valid) == 0:
        return INCOME_LABELS_2021, "2021 (default, no data)"

    max_code = int(valid.max())
    if max_code == 8:
        return INCOME_LABELS_2015, "2015 (INCOME2, 1-8)"
    if max_code == 11:
        return INCOME_LABELS_2021, "2021 (INCOME3, 1-11)"
    if max_code == 7:
        return INCOME_LABELS_2023, "2023 (_INCOMG1, 1-7)"
    return INCOME_LABELS_2021, f"2021 (default, max={max_code})"


def _compute_group_metrics(df, group_col, y_true_col="y_true", y_pred_col="y_pred"):
    """
    Compute TPR, FPR, PPR (predicted positive rate), n per group.

    Returns DataFrame with columns:
        group_value, n, n_positive, TPR, FPR, PPR
    """
    results = []
    for group_val, gdf in df.groupby(group_col):
        yt = gdf[y_true_col]
        yp = gdf[y_pred_col]

        n      = len(gdf)
        n_pos  = int(yt.sum())
        n_neg  = n - n_pos

        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())

        tpr = tp / n_pos  if n_pos > 0 else 0.0
        fpr = fp / n_neg  if n_neg > 0 else 0.0
        ppr = (tp + fp) / n  # predicted positive rate

        results.append({
            "group_value": group_val,
            "n":           n,
            "n_positive":  n_pos,
            "TPR":         round(tpr, 4),
            "FPR":         round(fpr, 4),
            "PPR":         round(ppr, 4),
        })

    return pd.DataFrame(results)


def _equalized_odds_summary(metrics_df, attribute_name, label_map=None):
    """
    Compute Equalized Odds violation from per-group metrics.

    Returns dict with:
        delta_tpr, delta_fpr, eo_violation, dp_violation,
        worst_tpr_group, best_tpr_group
    """
    if label_map:
        metrics_df = metrics_df.copy()
        metrics_df["group_label"] = metrics_df["group_value"].map(
            lambda v: label_map.get(v, str(v))
        )
    else:
        metrics_df["group_label"] = metrics_df["group_value"].astype(str)

    tpr_vals = metrics_df["TPR"]
    fpr_vals = metrics_df["FPR"]
    ppr_vals = metrics_df["PPR"]

    delta_tpr = round(tpr_vals.max() - tpr_vals.min(), 4)
    delta_fpr = round(fpr_vals.max() - fpr_vals.min(), 4)
    delta_ppr = round(ppr_vals.max() - ppr_vals.min(), 4)

    worst_tpr = metrics_df.loc[metrics_df["TPR"].idxmin(), "group_label"]
    best_tpr  = metrics_df.loc[metrics_df["TPR"].idxmax(), "group_label"]
    worst_fpr = metrics_df.loc[metrics_df["FPR"].idxmax(), "group_label"]

    # Equalized Odds violation = max(dTPR, dFPR)
    eo_violation = max(delta_tpr, delta_fpr)

    severity = (
        "severe"     if eo_violation >= 0.20 else
        "concerning" if eo_violation >= 0.10 else
        "acceptable" if eo_violation >= 0.05 else
        "good"
    )

    return {
        "attribute":       attribute_name,
        "delta_tpr":       delta_tpr,
        "delta_fpr":       delta_fpr,
        "delta_ppr":       delta_ppr,
        "eo_violation":    round(eo_violation, 4),
        "dp_violation":    delta_ppr,
        "severity":        severity,
        "worst_tpr_group": worst_tpr,
        "best_tpr_group":  best_tpr,
        "worst_fpr_group": worst_fpr,
        "detail_df":       metrics_df[["group_label","n","n_positive","TPR","FPR","PPR"]],
    }


def run_fairness_analysis(model, X_test, y_test, output_dir, model_name="model"):
    """
    Run full Equalized Odds fairness analysis.

    Analyzes 3 demographic attributes: Age, Sex, Income.

    Income schema is auto-detected from the data (BRFSS 2015/2021/2023).

    Outputs:
        fairness_equalized_odds_summary.csv  — summary per attribute
        fairness_age_detail.csv              — per-group metrics for Age
        fairness_sex_detail.csv              — per-group metrics for Sex
        fairness_income_detail.csv           — per-group metrics for Income

    Parameters
    ----------
    model      : fitted sklearn Pipeline
    X_test     : test features (DataFrame, must contain Age, Sex, Income columns)
    y_test     : true labels
    output_dir : Path
    model_name : str prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Running Equalized Odds fairness analysis for: {model_name}")

    # ── Predictions ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_test).ravel()

    # Build analysis DataFrame with original features
    df = X_test.copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    # ── Auto-detect Income schema if Income column exists ────────────────────
    if "Income" in df.columns:
        income_label_map, income_schema_desc = _detect_income_schema(df["Income"])
        logging.info(f"  Income schema detected: BRFSS {income_schema_desc}")
    else:
        income_label_map, income_schema_desc = INCOME_LABELS_2021, "2021 (default)"

    # ── Per-attribute analysis ────────────────────────────────────────────────
    attributes = [
        ("Age",    AGE_LABELS),
        ("Sex",    SEX_LABELS),
        ("Income", income_label_map),
    ]

    summaries = []
    detail_paths = {}

    for attr, label_map in attributes:
        if attr not in df.columns:
            logging.warning(f"  Column '{attr}' not found in X_test — skipping")
            continue

        metrics = _compute_group_metrics(df, group_col=attr)
        result  = _equalized_odds_summary(metrics, attr, label_map)

        # Save detail CSV
        detail_path = output_dir / f"{model_name}_fairness_{attr.lower()}_detail.csv"
        result["detail_df"].to_csv(detail_path, index=False)
        detail_paths[attr] = detail_path
        logging.info(f"  Saved: {detail_path}")

        summaries.append({
            "attribute":       result["attribute"],
            "delta_tpr":       result["delta_tpr"],
            "delta_fpr":       result["delta_fpr"],
            "delta_ppr":       result["delta_ppr"],
            "eo_violation":    result["eo_violation"],
            "severity":        result["severity"],
            "worst_tpr_group": result["worst_tpr_group"],
            "best_tpr_group":  result["best_tpr_group"],
            "worst_fpr_group": result["worst_fpr_group"],
        })

        logging.info(
            f"  {attr}: dTPR={result['delta_tpr']:.3f}  "
            f"dFPR={result['delta_fpr']:.3f}  "
            f"EO={result['eo_violation']:.3f}  "
            f"[{result['severity']}]"
        )

    # ── Summary CSV ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / f"{model_name}_fairness_equalized_odds_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"  Saved summary: {summary_path}")

    logging.info("Equalized Odds fairness analysis completed")

    return {
        "summary_df":      summary_df,
        "summary_path":    str(summary_path),
        "detail_paths":    {k: str(v) for k,v in detail_paths.items()},
        "income_schema":   income_schema_desc,
    }