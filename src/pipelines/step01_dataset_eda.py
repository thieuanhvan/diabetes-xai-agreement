"""
STEP 01 - Dataset EDA (generic, signature-compatible)

Generic EDA without hardcoding column names. Replaces the previous version
that only plotted 5 hardcoded columns.

Drop-in replacement: keeps the same call signature run_step01_dataset_eda(df,
outputs_dir) used by run_pipeline.py, so no changes to run_pipeline.py
required.

What's new:
    - Loops ALL columns and detects type (binary/ordinal/continuous)
    - Generates appropriate plot per column type
    - Saves CSV alongside each PNG (data behind every figure)
    - Writes cohort_summary.csv      (consumed by run_eda_comparative.py)
    - Writes column_inventory.csv    (consumed by run_eda_comparative.py)
    - Writes correlation_with_target.csv (consumed by run_eda_comparative.py)
    - Auto-detects dataset slug from outputs_dir.name (e.g.
      'outputs/cdc_brfss_2015_rebuilt' → slug = 'cdc_brfss_2015_rebuilt')

Public API (unchanged):
    run_step01_dataset_eda(df, outputs_dir)

Outputs (under outputs_dir/eda/):
    cohort_summary.csv              ← single-row summary for comparative
    column_inventory.csv            ← all columns + detected types
    class_distribution.png/.csv     ← target balance
    correlation_heatmap.png/.csv    ← Pearson matrix
    correlation_with_target.csv     ← sorted by abs(r), KEY OUTPUT
    distributions/<col>.png/.csv    ← one pair per non-target column
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# ─── Column type detection ────────────────────────────────────────────────────

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Try to detect target column automatically."""
    candidates = [
        "Diabetes_binary",
        "diabetes_binary",
        "Outcome",
        "target",
        "label",
        "class",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def detect_column_type(series: pd.Series, max_categories: int = 20) -> str:
    """Auto-detect: 'binary', 'ordinal', 'continuous', 'categorical', 'unknown'."""
    s = series.dropna()
    if len(s) == 0:
        return "unknown"
    n_unique = s.nunique()
    if n_unique <= 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        is_int_like = (s == s.astype(int)).all() if s.dtype.kind in "fc" else True
        if is_int_like and n_unique <= max_categories:
            return "ordinal"
        return "continuous"
    if n_unique <= max_categories:
        return "categorical"
    return "unknown"


def _column_stats(series: pd.Series) -> dict:
    """One-row stats summary."""
    s = series.dropna()
    out = {
        "n":             len(series),
        "n_missing":     int(series.isna().sum()),
        "n_unique":      int(s.nunique()) if len(s) > 0 else 0,
        "detected_type": detect_column_type(series),
    }
    if pd.api.types.is_numeric_dtype(s) and len(s) > 0:
        out.update({
            "mean":   round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std":    round(float(s.std()), 4),
            "min":    float(s.min()),
            "max":    float(s.max()),
        })
    return out


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _plot_binary(series: pd.Series, title: str, png_path: Path):
    counts = series.value_counts().sort_index()
    pcts = (counts / counts.sum() * 100).round(2)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Van changed on 2026-05-01
    #bars = ax.bar(counts.index.astype(str), counts.values,
    #               color=["#3498db", "#e67e22"], edgecolor="black")

    positions = np.arange(len(counts))
    bars = ax.bar(positions, counts.values,
                  color=["#3498db", "#e67e22"], edgecolor="black")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in counts.index])

    for bar, c, p in zip(bars, counts.values, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c:,}\n({p:.2f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel(series.name, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame({"value": counts.index, "count": counts.values,
                   "percent": pcts.values}).to_csv(png_path.with_suffix(".csv"),
                                                     index=False)


def _plot_ordinal(series: pd.Series, title: str, png_path: Path):
    counts = series.value_counts().sort_index()
    pcts = (counts / counts.sum() * 100).round(2)

    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(counts)), 4.5))

    # Van changed on 2026-05-01
    # bars = ax.bar(counts.index.astype(str), counts.values,
    #               color="#3498db", edgecolor="black", linewidth=0.7)
    positions = np.arange(len(counts))
    bars = ax.bar(positions, counts.values,
                  color="#3498db", edgecolor="black", linewidth=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in counts.index])

    for bar, p in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{p:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel(series.name, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45 if len(counts) > 8 else 0)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame({"value": counts.index, "count": counts.values,
                   "percent": pcts.values}).to_csv(png_path.with_suffix(".csv"),
                                                     index=False)


def _plot_continuous(series: pd.Series, title: str, png_path: Path):
    s = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(s, bins=50, color="#3498db", edgecolor="black",
                  linewidth=0.5, alpha=0.85)
    axes[0].axvline(s.mean(), color="red", linestyle="--", linewidth=2,
                     label=f"Mean = {s.mean():.2f}")
    axes[0].axvline(s.median(), color="orange", linestyle="--", linewidth=2,
                     label=f"Median = {s.median():.2f}")
    axes[0].set_xlabel(series.name, fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Distribution", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(s, vert=False)
    axes[1].set_xlabel(series.name, fontsize=11)
    axes[1].set_title("Boxplot", fontsize=11, fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    q = s.quantile([0, 0.25, 0.5, 0.75, 1.0])
    pd.DataFrame({
        "stat":  ["min", "p25", "median", "p75", "max", "mean", "std"],
        "value": [float(q[0]), float(q[0.25]), float(q[0.5]),
                  float(q[0.75]), float(q[1.0]), float(s.mean()), float(s.std())]
    }).to_csv(png_path.with_suffix(".csv"), index=False)


# ─── Top-level outputs ────────────────────────────────────────────────────────

def _plot_correlation_heatmap(df: pd.DataFrame, png_path: Path):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    side = min(20, max(8, 0.6 * len(corr)))
    fig, ax = plt.subplots(figsize=(side, side * 0.9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax,
                annot_kws={"fontsize": 7})
    ax.set_title("Pearson correlation matrix", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    corr.round(4).to_csv(png_path.with_suffix(".csv"))


def _write_correlation_with_target(df: pd.DataFrame, target_col: str,
                                     csv_path: Path):
    numeric = df.select_dtypes(include=[np.number])
    if target_col not in numeric.columns:
        return
    corrs = numeric.corr()[target_col].drop(target_col)
    pd.DataFrame({
        "feature":   corrs.index,
        "pearson_r": corrs.values.round(4),
        "abs_r":     np.abs(corrs.values).round(4),
    }).sort_values("abs_r", ascending=False).reset_index(drop=True).to_csv(
        csv_path, index=False)


def _write_cohort_summary(df: pd.DataFrame, target_col: Optional[str],
                            dataset_slug: str, csv_path: Path):
    """One-row summary used by run_eda_comparative.py."""
    n = len(df)
    target_pos = int((df[target_col] == 1).sum()) if (target_col and target_col in df.columns) else 0
    summary = {
        "dataset_slug":      dataset_slug,
        "n_rows":            n,
        "n_columns":         len(df.columns),
        "target_col":        target_col or "",
        "target_positive":   target_pos,
        "target_negative":   n - target_pos,
        "target_prevalence": round(target_pos / n, 6) if n > 0 else 0.0,
        "n_missing_total":   int(df.isna().sum().sum()),
    }

    # Quick per-column headline stats
    for col in df.columns:
        if col == target_col:
            continue
        ctype = detect_column_type(df[col])
        if ctype == "binary":
            summary[f"{col}__pos_pct"] = round(
                float((df[col] == 1).sum() / n * 100), 4)
        elif ctype == "continuous":
            summary[f"{col}__mean"]   = round(float(df[col].mean()), 4)
            summary[f"{col}__median"] = round(float(df[col].median()), 4)

    pd.DataFrame([summary]).to_csv(csv_path, index=False)


def _write_column_inventory(df: pd.DataFrame, csv_path: Path):
    rows = []
    for col in df.columns:
        s = _column_stats(df[col])
        s["column"] = col
        rows.append(s)
    inv = pd.DataFrame(rows)
    cols_order = ["column", "detected_type", "n", "n_missing", "n_unique"]
    extras = [c for c in inv.columns if c not in cols_order]
    inv[cols_order + extras].to_csv(csv_path, index=False)


# ─── Main EDA entry point (called from run_pipeline.py) ──────────────────────

def run_step01_dataset_eda(df, outputs_dir):
    """
    Generic EDA for the loaded dataset.

    Drop-in replacement: same signature as previous version. The dataset slug
    is auto-detected from outputs_dir.name (since get_outputs_dir() returns
    OUTPUT_DIR / ACTIVE_DATASET).

    Parameters
    ----------
    df : pandas.DataFrame
        The loaded dataset.
    outputs_dir : str or Path
        Project's outputs directory for this dataset, typically
        OUTPUT_DIR / ACTIVE_DATASET. Function writes to {outputs_dir}/eda/.
    """
    outputs_dir = Path(outputs_dir)
    eda_dir = outputs_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = eda_dir / "distributions"
    dist_dir.mkdir(exist_ok=True)

    # Auto-detect slug from outputs_dir.name
    dataset_slug = outputs_dir.name

    # Auto-detect target
    target_col = detect_target_column(df)

    print()
    print("STEP 01 - DATASET EDA")
    print("--------------------------------")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Detected target column: {target_col}")
    print(f"Dataset slug: {dataset_slug}")

    # 1) Cohort summary (single row, for cross-dataset comparison)
    _write_cohort_summary(df, target_col, dataset_slug,
                            eda_dir / "cohort_summary.csv")
    print(f"Saved: {eda_dir / 'cohort_summary.csv'}")

    # 2) Column inventory (all columns, detected types)
    _write_column_inventory(df, eda_dir / "column_inventory.csv")
    print(f"Saved: {eda_dir / 'column_inventory.csv'}")

    # 3) Target class distribution
    if target_col and target_col in df.columns:
        _plot_binary(df[target_col],
                       f"Target distribution: {target_col}",
                       eda_dir / "class_distribution.png")
        print(f"Saved: {eda_dir / 'class_distribution.png'}")

    # 4) Correlation heatmap
    _plot_correlation_heatmap(df, eda_dir / "correlation_heatmap.png")
    print(f"Saved: {eda_dir / 'correlation_heatmap.png'}")

    # 5) Correlation with target
    if target_col and target_col in df.columns:
        _write_correlation_with_target(df, target_col,
                                         eda_dir / "correlation_with_target.csv")
        print(f"Saved: {eda_dir / 'correlation_with_target.csv'}")

    # 6) Per-column distributions (loop ALL columns except target)
    n_plotted = 0
    for col in df.columns:
        if col == target_col:
            continue
        ctype = detect_column_type(df[col])
        title = f"{col} ({ctype})"
        png_path = dist_dir / f"{col}.png"
        try:
            if ctype == "binary":
                _plot_binary(df[col], title, png_path)
            elif ctype == "ordinal":
                _plot_ordinal(df[col], title, png_path)
            elif ctype == "continuous":
                _plot_continuous(df[col], title, png_path)
            else:
                continue
            n_plotted += 1
        except Exception as e:
            logger.warning(f"Failed to plot {col}: {e}")

    print(f"Saved: {n_plotted} per-column distribution plots in {dist_dir}")
    print()
    print("EDA completed")
    print()