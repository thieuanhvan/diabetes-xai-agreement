"""
Generic cross-dataset EDA comparative analysis.

Discovers dataset folders under outputs/ and aggregates per-dataset EDA
artifacts (produced by step01) into cross-dataset comparison tables and
figures.

Mirrors the architecture of run_xai_agreement.py:
  - Discovers folders by checking for outputs/<slug>/eda/cohort_summary.csv
  - Works with ANY number of datasets (2, 3, or more)
  - Schema-agnostic: column names, types, and counts are read from the data

Outputs (under outputs/eda_comparative/):
    cohort_comparison.csv             — all cohorts side-by-side
    target_prevalence_comparison.png  — target prevalence across datasets
    feature_availability_matrix.csv   — which columns appear in which datasets
    schema_diff.csv                   — columns present in some but not all
    binary_features_pct_comparison.csv — pos % per (col, dataset)
    binary_features_comparison.png    — grouped bar chart of binary features
    target_correlation_comparison.csv — corr(col, target) across datasets

Usage:
    python -m src.analysis.run_eda_comparative
    # or:
    python src/analysis/run_eda_comparative.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)s | %(message)s")


# ─── Path resolution (mirrors run_xai_agreement.py) ──────────────────────────

def find_project_root(start: Path = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "outputs").is_dir() and (candidate / "data").is_dir():
            return candidate
    return p


PROJECT_ROOT    = find_project_root(Path(__file__).resolve() if "__file__" in dir() else None)
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
COMPARATIVE_DIR = OUTPUTS_DIR / "eda_comparative"


# ─── Discovery ────────────────────────────────────────────────────────────────

def discover_dataset_folders() -> List[Path]:
    """
    Discover dataset output folders containing EDA artifacts.

    A folder qualifies if it has outputs/<slug>/eda/cohort_summary.csv.
    Skips outputs/eda_comparative/ and outputs/xai_agreement/ etc.
    """
    if not OUTPUTS_DIR.exists():
        logging.error(f"Outputs directory not found: {OUTPUTS_DIR}")
        sys.exit(1)

    folders = []
    for item in OUTPUTS_DIR.iterdir():
        if not item.is_dir():
            continue
        if (item / "eda" / "cohort_summary.csv").exists():
            folders.append(item)

    return sorted(folders)


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_cohort_summaries(folders: List[Path]) -> pd.DataFrame:
    rows = []
    for folder in folders:
        path = folder / "eda" / "cohort_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "dataset_slug" not in df.columns or pd.isna(df["dataset_slug"].iloc[0]) \
                or df["dataset_slug"].iloc[0] == "":
            df["dataset_slug"] = folder.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_column_inventories(folders: List[Path]) -> Dict[str, pd.DataFrame]:
    out = {}
    for folder in folders:
        path = folder / "eda" / "column_inventory.csv"
        if path.exists():
            out[folder.name] = pd.read_csv(path)
    return out


def load_target_correlations(folders: List[Path]) -> Dict[str, pd.DataFrame]:
    out = {}
    for folder in folders:
        path = folder / "eda" / "correlation_with_target.csv"
        if path.exists():
            out[folder.name] = pd.read_csv(path)
    return out


def load_distribution_csv(folder: Path, col: str) -> pd.DataFrame:
    """Load a single column's distribution CSV (binary/ordinal value counts)."""
    path = folder / "eda" / "distributions" / f"{col}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ─── Comparative outputs ─────────────────────────────────────────────────────

def make_cohort_comparison(cohort_df: pd.DataFrame, output_dir: Path):
    if cohort_df.empty:
        return
    out_csv = output_dir / "cohort_comparison.csv"
    cohort_df.to_csv(out_csv, index=False)
    logging.info(f"  Saved: {out_csv.name}")

    # Plot target prevalence
    if "target_prevalence" in cohort_df.columns:
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(cohort_df)), 5))
        prev = cohort_df["target_prevalence"] * 100
        bars = ax.bar(cohort_df["dataset_slug"], prev,
                       color="#3498db", edgecolor="black", linewidth=1.2)
        for bar, p in zip(bars, prev):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{p:.4f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_ylabel("Target prevalence (%)", fontsize=12)
        ax.set_title("Target prevalence across datasets",
                      fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out_png = output_dir / "target_prevalence_comparison.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"  Saved: {out_png.name}")


def make_feature_availability_matrix(inventories: Dict[str, pd.DataFrame],
                                       output_dir: Path):
    if not inventories:
        return

    all_cols = sorted(set().union(*(set(inv["column"]) for inv in inventories.values())))
    matrix = pd.DataFrame(index=all_cols)
    for slug, inv in inventories.items():
        col_to_type = dict(zip(inv["column"], inv["detected_type"]))
        matrix[slug] = matrix.index.map(col_to_type)
    matrix.fillna("(absent)", inplace=True)
    matrix.index.name = "column"

    out_csv = output_dir / "feature_availability_matrix.csv"
    matrix.reset_index().to_csv(out_csv, index=False)
    logging.info(f"  Saved: {out_csv.name}")

    # Schema diff: features present in some but not all
    n = len(inventories)
    issues = []
    for col in all_cols:
        present = [slug for slug in inventories
                    if col in set(inventories[slug]["column"])]
        if 0 < len(present) < n:
            issues.append({
                "column": col,
                "n_present": len(present),
                "n_absent":  n - len(present),
                "present_in": ",".join(present),
                "absent_from": ",".join([s for s in inventories if s not in present]),
            })
    if issues:
        out_csv = output_dir / "schema_diff.csv"
        pd.DataFrame(issues).to_csv(out_csv, index=False)
        logging.info(f"  Saved: {out_csv.name} ({len(issues)} mismatches)")


def make_binary_comparison(folders: List[Path],
                             inventories: Dict[str, pd.DataFrame],
                             output_dir: Path):
    """Compare positive % of all binary features across datasets."""
    binary_cols = set()
    for inv in inventories.values():
        binary_cols |= set(inv[inv["detected_type"] == "binary"]["column"])

    if not binary_cols:
        return

    rows = []
    for col in sorted(binary_cols):
        row = {"column": col}
        for folder in folders:
            slug = folder.name
            dist = load_distribution_csv(folder, col)
            if dist.empty:
                row[slug] = np.nan
                continue
            pos_row = dist[dist["value"] == 1]
            row[slug] = float(pos_row["percent"].iloc[0]) if not pos_row.empty else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = output_dir / "binary_features_pct_comparison.csv"
    df.to_csv(out_csv, index=False)
    logging.info(f"  Saved: {out_csv.name}")

    # Grouped bar chart
    dataset_cols = [f.name for f in folders if f.name in df.columns]
    if not dataset_cols:
        return

    fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(df)), 6))
    x = np.arange(len(df))
    n_datasets = len(dataset_cols)
    width = 0.85 / n_datasets
    palette = plt.cm.tab10(np.linspace(0, 1, max(n_datasets, 3)))

    for i, slug in enumerate(dataset_cols):
        values = df[slug].fillna(0).values
        offset = (i - (n_datasets - 1) / 2) * width
        ax.bar(x + offset, values, width, label=slug,
               color=palette[i], edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["column"], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Positive class prevalence (%)", fontsize=12)
    ax.set_title("Binary feature prevalence across datasets",
                  fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = output_dir / "binary_features_comparison.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Saved: {out_png.name}")


def make_target_correlation_comparison(target_corrs: Dict[str, pd.DataFrame],
                                         output_dir: Path):
    if not target_corrs:
        return

    all_features = sorted(set().union(*(set(df["feature"]) for df in target_corrs.values())))
    rows = []
    for feat in all_features:
        row = {"feature": feat}
        for slug, df in target_corrs.items():
            match = df[df["feature"] == feat]
            row[slug] = float(match["pearson_r"].iloc[0]) if not match.empty else np.nan
        rows.append(row)

    df_out = pd.DataFrame(rows)
    dataset_cols = [c for c in df_out.columns if c != "feature"]
    df_out["mean_abs"] = df_out[dataset_cols].abs().mean(axis=1)
    df_out = df_out.sort_values("mean_abs", ascending=False).drop(columns=["mean_abs"])

    out_csv = output_dir / "target_correlation_comparison.csv"
    df_out.to_csv(out_csv, index=False)
    logging.info(f"  Saved: {out_csv.name}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    logging.info("=" * 60)
    logging.info("Generic cross-dataset EDA comparative analysis")
    logging.info("=" * 60)
    logging.info(f"Project root: {PROJECT_ROOT}")
    logging.info(f"Outputs dir:  {OUTPUTS_DIR}")
    logging.info("")

    folders = discover_dataset_folders()
    if not folders:
        logging.error("No dataset folders with EDA artifacts found.")
        logging.error("Run pipeline (step01 generic EDA) first.")
        sys.exit(1)

    logging.info(f"Discovered {len(folders)} dataset folder(s):")
    for f in folders:
        logging.info(f"  - {f.name}")
    logging.info("")

    COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading per-dataset artifacts...")
    cohort_df    = load_cohort_summaries(folders)
    inventories  = load_column_inventories(folders)
    target_corrs = load_target_correlations(folders)

    logging.info("")
    logging.info("Generating comparative outputs...")
    make_cohort_comparison(cohort_df, COMPARATIVE_DIR)
    make_feature_availability_matrix(inventories, COMPARATIVE_DIR)
    make_binary_comparison(folders, inventories, COMPARATIVE_DIR)
    make_target_correlation_comparison(target_corrs, COMPARATIVE_DIR)

    logging.info("")
    logging.info("=" * 60)
    n_files = len(list(COMPARATIVE_DIR.iterdir()))
    logging.info(f"DONE. {n_files} files in {COMPARATIVE_DIR}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()