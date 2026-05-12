"""
Cross-cohort drift visualization for BRFSS analysis.

Produces a small-multiples grid (one panel per feature) showing the
summary statistic across all cohorts, with anomalies highlighted by
modified z-score. Plus a single-panel target prevalence chart.

Reads the cohort_distribution.csv and anomalies.csv produced by
run_anomaly_detection.py, and the cohort summaries from each
outputs/<slug>/eda/ folder.

Inputs:
    outputs/anomaly_detection/cohort_distribution.csv
    outputs/anomaly_detection/anomalies.csv
    outputs/<slug>/eda/cohort_summary.csv  (per cohort)

Outputs (under outputs/drift_visualization/):
    drift_grid.png                    — small-multiples grid
    target_prevalence_by_cohort.png   — single-line target chart

Usage:
    python -m src.analysis.run_drift_visualization
    # or:
    python src/analysis/run_drift_visualization.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


# ─── Path resolution ──────────────────────────────────────────────────────────

def find_project_root(start: Optional[Path] = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "outputs").is_dir() and (candidate / "data").is_dir():
            return candidate
    return p


PROJECT_ROOT = find_project_root(
    Path(__file__).resolve() if "__file__" in dir() else None
)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANOMALY_DIR = OUTPUTS_DIR / "anomaly_detection"
DRIFT_DIR = OUTPUTS_DIR / "drift_visualization"


# ─── Discovery ────────────────────────────────────────────────────────────────

def discover_dataset_folders() -> List[Path]:
    """Cohort folders with eda/cohort_summary.csv. Sorted by name."""
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

def load_anomalies() -> pd.DataFrame:
    """Anomaly CSV from run_anomaly_detection.py. Empty if absent."""
    path = ANOMALY_DIR / "anomalies.csv"
    if not path.exists():
        logging.warning(f"Anomalies file missing: {path}")
        logging.warning("  Run run_anomaly_detection.py first for "
                        "anomaly highlighting in plots.")
        return pd.DataFrame(columns=["feature", "cohort"])
    return pd.read_csv(path)


def load_cohort_distribution() -> pd.DataFrame:
    """Distribution CSV from run_anomaly_detection.py."""
    path = ANOMALY_DIR / "cohort_distribution.csv"
    if not path.exists():
        logging.error(f"Distribution file missing: {path}")
        logging.error("  Run run_anomaly_detection.py first.")
        sys.exit(1)
    return pd.read_csv(path)


def load_target_prevalence_per_cohort(folders: List[Path]) -> Dict[str, float]:
    """Single-row cohort_summary.csv has target_prevalence column."""
    out: Dict[str, float] = {}
    for folder in folders:
        path = folder / "eda" / "cohort_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        prev = df.iloc[0].get("target_prevalence")
        if prev is None or pd.isna(prev):
            continue
        out[folder.name] = float(prev)
    return out


# ─── Numeric parsing of distribution table ────────────────────────────────────

def parse_distribution_to_numeric(
    dist_df: pd.DataFrame,
    cohort_names: List[str],
) -> pd.DataFrame:
    """Convert display strings ("13.93%", "28.42") back to floats.

    Returns a DataFrame indexed by feature name, with one column per
    cohort, plus a 'type' column.
    """
    rows = []
    for _, row in dist_df.iterrows():
        feature = row["feature"]
        col_type = row.get("type", "unknown")
        out_row = {"feature": feature, "type": col_type}
        for cohort in cohort_names:
            v = row.get(cohort)
            if v is None or v == "—" or pd.isna(v):
                out_row[cohort] = np.nan
                continue
            v_str = str(v).strip()
            if v_str.endswith("%"):
                try:
                    out_row[cohort] = float(v_str[:-1]) / 100.0
                except ValueError:
                    out_row[cohort] = np.nan
            else:
                try:
                    out_row[cohort] = float(v_str)
                except ValueError:
                    out_row[cohort] = np.nan
        rows.append(out_row)

    return pd.DataFrame(rows).set_index("feature")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_drift_grid(
    numeric_dist: pd.DataFrame,
    anomalies: pd.DataFrame,
    cohort_names: List[str],
    out_path: Path,
) -> None:
    """Small-multiples grid: 1 panel per feature, lines across cohorts.

    A red ring marks (feature, cohort) pairs flagged in anomalies.csv.
    Binary features render on percent y-axis; others render raw scale.
    """
    features = list(numeric_dist.index)
    n = len(features)
    ncols = min(5, max(2, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.array(axes).flatten() if nrows * ncols > 1 else np.array([axes])

    # Build a fast lookup of anomalies: (feature, cohort) -> mz
    anomaly_lookup = {}
    if not anomalies.empty:
        for _, r in anomalies.iterrows():
            anomaly_lookup[(r["feature"], r["cohort"])] = r["modified_z_score"]

    # x-axis: positional index for cohorts (cohort names go on tick labels)
    xs = list(range(len(cohort_names)))

    for i, feature in enumerate(features):
        ax = axes[i]
        col_type = numeric_dist.loc[feature, "type"]
        is_binary = col_type == "binary"

        ys_raw = [numeric_dist.loc[feature, c] for c in cohort_names]
        ys_plot = [v * 100 if is_binary and not pd.isna(v) else v
                   for v in ys_raw]

        ax.plot(xs, ys_plot, marker="o", linewidth=1.5, color="#1f77b4")
        ax.set_title(feature, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(xs)
        ax.set_xticklabels(_short_cohort_labels(cohort_names),
                           rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

        # Anomaly highlight
        for j, cohort in enumerate(cohort_names):
            if (feature, cohort) in anomaly_lookup and not pd.isna(ys_plot[j]):
                ax.scatter([xs[j]], [ys_plot[j]], s=120, facecolor="none",
                           edgecolor="red", linewidth=2, zorder=5)

        if is_binary:
            ax.set_ylabel("%", fontsize=7)
        else:
            ax.set_ylabel("mean", fontsize=7)

    # Hide unused panels
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Cross-cohort drift per feature  "
        "(red circle = modified-z anomaly)",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {out_path}")


def plot_target_prevalence(
    target_prev: Dict[str, float],
    out_path: Path,
) -> None:
    """Single-panel line chart of target prevalence across cohorts."""
    cohorts = list(target_prev.keys())
    vals = [target_prev[c] * 100 for c in cohorts]
    xs = list(range(len(cohorts)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, vals, marker="o", linewidth=2, color="#d62728")
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.2f}%", xy=(x, v), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=10)

    ax.set_title("Target prevalence by cohort",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Prevalence (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels(_short_cohort_labels(cohorts),
                       rotation=30, ha="right")
    ax.grid(True, alpha=0.3)

    if vals:
        margin = max(0.5, (max(vals) - min(vals)) * 0.3)
        ax.set_ylim(min(vals) - margin, max(vals) + margin)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {out_path}")


def _short_cohort_labels(cohorts: List[str]) -> List[str]:
    """Drop common prefix and trailing cdc_brfss_diabetes_ for plot legibility.

    'cdc_brfss_diabetes_2015' -> '2015'
    'cdc_brfss_diabetes_2021' -> '2021'
    """
    labels = []
    for c in cohorts:
        s = c.replace("cdc_brfss_diabetes_", "")
        labels.append(s)
    return labels


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    folders = discover_dataset_folders()
    if len(folders) < 2:
        logging.error(
            f"Need >=2 cohort folders. Found {len(folders)}."
        )
        sys.exit(1)

    cohort_names = [f.name for f in folders]
    logging.info(f"Cohorts: {cohort_names}")

    # Load anomaly outputs
    dist_df = load_cohort_distribution()
    anomalies = load_anomalies()
    target_prev = load_target_prevalence_per_cohort(folders)

    # Convert display strings back to numeric for plotting
    numeric = parse_distribution_to_numeric(dist_df, cohort_names)

    # Drift grid
    plot_drift_grid(numeric, anomalies, cohort_names,
                    DRIFT_DIR / "drift_grid.png")

    # Target prevalence
    if target_prev:
        plot_target_prevalence(target_prev,
                               DRIFT_DIR / "target_prevalence_by_cohort.png")
    else:
        logging.warning("No target_prevalence values found in cohort summaries; "
                        "skipping target prevalence chart.")

    logging.info("Done.")


if __name__ == "__main__":
    main()