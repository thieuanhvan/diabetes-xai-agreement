"""
BRFSS-specific pandemic 3-year overlay analysis.

Domain-specific narrative on top of the generic comparative outputs.
Maps the 3 BRFSS datasets to pandemic phases (pre/mid/post) and produces
narrative figures for the manuscript Chapter 4 EDA.

Generic comparative outputs (run_eda_comparative.py) treat datasets equally.
This overlay adds:
  - Phase ordering and pandemic-style colors
  - Acceleration ratio annotation (post-pandemic vs pre-pandemic rate)
  - Diverging trends auto-detection (predictor declining vs target rising)
  - Schema heterogeneity highlights with phase coloring

If your project has no temporal/phase narrative, skip this file and use
only run_eda_comparative.py.

Inputs:
    outputs/eda_comparative/cohort_comparison.csv
    outputs/eda_comparative/binary_features_pct_comparison.csv
    outputs/eda_comparative/feature_availability_matrix.csv

Outputs (under outputs/eda_pandemic_3year/):
    pandemic_diabetes_acceleration.png  — annotated +Δpp narrative
    pandemic_diverging_trends.png        — predictors moving opposite to target
    pandemic_schema_changes.png          — features added/removed across phases

Configuration:
    PHASE_MAP below. Edit if your dataset slugs differ.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)s | %(message)s")


# ─── BRFSS 3-year domain configuration ────────────────────────────────────────

# Maps dataset slug → (calendar year, phase label, color).
# Edit if you rename datasets or add years.
PHASE_MAP = {
    "cdc_brfss_2015_rebuilt": (2015, "Pre-pandemic",  "#3498db"),
    "cdc_brfss_2021_rebuilt": (2021, "Mid-pandemic",  "#e67e22"),
    "cdc_brfss_2023_rebuilt": (2023, "Post-acute",    "#c0392b"),
}


# ─── Path resolution ──────────────────────────────────────────────────────────

def find_project_root(start: Path = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "outputs").is_dir() and (candidate / "data").is_dir():
            return candidate
    return p


PROJECT_ROOT    = find_project_root(Path(__file__).resolve() if "__file__" in dir() else None)
COMPARATIVE_DIR = PROJECT_ROOT / "outputs" / "eda_comparative"
PANDEMIC_DIR    = PROJECT_ROOT / "outputs" / "eda_pandemic_3year"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def order_phases(slugs):
    """Sort dataset slugs by calendar year using PHASE_MAP."""
    known = [s for s in slugs if s in PHASE_MAP]
    return sorted(known, key=lambda s: PHASE_MAP[s][0])


def get_phase_label(slug: str) -> str:
    if slug not in PHASE_MAP:
        return slug
    year, phase, _ = PHASE_MAP[slug]
    return f"{year} ({phase})"


def get_color(slug: str) -> str:
    return PHASE_MAP.get(slug, (None, None, "#888"))[2]


# ─── Figure 1: Diabetes prevalence acceleration ──────────────────────────────

def fig_diabetes_acceleration(cohort_df: pd.DataFrame, output_dir: Path):
    """Annotated bar chart with acceleration ratio between phases."""
    df = cohort_df[cohort_df["dataset_slug"].isin(PHASE_MAP)].copy()
    if df.empty or len(df) < 2:
        return
    df["year"] = df["dataset_slug"].map(lambda s: PHASE_MAP[s][0])
    df = df.sort_values("year").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    prev = (df["target_prevalence"] * 100).values
    labels = [get_phase_label(s) for s in df["dataset_slug"]]
    colors = [get_color(s) for s in df["dataset_slug"]]

    bars = ax.bar(labels, prev, color=colors, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, prev):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.4f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    # Annotate increments + acceleration ratio
    deltas, gaps = [], []
    for i in range(len(prev) - 1):
        deltas.append(prev[i + 1] - prev[i])
        gaps.append(int(df.iloc[i + 1]["year"] - df.iloc[i]["year"]))

    for i, (d, g) in enumerate(zip(deltas, gaps)):
        color = "#c0392b" if i == len(deltas) - 1 else "#555"
        weight = "bold" if i == len(deltas) - 1 else "normal"
        ax.annotate(f"+{d:.2f}pp\n({g} years)",
                     xy=(i + 0.5, max(prev) + 0.3),
                     ha="center", fontsize=10, color=color, fontweight=weight)

    if len(deltas) >= 2 and gaps[0] > 0 and gaps[-1] > 0:
        rate_pre  = deltas[0]  / gaps[0]
        rate_post = deltas[-1] / gaps[-1]
        if rate_pre > 0:
            accel = rate_post / rate_pre
            ax.text(0.5, 0.97,
                     f"Acceleration: {accel:.1f}x faster post-pandemic",
                     transform=ax.transAxes, ha="center", fontsize=11,
                     color="#c0392b", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4",
                                facecolor="#ffe6e6", edgecolor="#c0392b"))

    ax.set_ylabel("Diabetes prevalence (%)", fontsize=12)
    ax.set_ylim(0, max(prev) + 1.8)
    ax.set_title("Diabetes prevalence: Pandemic acceleration narrative",
                  fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "pandemic_diabetes_acceleration.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Saved: {out_path.name}")


# ─── Figure 2: Diverging trends ──────────────────────────────────────────────

def fig_diverging_trends(cohort_df: pd.DataFrame, binary_df: pd.DataFrame,
                           output_dir: Path, change_threshold: float = 2.0):
    """Find predictors moving opposite to target."""
    target_df = cohort_df[cohort_df["dataset_slug"].isin(PHASE_MAP)].copy()
    if len(target_df) < 2:
        return
    target_df["year"] = target_df["dataset_slug"].map(lambda s: PHASE_MAP[s][0])
    target_df = target_df.sort_values("year").reset_index(drop=True)

    target_pcts = (target_df["target_prevalence"] * 100).values
    target_change = target_pcts[-1] - target_pcts[0]
    years = target_df["year"].values

    if binary_df.empty:
        return

    ordered_slugs = list(target_df["dataset_slug"])
    ordered_slugs = [s for s in ordered_slugs if s in binary_df.columns]
    if len(ordered_slugs) < 2:
        return

    changes = []
    for _, row in binary_df.iterrows():
        col = row["column"]
        values = [row[s] for s in ordered_slugs if not pd.isna(row[s])]
        if len(values) < 2:
            continue
        change = values[-1] - values[0]
        diverges = (change * target_change) < 0
        if abs(change) >= change_threshold:
            changes.append({
                "column": col,
                "values": [row[s] for s in ordered_slugs],
                "change": change,
                "abs_change": abs(change),
                "diverges": diverges,
            })

    if not changes:
        logging.info(f"  No binary predictors with |change| >= {change_threshold}pp")
        return

    changes.sort(key=lambda x: -x["abs_change"])
    top_n = min(5, len(changes))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, target_pcts, marker="s", linewidth=3, markersize=14,
             color="#c0392b",
             label=f"Diabetes (target, Δ{target_change:+.2f}pp)", zorder=10)
    for x, y in zip(years, target_pcts):
        ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                     xytext=(0, -20), ha="center", fontsize=10,
                     color="#c0392b", fontweight="bold")

    palette = ["#3498db", "#16a085", "#8e44ad", "#f39c12", "#27ae60"]
    for i, ch in enumerate(changes[:top_n]):
        col = palette[i % len(palette)]
        marker = "v" if ch["diverges"] else "o"
        suffix = " ⚠ diverges" if ch["diverges"] else ""
        ax.plot(years, ch["values"], marker=marker, linewidth=2, markersize=10,
                 color=col, label=f"{ch['column']} (Δ{ch['change']:+.1f}pp){suffix}")
        for x, y in zip(years, ch["values"]):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, color=col)

    ax.set_xticks(years)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Prevalence (%)", fontsize=12)
    ax.set_title(f"Diverging trends: target vs predictors (|Δ| >= {change_threshold}pp)",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="center right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "pandemic_diverging_trends.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Saved: {out_path.name}")


# ─── Figure 3: Schema changes ────────────────────────────────────────────────

def fig_schema_changes(matrix_df: pd.DataFrame, output_dir: Path):
    pandemic_slugs = [s for s in matrix_df.columns
                       if s != "column" and s in PHASE_MAP]
    if len(pandemic_slugs) < 2:
        return
    pandemic_slugs = order_phases(pandemic_slugs)

    schema_diff_rows = []
    for _, row in matrix_df.iterrows():
        col = row["column"]
        present = [s for s in pandemic_slugs if row[s] != "(absent)"]
        if 0 < len(present) < len(pandemic_slugs):
            schema_diff_rows.append({"column": col, "present_in": present})

    if not schema_diff_rows:
        logging.info("  No schema differences across pandemic phases")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(schema_diff_rows))))
    n_phases = len(pandemic_slugs)
    for i, item in enumerate(schema_diff_rows):
        for j, slug in enumerate(pandemic_slugs):
            present = slug in item["present_in"]
            color = get_color(slug) if present else "#dddddd"
            ax.barh(i, 1, left=j, color=color, edgecolor="black", linewidth=0.8)
            label = "✓" if present else "✗"
            ax.text(j + 0.5, i, label, ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if present else "#888")

    ax.set_yticks(range(len(schema_diff_rows)))
    ax.set_yticklabels([item["column"] for item in schema_diff_rows], fontsize=11)
    ax.set_xticks([j + 0.5 for j in range(n_phases)])
    ax.set_xticklabels([get_phase_label(s) for s in pandemic_slugs],
                        fontsize=11, rotation=20, ha="right")
    ax.set_xlim(-0.05, n_phases + 0.05)
    ax.invert_yaxis()
    ax.set_title("Schema changes across pandemic phases",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "pandemic_schema_changes.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Saved: {out_path.name}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    logging.info("=" * 60)
    logging.info("BRFSS pandemic 3-year overlay analysis")
    logging.info("=" * 60)
    logging.info(f"Project root: {PROJECT_ROOT}")
    logging.info("")

    if not COMPARATIVE_DIR.exists():
        logging.error(f"Comparative dir not found: {COMPARATIVE_DIR}")
        logging.error("Run run_eda_comparative.py first.")
        sys.exit(1)

    PANDEMIC_DIR.mkdir(parents=True, exist_ok=True)

    cohort_path = COMPARATIVE_DIR / "cohort_comparison.csv"
    binary_path = COMPARATIVE_DIR / "binary_features_pct_comparison.csv"
    matrix_path = COMPARATIVE_DIR / "feature_availability_matrix.csv"

    cohort_df = pd.read_csv(cohort_path) if cohort_path.exists() else pd.DataFrame()
    binary_df = pd.read_csv(binary_path) if binary_path.exists() else pd.DataFrame()
    matrix_df = pd.read_csv(matrix_path) if matrix_path.exists() else pd.DataFrame()

    logging.info(f"Loaded cohort_comparison: {len(cohort_df)} rows")
    logging.info(f"Loaded binary_features: {len(binary_df)} rows")
    logging.info(f"Loaded feature_matrix: {len(matrix_df)} rows")
    logging.info("")

    if not cohort_df.empty:
        pandemic_present = set(cohort_df["dataset_slug"]) & set(PHASE_MAP.keys())
        if len(pandemic_present) < 2:
            logging.warning(f"Only {len(pandemic_present)} pandemic dataset(s) found.")
            logging.warning(f"Need at least 2. Edit PHASE_MAP if your slugs differ.")
            return
        logging.info(f"Pandemic datasets discovered: {sorted(pandemic_present)}")
        logging.info("")

    logging.info("Generating pandemic-narrative figures...")
    fig_diabetes_acceleration(cohort_df, PANDEMIC_DIR)
    fig_diverging_trends(cohort_df, binary_df, PANDEMIC_DIR)
    fig_schema_changes(matrix_df, PANDEMIC_DIR)

    logging.info("")
    logging.info("=" * 60)
    n_files = len(list(PANDEMIC_DIR.iterdir()))
    logging.info(f"DONE. {n_files} figures in {PANDEMIC_DIR}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()