"""
Anomaly detection across BRFSS cohorts using modified z-score.

Reads per-cohort EDA artifacts produced by step01_dataset_eda.py and
flags feature-cohort pairs whose summary statistic deviates strongly
from the median across the other cohorts. Detects encoding bugs
(e.g. CholCheck schema mismatch in 2017 development) and documents
genuine cross-cohort drift.

Schema-agnostic and cohort-count-agnostic — works with any number
of cohorts under outputs/ that have the eda/cohort_summary.csv
artifact.

Inputs (per cohort, produced by step01_dataset_eda.py):
    outputs/<slug>/eda/cohort_summary.csv
    outputs/<slug>/eda/column_inventory.csv
    outputs/<slug>/eda/distributions/<col>.csv

Outputs (under outputs/anomaly_detection/):
    cohort_distribution.csv       — feature × cohort summary stats
    sample_sizes.csv              — rows × cols × prevalence per cohort
    anomalies.csv                 — flagged (feature, cohort) pairs

Usage:
    python -m src.analysis.run_anomaly_detection
    # or:
    python src/analysis/run_anomaly_detection.py
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


# ─── Path resolution (mirrors run_eda_comparative.py) ────────────────────────

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


# ─── Configuration ────────────────────────────────────────────────────────────

# Modified z-score threshold for flagging. 3.5 is the textbook outlier
# cutoff (Iglewicz & Hoaglin 1993). Pipeline encoding bugs typically
# score |z| > 50; methodology changes 10–50; documented real-world
# drift 3.5–10.
ANOMALY_Z_THRESHOLD = 3.5

# Columns whose raw scale is known to differ across cohorts and where
# direct cross-cohort comparison of the mean would be misleading by
# design (the encoding pipeline preserves these scales). Explicitly
# excluded from anomaly detection rather than silently flagged.
SCALE_DIFFERENT_COLS = {"Income"}


# ─── Discovery (mirrors run_eda_comparative.py) ──────────────────────────────

def discover_dataset_folders() -> List[Path]:
    """Folders qualifying for cross-cohort analysis.

    A folder qualifies if it has outputs/<slug>/eda/cohort_summary.csv.
    Skips the comparative / xai_agreement / anomaly_detection folders.
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

def load_cohort_summary(folder: Path) -> Optional[Dict]:
    """Single-row summary per cohort."""
    path = folder / "eda" / "cohort_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df.iloc[0].to_dict()


def load_column_inventory(folder: Path) -> Optional[pd.DataFrame]:
    """Per-column type info (binary / ordinal / continuous)."""
    path = folder / "eda" / "column_inventory.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_feature_summary_stats(folder: Path) -> Dict[str, float]:
    """Per-feature summary statistic for one cohort.

    For binary columns: prevalence (fraction of 1) — equal to the mean
    of a 0/1 column.
    For ordinal/continuous: arithmetic mean.

    Reads directly from the `mean` column in column_inventory.csv
    written by step01_dataset_eda.py. This is uniform across types
    and avoids parsing the per-column distribution files (which have
    two different schemas: `value,count,percent` for binary/ordinal
    and `stat,value` for continuous).
    """
    stats: Dict[str, float] = {}
    inv = load_column_inventory(folder)
    if inv is None or "mean" not in inv.columns:
        return stats

    for _, row in inv.iterrows():
        col = row["column"]
        m = row.get("mean")
        if m is None or pd.isna(m):
            continue
        stats[col] = float(m)

    return stats


# ─── Builders ─────────────────────────────────────────────────────────────────

def build_cohort_distribution(
    folders: List[Path],
    inventories: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Feature × cohort summary stats (display-formatted strings)."""
    per_cohort_stats: Dict[str, Dict[str, float]] = {}
    for folder in folders:
        per_cohort_stats[folder.name] = load_feature_summary_stats(folder)

    # Determine union of all features across cohorts
    all_features = set()
    for s in per_cohort_stats.values():
        all_features.update(s.keys())
    all_features = sorted(all_features)

    # Map column to canonical type using any cohort that has it
    col_types: Dict[str, str] = {}
    for slug, inv in inventories.items():
        for _, row in inv.iterrows():
            c = row["column"]
            if c not in col_types:
                col_types[c] = row.get("detected_type", row.get("type", "unknown"))

    rows = []
    for col in all_features:
        row = {"feature": col, "type": col_types.get(col, "unknown")}
        for folder in folders:
            slug = folder.name
            v = per_cohort_stats[slug].get(col)
            if v is None:
                row[slug] = "—"
                continue
            if col_types.get(col) == "binary":
                row[slug] = f"{v * 100:.2f}%"
            elif col == "BMI":
                row[slug] = f"{v:.2f}"
            else:
                row[slug] = f"{v:.2f}"
        rows.append(row)

    return pd.DataFrame(rows)


def build_sample_sizes(folders: List[Path]) -> pd.DataFrame:
    """One row per cohort with size + target prevalence."""
    rows = []
    for folder in folders:
        summary = load_cohort_summary(folder)
        if summary is None:
            continue
        rows.append({
            "cohort": folder.name,
            "rows": int(summary.get("n_rows", -1)),
            "columns": int(summary.get("n_cols", -1)),
            "target_prevalence_pct": round(
                float(summary.get("target_prevalence", float("nan"))) * 100, 2
            ) if summary.get("target_prevalence") is not None else None,
        })
    return pd.DataFrame(rows)


def detect_anomalies(
    folders: List[Path],
    inventories: Dict[str, pd.DataFrame],
    threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.DataFrame:
    """Modified z-score outlier detection (Iglewicz & Hoaglin 1993).

    Robust to a single extreme value among the cohorts (median + MAD,
    not mean + std). This is exactly the property that catches a
    pipeline bug like CholCheck = 0% in one cohort vs ~96% in others.

    Excludes columns in SCALE_DIFFERENT_COLS by design.
    """
    per_cohort_stats: Dict[str, Dict[str, float]] = {}
    for folder in folders:
        per_cohort_stats[folder.name] = load_feature_summary_stats(folder)

    # Type lookup
    col_types: Dict[str, str] = {}
    for slug, inv in inventories.items():
        for _, row in inv.iterrows():
            c = row["column"]
            if c not in col_types:
                col_types[c] = row.get("detected_type", row.get("type", "unknown"))

    # Union of features
    all_features = set()
    for s in per_cohort_stats.values():
        all_features.update(s.keys())

    findings = []
    for col in sorted(all_features):
        if col in SCALE_DIFFERENT_COLS:
            continue

        per_cohort = {
            slug: s[col] for slug, s in per_cohort_stats.items()
            if col in s
        }
        if len(per_cohort) < 3:
            continue  # need >=3 cohorts for a meaningful median + MAD

        vals = np.array(list(per_cohort.values()), dtype=float)
        slugs = list(per_cohort.keys())

        median = float(np.median(vals))
        mad = float(np.median(np.abs(vals - median)))
        if mad < 1e-12:
            continue  # degenerate

        for slug, v in zip(slugs, vals):
            mz = 0.6745 * (v - median) / mad
            if abs(mz) > threshold:
                findings.append({
                    "feature": col,
                    "type": col_types.get(col, "unknown"),
                    "cohort": slug,
                    "value": round(float(v), 4),
                    "median_other_cohorts": round(median, 4),
                    "modified_z_score": round(float(mz), 2),
                    "direction": "high" if mz > 0 else "low",
                })

    if not findings:
        return pd.DataFrame(columns=[
            "feature", "type", "cohort", "value",
            "median_other_cohorts", "modified_z_score", "direction",
        ])

    return pd.DataFrame(findings).sort_values(
        by="modified_z_score",
        key=lambda s: s.abs(),
        ascending=False,
    ).reset_index(drop=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)

    folders = discover_dataset_folders()
    if len(folders) < 2:
        logging.error(
            f"Need >=2 cohort folders with eda/cohort_summary.csv. "
            f"Found {len(folders)}: {[f.name for f in folders]}"
        )
        logging.error("Run step01_dataset_eda for each cohort first.")
        sys.exit(1)

    logging.info(f"Found {len(folders)} cohorts:")
    for f in folders:
        logging.info(f"  • {f.name}")

    # Load inventories once
    inventories: Dict[str, pd.DataFrame] = {}
    for f in folders:
        inv = load_column_inventory(f)
        if inv is not None:
            inventories[f.name] = inv

    # 1. Sample sizes
    sizes = build_sample_sizes(folders)
    sizes_path = ANOMALY_DIR / "sample_sizes.csv"
    sizes.to_csv(sizes_path, index=False)
    logging.info(f"[1/3] Sample sizes -> {sizes_path}")
    logging.info(f"\n{sizes.to_string(index=False)}")

    # 2. Cohort distribution
    dist = build_cohort_distribution(folders, inventories)
    dist_path = ANOMALY_DIR / "cohort_distribution.csv"
    dist.to_csv(dist_path, index=False)
    logging.info(f"[2/3] Cohort distribution -> {dist_path}")
    logging.info(f"\n{dist.to_string(index=False)}")

    # 3. Anomaly detection
    anomalies = detect_anomalies(folders, inventories,
                                 threshold=ANOMALY_Z_THRESHOLD)
    anomalies_path = ANOMALY_DIR / "anomalies.csv"
    anomalies.to_csv(anomalies_path, index=False)
    logging.info(
        f"[3/3] Anomalies (|modified z| > {ANOMALY_Z_THRESHOLD}, "
        f"{', '.join(SCALE_DIFFERENT_COLS)} excluded) -> {anomalies_path}"
    )
    if anomalies.empty:
        logging.info("  No anomalies above threshold.")
    else:
        logging.info(f"\n{anomalies.to_string(index=False)}")
        logging.info("")
        logging.info("Interpretation guide:")
        logging.info("  • |z| > 50  -> almost certainly a pipeline bug")
        logging.info("  • |z| > 10  -> strong methodology change")
        logging.info("  • 3.5 < |z| < 10  -> documented drift; cross-check "
                     "against published statistics")

    logging.info("Done.")


if __name__ == "__main__":
    main()
