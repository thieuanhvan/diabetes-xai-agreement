from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
import math
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
AGREEMENT_DIR = OUTPUT_DIR / "xai_agreement"
PLOTS_DIR = AGREEMENT_DIR / "plots"

TARGET_DATASETS = [
    "cdc_brfss_2015_rebuilt",
    "cdc_brfss_2021_rebuilt",
    "cdc_brfss_2023_rebuilt",
]

TARGET_MODELS = [
    "xgboost",
    "random_forest",
    "logistic_regression",
]

ALPHA_LEVELS = [0.70, 0.80, 0.90]
RBO_P = 0.90


# ============================================================
# LOGGING
# ============================================================

def setup_run_logger(script_name: str = "run_xai_agreement") -> Path:
    """
    Configure timestamped logging for this run script.

    Each execution creates one separate log file under the project-level logs/
    folder while also printing the same messages to the console.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{script_name}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate log lines in IDE reruns.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("=" * 80)
    logging.info("START SCRIPT: %s.py", script_name)
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Output dir: %s", OUTPUT_DIR)
    logging.info("Agreement output dir: %s", AGREEMENT_DIR)
    logging.info("Plots dir: %s", PLOTS_DIR)
    logging.info("Log file: %s", log_path)
    logging.info("=" * 80)

    return log_path


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class CABCResult:
    feature_order: List[str]
    importance: np.ndarray
    n: int

    boundary_ab_idx_1based: int
    boundary_bc_local_idx_1based: int
    boundary_bc_global_idx_1based: int

    cum_at_ab: float
    cum_at_bc: float

    group_a: List[str]
    group_b: List[str]
    group_c: List[str]

    gap_pass1: np.ndarray
    gap_pass2: np.ndarray
    cum_pass1: np.ndarray
    cum_pass2: np.ndarray
    remaining_features: List[str]


# ============================================================
# HELPERS
# ============================================================

def ensure_dirs() -> None:
    AGREEMENT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def dataset_slug_to_year(dataset_slug: str) -> int:
    for token in dataset_slug.split("_"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    raise ValueError(f"Could not parse year from dataset slug: {dataset_slug}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize importance CSV columns to: feature, importance
    """
    feature_col = None
    importance_col = None

    for c in df.columns:
        cl = c.lower()
        if cl in {"feature", "features"}:
            feature_col = c
        if cl in {"importance", "mean_abs_shap", "mean_importance", "score"}:
            importance_col = c

    if feature_col is None or importance_col is None:
        raise ValueError(f"Could not normalize columns from: {list(df.columns)}")

    out = df[[feature_col, importance_col]].copy()
    out.columns = ["feature", "importance"]
    out["importance"] = pd.to_numeric(out["importance"], errors="coerce").fillna(0.0)
    return out


def load_importance(dataset_slug: str, model: str, method: str) -> Optional[pd.DataFrame]:
    """
    method:
        SHAP -> outputs/<dataset>/shap/<model>_shap_feature_importance.csv
        FI   -> outputs/<dataset>/fi/<model>_feature_importance.csv
    """
    if method == "SHAP":
        path = OUTPUT_DIR / dataset_slug / "shap" / f"{model}_shap_feature_importance.csv"
    elif method == "FI":
        path = OUTPUT_DIR / dataset_slug / "fi" / f"{model}_feature_importance.csv"
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not path.exists():
        logging.warning("Missing importance file: %s", path)
        return None

    df = pd.read_csv(path)
    logging.info(
        "Loaded %s importance | dataset=%s | model=%s | rows=%d | path=%s",
        method,
        dataset_slug,
        model,
        len(df),
        path,
    )
    return normalize_columns(df)


def align_feature_vectors(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two importance tables on union of features.
    Missing features get zero importance.
    """
    all_features = sorted(set(df_a["feature"]) | set(df_b["feature"]))

    a = df_a.set_index("feature").reindex(all_features).fillna(0.0).reset_index()
    b = df_b.set_index("feature").reindex(all_features).fillna(0.0).reset_index()

    a.columns = ["feature", "importance"]
    b.columns = ["feature", "importance"]
    return a, b


def sort_desc(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def cosine_similarity_from_importance(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    a, b = align_feature_vectors(df_a, df_b)
    va = a["importance"].to_numpy(dtype=float)
    vb = b["importance"].to_numpy(dtype=float)

    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def rbo_score(rank_a: List[str], rank_b: List[str], p: float = 0.90) -> float:
    """
    Simple finite-depth RBO approximation.
    """
    depth = max(len(rank_a), len(rank_b))
    if depth == 0:
        return 0.0

    overlap = 0.0
    seen_a = set()
    seen_b = set()
    score = 0.0

    for d in range(1, depth + 1):
        if d <= len(rank_a):
            seen_a.add(rank_a[d - 1])
        if d <= len(rank_b):
            seen_b.add(rank_b[d - 1])

        overlap = len(seen_a & seen_b)
        agreement = overlap / d
        score += agreement * (p ** (d - 1))

    return float((1 - p) * score)


def prefix_set_by_alpha(df_sorted: pd.DataFrame, alpha: float) -> Tuple[set, int]:
    """
    Return the smallest prefix whose cumulative normalized importance >= alpha.
    """
    x = df_sorted.copy()
    total = float(x["importance"].sum())
    if total <= 0:
        return set(), 0

    x["norm"] = x["importance"] / total
    x["cum"] = x["norm"].cumsum()

    k = int((x["cum"] >= alpha).idxmax()) + 1
    return set(x.head(k)["feature"]), k


def jaccard_for_alpha(df_a: pd.DataFrame, df_b: pd.DataFrame, alpha: float) -> Tuple[float, int, int]:
    a, b = align_feature_vectors(df_a, df_b)
    a = sort_desc(a)
    b = sort_desc(b)

    sa, ka = prefix_set_by_alpha(a, alpha)
    sb, kb = prefix_set_by_alpha(b, alpha)

    union = sa | sb
    if not union:
        return 0.0, ka, kb

    j = len(sa & sb) / len(union)
    return float(j), ka, kb


def build_cabc(df: pd.DataFrame) -> CABCResult:
    """
    2-pass cABC recursion:
      Pass 1 on all features -> A / B-C
      Pass 2 on remaining features -> B / C
    """
    x = sort_desc(df)
    features = x["feature"].tolist()
    imp = x["importance"].to_numpy(dtype=float)
    n = len(features)

    total = float(imp.sum())
    if total <= 0:
        raise ValueError("Importance vector sums to zero; cannot build cABC.")

    norm = imp / total
    cum = np.cumsum(norm)
    xs = np.arange(1, n + 1) / n
    gap1 = cum - xs

    ab_idx0 = int(np.argmax(gap1))
    ab_idx1 = ab_idx0 + 1
    cum_ab = float(cum[ab_idx0])

    remaining_imp = imp[ab_idx1:]
    remaining_features = features[ab_idx1:]

    if len(remaining_imp) == 0:
        group_a = features
        group_b = []
        group_c = []
        return CABCResult(
            feature_order=features,
            importance=imp,
            n=n,
            boundary_ab_idx_1based=ab_idx1,
            boundary_bc_local_idx_1based=0,
            boundary_bc_global_idx_1based=ab_idx1,
            cum_at_ab=cum_ab,
            cum_at_bc=1.0,
            group_a=group_a,
            group_b=group_b,
            group_c=group_c,
            gap_pass1=gap1,
            gap_pass2=np.array([]),
            cum_pass1=cum,
            cum_pass2=np.array([]),
            remaining_features=[],
        )

    rem_norm = remaining_imp / float(remaining_imp.sum())
    rem_cum = np.cumsum(rem_norm)
    rem_xs = np.arange(1, len(remaining_imp) + 1) / len(remaining_imp)
    gap2 = rem_cum - rem_xs

    bc_local0 = int(np.argmax(gap2))
    bc_local1 = bc_local0 + 1
    bc_global1 = ab_idx1 + bc_local1
    cum_bc = float(cum[bc_global1 - 1])

    group_a = features[:ab_idx1]
    group_b = features[ab_idx1:bc_global1]
    group_c = features[bc_global1:]

    return CABCResult(
        feature_order=features,
        importance=imp,
        n=n,
        boundary_ab_idx_1based=ab_idx1,
        boundary_bc_local_idx_1based=bc_local1,
        boundary_bc_global_idx_1based=bc_global1,
        cum_at_ab=cum_ab,
        cum_at_bc=cum_bc,
        group_a=group_a,
        group_b=group_b,
        group_c=group_c,
        gap_pass1=gap1,
        gap_pass2=gap2,
        cum_pass1=cum,
        cum_pass2=rem_cum,
        remaining_features=remaining_features,
    )


def jaccard_sets(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb) / len(union))


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = AGREEMENT_DIR / name
    df.to_csv(path, index=False)
    logging.info("Saved CSV: %s | rows=%d | columns=%d", path, len(df), len(df.columns))
    return path


def create_output_subdir(*parts: str) -> Path:
    d = PLOTS_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# PLOTS
# ============================================================

def plot_cabc_both_breakpoints(dataset_slug: str, model: str, method: str, cabc: CABCResult) -> Path:
    year = dataset_slug_to_year(dataset_slug)
    plot_dir = create_output_subdir("cabc_per_run")

    norm = cabc.importance / cabc.importance.sum()
    cum = np.cumsum(norm)
    x = np.arange(1, cabc.n + 1) / cabc.n

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot([0, 1], [0, 1], linestyle="--", color="dimgray", linewidth=1.5, label="Uniform reference (y = x)")
    ax.plot(np.r_[0, x], np.r_[0, cum], marker="o", linewidth=2, color="#2a7fad", label="Lorenz curve")

    # Regions
    xa = cabc.boundary_ab_idx_1based / cabc.n
    xb = cabc.boundary_bc_global_idx_1based / cabc.n

    ax.axvspan(0, xa, color="#7cb37a", alpha=0.25, label=f"Group A: {len(cabc.group_a)} features (essential)")
    ax.axvspan(xa, xb, color="#ead89f", alpha=0.35, label=f"Group B: {len(cabc.group_b)} features (supporting)")
    ax.axvspan(xb, 1.0, color="#cfcfcf", alpha=0.5, label=f"Group C: {len(cabc.group_c)} features (marginal)")

    ax.axvline(x=xa, color="#f08a5d", linestyle=":", linewidth=1)
    ax.axvline(x=xb, color="#b39ddb", linestyle=":", linewidth=1)
    ax.axhline(y=cabc.cum_at_ab, color="#f08a5d", linestyle=":", linewidth=1)
    ax.axhline(y=cabc.cum_at_bc, color="#b39ddb", linestyle=":", linewidth=1)

    ax.scatter(
        [xa], [cabc.cum_at_ab], s=280, color="#ef4444", edgecolors="black",
        zorder=5, label="Breakpoint A/B (argmax from Pass 1)"
    )
    ax.scatter(
        [xb], [cabc.cum_at_bc], s=220, color="#7b1fa2", edgecolors="black", marker="s",
        zorder=5, label="Breakpoint B/C (argmax from Pass 2 — recursion)"
    )

    # Feature labels near points
    for idx, feat in enumerate(cabc.feature_order, start=1):
        fx = idx / cabc.n
        fy = cum[idx - 1]
        if idx <= min(cabc.n, 10):
            ax.annotate(
                feat,
                (fx, fy),
                textcoords="offset points",
                xytext=(4, -12),
                ha="left",
                fontsize=9,
                color="#2a7fad",
            )

    ax.text(
        0.02, 0.98,
        "2-pass recursion:\n"
        "Pass 1 on all n features → A/B\n"
        "Pass 2 on (n − |A|) features → B/C",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f2ebd3", edgecolor="#888", alpha=0.95),
    )

    ax.annotate(
        f"Breakpoint A/B\nFeature #{cabc.boundary_ab_idx_1based} ({cabc.feature_order[cabc.boundary_ab_idx_1based - 1]})\n"
        f"cum = {cabc.cum_at_ab * 100:.2f}%\n→ Group A = {len(cabc.group_a)} features",
        xy=(xa, cabc.cum_at_ab),
        xytext=(0.40, 0.58),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-", color="#ef4444", lw=1.5),
        fontsize=12,
        color="#ef4444",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ef4444", alpha=0.95),
    )

    ax.annotate(
        f"Breakpoint B/C\nFeature #{cabc.boundary_bc_global_idx_1based} ({cabc.feature_order[cabc.boundary_bc_global_idx_1based - 1]})\n"
        f"cum = {cabc.cum_at_bc * 100:.2f}%\n→ Group B = {len(cabc.group_b)} features",
        xy=(xb, cabc.cum_at_bc),
        xytext=(0.67, 0.60),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-", color="#7b1fa2", lw=1.5),
        fontsize=12,
        color="#7b1fa2",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#7b1fa2", alpha=0.95),
    )

    ax.set_title(
        f"cABC with BOTH breakpoints — {model.replace('_', ' ').title()} {method} on BRFSS {year}\n"
        f"(n = {cabc.n} features; 2-pass recursion)",
        fontsize=16,
        weight="bold",
        pad=14,
    )
    ax.set_xlabel("Cumulative fraction of features (x = i/n)", fontsize=14, weight="bold")
    ax.set_ylabel("Cumulative importance (y)", fontsize=14, weight="bold")
    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)

    out = plot_dir / f"{dataset_slug}_{model}_{method.lower()}_cabc_both_breakpoints.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot: %s", out)
    return out


def plot_cabc_two_pass(dataset_slug: str, model: str, method: str, cabc: CABCResult) -> Path:
    year = dataset_slug_to_year(dataset_slug)
    plot_dir = create_output_subdir("cabc_per_run")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Pass 1
    ax = axes[0]
    i = np.arange(1, cabc.n + 1)
    y = cabc.gap_pass1 * 100
    ax.plot(i, y, marker="o", color="#2a7fad", linewidth=2)
    ax.axvspan(0.5, cabc.boundary_ab_idx_1based + 0.5, color="#8cc08c", alpha=0.28)
    ax.axvspan(cabc.boundary_ab_idx_1based + 0.5, cabc.n + 0.5, color="#b7d3e8", alpha=0.35)
    ax.scatter(
        [cabc.boundary_ab_idx_1based],
        [y[cabc.boundary_ab_idx_1based - 1]],
        s=220,
        color="#ef4444",
        edgecolors="black",
        zorder=5,
    )
    ax.annotate(
        f"argmax at i = {cabc.boundary_ab_idx_1based}\n"
        f"(y−x = {y[cabc.boundary_ab_idx_1based - 1]:.2f}pp)\n→ Group A boundary",
        xy=(cabc.boundary_ab_idx_1based, y[cabc.boundary_ab_idx_1based - 1]),
        xytext=(cabc.boundary_ab_idx_1based + 2, max(y) * 0.70),
        arrowprops=dict(arrowstyle="-", color="#ef4444", lw=1.5),
        fontsize=11,
        color="#ef4444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ef4444", alpha=0.95),
    )
    ax.set_title(f"Pass 1: Find breakpoint A/B on ALL {cabc.n} features", fontsize=12, weight="bold")
    ax.set_xlabel("Feature rank i", fontsize=12, weight="bold")
    ax.set_ylabel("y − x (percentage points)", fontsize=12, weight="bold")
    ax.set_xticks(i)
    ax.set_xticklabels(cabc.feature_order, rotation=60, ha="right", fontsize=8)
    ax.grid(alpha=0.25)

    # Pass 2
    ax = axes[1]
    m = len(cabc.remaining_features)
    if m > 0:
        i2 = np.arange(1, m + 1)
        y2 = cabc.gap_pass2 * 100
        ax.plot(i2, y2, marker="s", color="#8e44ad", linewidth=2)
        ax.axvspan(0.5, cabc.boundary_bc_local_idx_1based + 0.5, color="#f1c27d", alpha=0.32)
        ax.axvspan(cabc.boundary_bc_local_idx_1based + 0.5, m + 0.5, color="#cdb4db", alpha=0.35)
        ax.scatter(
            [cabc.boundary_bc_local_idx_1based],
            [y2[cabc.boundary_bc_local_idx_1based - 1]],
            s=180,
            color="#ef4444",
            edgecolors="black",
            marker="s",
            zorder=5,
        )
        ax.annotate(
            f"argmax at i_rem = {cabc.boundary_bc_local_idx_1based}\n"
            f"(global rank {cabc.boundary_bc_global_idx_1based})\n→ Group B boundary",
            xy=(cabc.boundary_bc_local_idx_1based, y2[cabc.boundary_bc_local_idx_1based - 1]),
            xytext=(cabc.boundary_bc_local_idx_1based + 1.5, max(y2) * 0.68),
            arrowprops=dict(arrowstyle="-", color="#ef4444", lw=1.5),
            fontsize=11,
            color="#ef4444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ef4444", alpha=0.95),
        )
        ax.set_xticks(i2)
        ax.set_xticklabels(cabc.remaining_features, rotation=60, ha="right", fontsize=8)

    ax.set_title(
        f"Pass 2: Find breakpoint B/C on (n−|A|) = {m} remaining features\n(re-normalized within subset)",
        fontsize=12,
        weight="bold",
    )
    ax.set_xlabel("Feature rank within remaining subset i_rem", fontsize=12, weight="bold")
    ax.set_ylabel("y_rem − x_rem (pp; re-normalized)", fontsize=12, weight="bold")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "cABC — 2-pass recursion produces 3 groups (ABC)",
        fontsize=16,
        weight="bold",
        y=1.02,
    )

    out = plot_dir / f"{dataset_slug}_{model}_{method.lower()}_cabc_two_pass.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot: %s", out)
    return out


def plot_breakpoint_summary(cabc_groups_df: pd.DataFrame) -> Path:
    plot_dir = create_output_subdir("summaries")
    df = cabc_groups_df.copy()
    df["label"] = (
        df["model"].str.replace("_", "-", regex=False).str.upper()
        + "-"
        + df["method"]
        + "\n("
        + df["year"].astype(str)
        + ")"
    )
    df["A_pct"] = df["cum_A_pct"]

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = df["method"].map({"SHAP": "#2a85b3", "FI": "#f28e00"}).fillna("#777")
    bars = ax.bar(df["label"], df["A_pct"], color=colors, edgecolor="black", alpha=0.95)

    mean_a = df["A_pct"].mean()
    ax.axhline(mean_a, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_a:.2f}%")
    ax.axhline(80.0, color="green", linestyle="--", linewidth=1.5, label="Pareto 80%")

    ax.axhspan(77.5, 82.5, color="#9fc4a0", alpha=0.25, label="80% ± 2.5% band")

    for rect, val in zip(bars, df["A_pct"]):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    std = df["A_pct"].std(ddof=0)
    n_in_band = int(((df["A_pct"] >= 77.5) & (df["A_pct"] <= 82.5)).sum())

    ax.set_title(
        f"cABC breakpoint %A across {len(df)} runs — data-driven, NOT fixed at 80%\n"
        f"Range: {df['A_pct'].min():.2f}%–{df['A_pct'].max():.2f}% | Std: {std:.2f}pp | Only {n_in_band}/{len(df)} in Pareto band",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    ax.set_ylabel("%A (Group A cumulative importance)", fontsize=12, weight="bold")
    ax.set_ylim(60, max(95, df["A_pct"].max() + 5))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    out = plot_dir / "cabc_breakpoint_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot: %s", out)
    return out


def plot_three_methods_for_all_runs(within_model_df: pd.DataFrame, within_model_cabc_df: pd.DataFrame) -> List[Path]:
    """
    Create one "three methods" summary figure for EACH (year, model) run.
    Filename pattern: three_methods_[year]_[model].png

    Previously this function only plotted the first run; now it iterates
    over all 9 runs (3 years x 3 models) for complete coverage.
    """
    if within_model_df.empty or within_model_cabc_df.empty:
        return []

    saved_paths: List[Path] = []
    for _, row_agree in within_model_df.iterrows():
        path = _plot_single_three_methods(row_agree, within_model_cabc_df)
        if path is not None:
            saved_paths.append(path)
    return saved_paths


def _plot_single_three_methods(row_agree: pd.Series, within_model_cabc_df: pd.DataFrame) -> Optional[Path]:
    """Render one three-methods panel for a given (year, model) run."""
    plot_dir = create_output_subdir("summaries")

    year = int(row_agree["year"])
    model = str(row_agree["model"])

    row_cabc = within_model_cabc_df[
        (within_model_cabc_df["year"] == year) &
        (within_model_cabc_df["model"] == model)
    ]
    if row_cabc.empty:
        return None

    row_cabc = row_cabc.iloc[0]

    alphas = [0.70, 0.80, 0.90]
    jvals = [row_agree[f"J@{int(a*100)}"] for a in alphas]

    cabc_vals = [row_cabc["J_A"], row_cabc["J_B"], row_cabc["J_C"]]
    cabc_labels = [
        f"J_A\n(essential,\n|A|={int(row_cabc['|A_1|'])})",
        f"J_B\n(supporting,\n|B|={int(row_cabc['|B_1|'])})",
        f"J_C\n(marginal,\n|C|={int(row_cabc['|C_1|'])})",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Method 1: top-K proxy using overlap at K=5/10 plus interpolated line
    ax = axes[0]
    ks = [5, 10]
    kvals = [row_agree.get("top5_overlap", np.nan), row_agree.get("top10_overlap", np.nan)]
    if any(pd.notnull(kvals)):
        ax.plot(ks, kvals, marker="o", color="#f28e00", linewidth=2)
        for k, v in zip(ks, kvals):
            if pd.notnull(v):
                ax.annotate(f"K={k}: J={v:.2f}", (k, v), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_title("Method 1: Top-K overlap\nAgreement depends on chosen K", fontsize=10, weight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Jaccard agreement")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

    # Method 2: J@alpha
    ax = axes[1]
    ax.plot(alphas, jvals, marker="o", color="#8e44ad", linewidth=2)
    for a, j in zip(alphas, jvals):
        ax.annotate(f"α={a:.2f}\nJ={j:.2f}", (a, j), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axvline(0.80, color="red", linestyle="--", alpha=0.6)
    ax.set_title("Method 2: Coverage-Jaccard@α\nAgreement depends on chosen α", fontsize=10, weight="bold")
    ax.set_xlabel("α (cumulative importance threshold)")
    ax.set_ylabel("Jaccard agreement")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

    # Method 3: cABC
    ax = axes[2]
    bars = ax.bar([0, 1, 2], cabc_vals, color=["#2a9d8f", "#f4a261", "#7a7a7a"], edgecolor="black", alpha=0.95)
    for i, (bar, v) in enumerate(zip(bars, cabc_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(cabc_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pairwise group Jaccard")
    ax.set_title("Method 3: cABC (per-group Jaccard)\n2 breakpoints → 3 groups\n✓ visual agreement\n✓ clinical interpretation", fontsize=10, weight="bold", color="darkgreen")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Three methods on same data ({model.replace('_', ' ').title()} SHAP vs FI, BRFSS {year})",
        fontsize=13,
        weight="bold",
    )

    out = plot_dir / f"three_methods_{year}_{model}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot: %s", out)
    return out


# ============================================================
# AGREEMENT TABLES
# ============================================================

def build_cabc_groups() -> pd.DataFrame:
    rows = []
    for dataset_slug in TARGET_DATASETS:
        year = dataset_slug_to_year(dataset_slug)
        for model in TARGET_MODELS:
            for method in ["SHAP", "FI"]:
                df = load_importance(dataset_slug, model, method)
                if df is None:
                    logging.warning(f"Missing {method} importance: {dataset_slug} | {model}")
                    continue

                cabc = build_cabc(df)

                rows.append({
                    "year": year,
                    "dataset": dataset_slug,
                    "model": model,
                    "method": method,
                    "|A|": len(cabc.group_a),
                    "|B|": len(cabc.group_b),
                    "|C|": len(cabc.group_c),
                    "cum_A_pct": round(cabc.cum_at_ab * 100, 4),
                    "cum_B_pct": round((cabc.cum_at_bc - cabc.cum_at_ab) * 100, 4),
                    "cum_C_pct": round((1.0 - cabc.cum_at_bc) * 100, 4),
                    "A": ",".join(cabc.group_a),
                    "B": ",".join(cabc.group_b),
                    "C": ",".join(cabc.group_c),
                    "breakpoint_ab_feature": cabc.feature_order[cabc.boundary_ab_idx_1based - 1],
                    "breakpoint_bc_feature": cabc.feature_order[cabc.boundary_bc_global_idx_1based - 1] if cabc.group_b else "",
                })

                # Per-run cABC plots
                plot_cabc_both_breakpoints(dataset_slug, model, method, cabc)
                plot_cabc_two_pass(dataset_slug, model, method, cabc)

    return pd.DataFrame(rows)


def build_within_model_agreement(cabc_lookup: Dict[Tuple[str, str, str], CABCResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agreement_rows = []
    cabc_rows = []

    for dataset_slug in TARGET_DATASETS:
        year = dataset_slug_to_year(dataset_slug)
        for model in TARGET_MODELS:
            shap_df = load_importance(dataset_slug, model, "SHAP")
            fi_df = load_importance(dataset_slug, model, "FI")

            if shap_df is None or fi_df is None:
                continue

            row = {
                "year": year,
                "dataset": dataset_slug,
                "model": model,
                "method_A": "SHAP",
                "method_B": "PI",
                "note": "",
            }

            for alpha in ALPHA_LEVELS:
                j, ka, kb = jaccard_for_alpha(shap_df, fi_df, alpha)
                row[f"J@{int(alpha * 100)}"] = round(j, 4)
                row[f"|Sa|@{int(alpha * 100)}"] = ka
                row[f"|Sb|@{int(alpha * 100)}"] = kb

            row["RBO"] = round(rbo_score(
                sort_desc(shap_df)["feature"].tolist(),
                sort_desc(fi_df)["feature"].tolist(),
                p=RBO_P,
            ), 4)
            row["cosine"] = round(cosine_similarity_from_importance(shap_df, fi_df), 4)

            # Convenience fields for summary plot
            row["top5_overlap"] = round(compute_topk_overlap(shap_df, fi_df, 5), 4)
            row["top10_overlap"] = round(compute_topk_overlap(shap_df, fi_df, 10), 4)

            agreement_rows.append(row)

            c1 = cabc_lookup[(dataset_slug, model, "SHAP")]
            c2 = cabc_lookup[(dataset_slug, model, "FI")]

            cabc_rows.append({
                "year": year,
                "dataset": dataset_slug,
                "model": model,
                "method_A": "SHAP",
                "method_B": "PI",
                "|A_1|": len(c1.group_a),
                "|A_2|": len(c2.group_a),
                "|B_1|": len(c1.group_b),
                "|B_2|": len(c2.group_b),
                "|C_1|": len(c1.group_c),
                "|C_2|": len(c2.group_c),
                "J_A": round(jaccard_sets(c1.group_a, c2.group_a), 4),
                "J_B": round(jaccard_sets(c1.group_b, c2.group_b), 4),
                "J_C": round(jaccard_sets(c1.group_c, c2.group_c), 4),
                "A_1": ",".join(c1.group_a),
                "A_2": ",".join(c2.group_a),
                "B_1": ",".join(c1.group_b),
                "B_2": ",".join(c2.group_b),
                "C_1": ",".join(c1.group_c),
                "C_2": ",".join(c2.group_c),
            })

    return pd.DataFrame(agreement_rows), pd.DataFrame(cabc_rows)


def build_cross_model_shap_agreement(cabc_lookup: Dict[Tuple[str, str, str], CABCResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agreement_rows = []
    cabc_rows = []

    for dataset_slug in TARGET_DATASETS:
        year = dataset_slug_to_year(dataset_slug)
        for model_a, model_b in combinations(TARGET_MODELS, 2):
            a = load_importance(dataset_slug, model_a, "SHAP")
            b = load_importance(dataset_slug, model_b, "SHAP")
            if a is None or b is None:
                continue

            row = {
                "year": year,
                "dataset": dataset_slug,
                "model_A": model_a,
                "model_B": model_b,
                "method": "SHAP",
            }

            for alpha in ALPHA_LEVELS:
                j, ka, kb = jaccard_for_alpha(a, b, alpha)
                row[f"J@{int(alpha * 100)}"] = round(j, 4)
                row[f"|Sa|@{int(alpha * 100)}"] = ka
                row[f"|Sb|@{int(alpha * 100)}"] = kb

            row["RBO"] = round(rbo_score(sort_desc(a)["feature"].tolist(), sort_desc(b)["feature"].tolist(), p=RBO_P), 4)
            row["cosine"] = round(cosine_similarity_from_importance(a, b), 4)

            agreement_rows.append(row)

            c1 = cabc_lookup[(dataset_slug, model_a, "SHAP")]
            c2 = cabc_lookup[(dataset_slug, model_b, "SHAP")]

            cabc_rows.append({
                "year": year,
                "dataset": dataset_slug,
                "model_A": model_a,
                "model_B": model_b,
                "method": "SHAP",
                "|A_1|": len(c1.group_a),
                "|A_2|": len(c2.group_a),
                "|B_1|": len(c1.group_b),
                "|B_2|": len(c2.group_b),
                "|C_1|": len(c1.group_c),
                "|C_2|": len(c2.group_c),
                "J_A": round(jaccard_sets(c1.group_a, c2.group_a), 4),
                "J_B": round(jaccard_sets(c1.group_b, c2.group_b), 4),
                "J_C": round(jaccard_sets(c1.group_c, c2.group_c), 4),
                "A_1": ",".join(c1.group_a),
                "A_2": ",".join(c2.group_a),
                "B_1": ",".join(c1.group_b),
                "B_2": ",".join(c2.group_b),
                "C_1": ",".join(c1.group_c),
                "C_2": ",".join(c2.group_c),
            })

    return pd.DataFrame(agreement_rows), pd.DataFrame(cabc_rows)


def build_temporal_stability(cabc_lookup: Dict[Tuple[str, str, str], CABCResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agreement_rows = []
    cabc_rows = []

    year_map = {dataset_slug_to_year(ds): ds for ds in TARGET_DATASETS}
    years_sorted = sorted(year_map.keys())

    for model in TARGET_MODELS:
        for method in ["SHAP", "FI"]:
            for year_a, year_b in combinations(years_sorted, 2):
                ds_a = year_map[year_a]
                ds_b = year_map[year_b]

                a = load_importance(ds_a, model, method)
                b = load_importance(ds_b, model, method)
                if a is None or b is None:
                    continue

                row = {
                    "model": model,
                    "method": method,
                    "year_A": year_a,
                    "year_B": year_b,
                }

                for alpha in ALPHA_LEVELS:
                    j, ka, kb = jaccard_for_alpha(a, b, alpha)
                    row[f"J@{int(alpha * 100)}"] = round(j, 4)
                    row[f"|Sa|@{int(alpha * 100)}"] = ka
                    row[f"|Sb|@{int(alpha * 100)}"] = kb

                row["RBO"] = round(rbo_score(sort_desc(a)["feature"].tolist(), sort_desc(b)["feature"].tolist(), p=RBO_P), 4)
                row["cosine"] = round(cosine_similarity_from_importance(a, b), 4)

                agreement_rows.append(row)

                c1 = cabc_lookup[(ds_a, model, method)]
                c2 = cabc_lookup[(ds_b, model, method)]

                cabc_rows.append({
                    "model": model,
                    "method": method,
                    "year_A": year_a,
                    "year_B": year_b,
                    "|A_1|": len(c1.group_a),
                    "|A_2|": len(c2.group_a),
                    "|B_1|": len(c1.group_b),
                    "|B_2|": len(c2.group_b),
                    "|C_1|": len(c1.group_c),
                    "|C_2|": len(c2.group_c),
                    "J_A": round(jaccard_sets(c1.group_a, c2.group_a), 4),
                    "J_B": round(jaccard_sets(c1.group_b, c2.group_b), 4),
                    "J_C": round(jaccard_sets(c1.group_c, c2.group_c), 4),
                    "A_1": ",".join(c1.group_a),
                    "A_2": ",".join(c2.group_a),
                    "B_1": ",".join(c1.group_b),
                    "B_2": ",".join(c2.group_b),
                    "C_1": ",".join(c1.group_c),
                    "C_2": ",".join(c2.group_c),
                })

    return pd.DataFrame(agreement_rows), pd.DataFrame(cabc_rows)


def compute_topk_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, k: int = 5) -> float:
    a = sort_desc(df_a)
    b = sort_desc(df_b)
    sa = set(a.head(k)["feature"])
    sb = set(b.head(k)["feature"])
    if k == 0:
        return 0.0
    return len(sa & sb) / k


# ============================================================
# ADDITIONAL SUMMARY FIGURES (added for manuscript)
# ============================================================

def plot_core_groupA_heatmap(cabc_groups_df: pd.DataFrame) -> Optional[Path]:
    """
    Heatmap of Group A membership across all 18 cABC runs.

    Rows: 18 runs (one per combination of model x year x method), sorted
          as "MODEL-METHOD-YEAR" with model, year, method nested.
    Columns: all 21 predictors, sorted by Group A frequency (descending).
    Cell value: 1 if feature in Group A for that run, 0 otherwise.

    Features that never appear in any Group A are still displayed on the
    right with "0/18" annotation, so the reader sees the invariant core
    (left, all dark) and the never-essential set (right, all light).
    """
    if cabc_groups_df.empty:
        return None

    plot_dir = create_output_subdir("summaries")

    # Gather all unique features from A, B, C across runs
    all_features: List[str] = []
    for _, r in cabc_groups_df.iterrows():
        for bucket in ["A", "B", "C"]:
            raw = str(r[bucket]) if pd.notnull(r[bucket]) else ""
            for feat in raw.split(","):
                feat = feat.strip()
                if feat and feat not in all_features:
                    all_features.append(feat)

    # Frequency each feature appears in Group A
    freq_A: Dict[str, int] = {f: 0 for f in all_features}
    for _, r in cabc_groups_df.iterrows():
        for feat in str(r["A"]).split(","):
            feat = feat.strip()
            if feat in freq_A:
                freq_A[feat] += 1

    features_sorted = sorted(all_features, key=lambda f: (-freq_A[f], f))

    # Sort runs by year -> model -> method (SHAP then FI)
    df = cabc_groups_df.copy()
    method_order = {"SHAP": 0, "FI": 1}
    model_order = {m: i for i, m in enumerate(["logistic_regression", "random_forest", "xgboost"])}
    df["_method_order"] = df["method"].map(method_order).fillna(99).astype(int)
    df["_model_order"] = df["model"].map(model_order).fillna(99).astype(int)
    df = df.sort_values(["year", "_model_order", "_method_order"]).reset_index(drop=True)

    n_runs = len(df)
    n_feats = len(features_sorted)

    # Build matrix
    matrix = np.zeros((n_runs, n_feats), dtype=int)
    labels = []
    for i, (_, r) in enumerate(df.iterrows()):
        model_short = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGB"}.get(r["model"], r["model"][:3].upper())
        labels.append(f"{model_short}-{r['method']}-{r['year']}")
        in_A = [f.strip() for f in str(r["A"]).split(",") if f.strip()]
        for feat in in_A:
            if feat in features_sorted:
                j = features_sorted.index(feat)
                matrix[i, j] = 1

    # Plot
    fig_w = max(12, 0.55 * n_feats)
    fig_h = max(7, 0.42 * n_runs)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)

    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(features_sorted, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_runs))
    ax.set_yticklabels(labels, fontsize=9)

    # Thin white grid
    for i in range(n_runs + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5)
    for j in range(n_feats + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.5)

    # Frequency annotation below each column
    for j, f in enumerate(features_sorted):
        freq = freq_A[f]
        color = "red" if freq == n_runs else ("black" if freq > 0 else "gray")
        weight = "bold" if freq == n_runs else "normal"
        ax.text(j, n_runs - 0.2, f"{freq}/{n_runs}", ha="center", va="top",
                fontsize=8, color=color, weight=weight)

    ax.set_title(
        f"Group A membership across {n_runs} cABC runs\n"
        f"(dark = in Group A; red numbers mark invariant core features)",
        fontsize=13, weight="bold", pad=12,
    )
    ax.set_xlabel("Features (sorted by Group A frequency)", fontsize=11)
    ax.set_ylabel("Run (Model-Method-Year)", fontsize=11)

    out = plot_dir / "core_groupA_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot: %s", out)
    return out


def plot_fairness_drift(
    attributes: Tuple[str, ...] = ("Age", "Income"),
    severity_thresholds: Tuple[float, float] = (0.1, 0.2),
) -> Optional[Path]:
    """
    Line plot of Equalized Odds delta_tpr across 3 models x 3 years,
    one panel per protected attribute.

    Reads from per-dataset analysis folders:
        outputs/cdc_brfss_YYYY_rebuilt/analysis/MODEL_fairness_equalized_odds_summary.csv
    Colored severity bands: Good (<0.1), Concerning (0.1-0.2), Critical (>=0.2).
    """
    plot_dir = create_output_subdir("summaries")

    years = [2015, 2021, 2023]
    models = ["logistic_regression", "random_forest", "xgboost"]
    model_display = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
    model_colors = {"logistic_regression": "#1f77b4", "random_forest": "#2ca02c", "xgboost": "#d62728"}
    model_markers = {"logistic_regression": "o", "random_forest": "s", "xgboost": "^"}

    # Collect data[attr][model] = [val_2015, val_2021, val_2023]
    data: Dict[str, Dict[str, List[float]]] = {a: {m: [] for m in models} for a in attributes}

    for year in years:
        ds_folder = OUTPUT_DIR / f"cdc_brfss_{year}_rebuilt" / "analysis"
        for model in models:
            csv_path = ds_folder / f"{model}_fairness_equalized_odds_summary.csv"
            if not csv_path.exists():
                for a in attributes:
                    data[a][model].append(float("nan"))
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                for a in attributes:
                    data[a][model].append(float("nan"))
                continue
            for a in attributes:
                match = df[df["attribute"] == a]
                if match.empty:
                    data[a][model].append(float("nan"))
                else:
                    data[a][model].append(float(match.iloc[0]["delta_tpr"]))

    # Plot
    n = len(attributes)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5), sharey=False)
    if n == 1:
        axes = [axes]

    concerning, critical = severity_thresholds
    for ax, attr in zip(axes, attributes):
        y_vals = [v for model in models for v in data[attr][model] if not np.isnan(v)]
        y_max = max(y_vals + [critical + 0.1]) * 1.1 if y_vals else 1.0

        ax.axhspan(critical, y_max, color="#e74c3c", alpha=0.08, label=f"Critical (>={critical})")
        ax.axhspan(concerning, critical, color="#f39c12", alpha=0.08, label=f"Concerning ({concerning}-{critical})")
        ax.axhspan(0.0, concerning, color="#27ae60", alpha=0.05, label=f"Good (<{concerning})")
        ax.axhline(concerning, color="orange", linestyle=":", alpha=0.7, linewidth=1)
        ax.axhline(critical, color="red", linestyle=":", alpha=0.7, linewidth=1)

        for model in models:
            ax.plot(
                years, data[attr][model],
                marker=model_markers[model], markersize=10, linewidth=2.5,
                color=model_colors[model], label=model_display[model],
            )
        ax.set_title(f"{attr} δTPR", fontsize=13, weight="bold")
        ax.set_xlabel("BRFSS Year", fontsize=11)
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, y_max)

    axes[0].set_ylabel("Equalized Odds δTPR", fontsize=11)

    # Unified legend from first panel (deduplicate)
    handles, labels_l = axes[0].get_legend_handles_labels()
    seen = set()
    unique = []
    for h, l in zip(handles, labels_l):
        if l not in seen:
            seen.add(l)
            unique.append((h, l))
    fig.legend(
        [u[0] for u in unique], [u[1] for u in unique],
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=min(6, len(unique)), fontsize=10, frameon=True,
    )

    fig.suptitle(
        "Temporal fairness drift across three models",
        fontsize=14, weight="bold", y=1.02,
    )

    out = plot_dir / "fairness_drift.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot: %s", out)
    return out


# ============================================================
# MAIN
# ============================================================

def run_xai_agreement_full() -> None:
    """
    Run the full XAI agreement analysis.

    This version preserves all outputs from the uploaded v2 script, including:
    - cABC CSVs
    - within-model agreement CSVs
    - cross-model SHAP agreement CSVs
    - temporal stability CSVs
    - per-run cABC plots
    - two-pass cABC plots
    - breakpoint summary plot
    - all three-methods summary plots
    - core Group A heatmap
    - fairness drift plot

    The only behavioral addition is timestamped file logging under logs/.
    """
    log_path = setup_run_logger("run_xai_agreement")
    start_time = time.time()

    try:
        ensure_dirs()

        # Validate outputs
        existing = []
        for ds in TARGET_DATASETS:
            dataset_output_dir = OUTPUT_DIR / ds
            if dataset_output_dir.exists():
                existing.append(ds)
                logging.info("Dataset output folder found: %s", dataset_output_dir)
            else:
                logging.warning("Dataset output folder not found: %s", dataset_output_dir)

        if len(existing) < 2:
            raise ValueError(f"Need at least 2 dataset output folders. Found: {existing}")

        logging.info("Using datasets: %s", existing)
        logging.info("Using models: %s", TARGET_MODELS)
        logging.info("Alpha levels: %s", ALPHA_LEVELS)
        logging.info("RBO p: %.2f", RBO_P)

        # 1) cABC groups for every run
        logging.info("STEP 01 | Building cABC groups and per-run plots")
        cabc_groups_df = build_cabc_groups()
        if cabc_groups_df.empty:
            raise ValueError("No cABC groups were generated. Check SHAP/FI inputs.")

        save_csv(cabc_groups_df, "cabc_groups.csv")
        logging.info("Saved cabc_groups.csv")

        # Build lookup
        logging.info("STEP 02 | Building cABC lookup")
        cabc_lookup: Dict[Tuple[str, str, str], CABCResult] = {}
        for ds in TARGET_DATASETS:
            for model in TARGET_MODELS:
                for method in ["SHAP", "FI"]:
                    df = load_importance(ds, model, method)
                    if df is None:
                        continue
                    cabc_lookup[(ds, model, method)] = build_cabc(df)
        logging.info("Built cABC lookup entries: %d", len(cabc_lookup))

        # 2) Within-model agreement (SHAP vs FI)
        logging.info("STEP 03 | Building within-model agreement CSVs")
        within_model_df, within_model_cabc_df = build_within_model_agreement(cabc_lookup)
        save_csv(within_model_df, "within_model_agreement.csv")
        save_csv(within_model_cabc_df, "within_model_cabc.csv")
        logging.info("Saved within-model agreement CSVs")

        # 3) Cross-model SHAP agreement
        logging.info("STEP 04 | Building cross-model SHAP agreement CSVs")
        cross_model_shap_df, cross_model_shap_cabc_df = build_cross_model_shap_agreement(cabc_lookup)
        save_csv(cross_model_shap_df, "cross_model_shap_agreement.csv")
        save_csv(cross_model_shap_cabc_df, "cross_model_shap_cabc.csv")
        logging.info("Saved cross-model SHAP agreement CSVs")

        # 4) Temporal stability (for SHAP and FI)
        logging.info("STEP 05 | Building temporal stability CSVs")
        temporal_df, temporal_cabc_df = build_temporal_stability(cabc_lookup)
        save_csv(temporal_df, "temporal_stability.csv")
        save_csv(temporal_cabc_df, "temporal_stability_cabc.csv")
        logging.info("Saved temporal stability CSVs")

        # 5) Summary plots
        logging.info("STEP 06 | Building summary plots")
        breakpoint_path = plot_breakpoint_summary(cabc_groups_df)
        logging.info("Saved cABC breakpoint summary plot: %s", breakpoint_path)

        three_paths = plot_three_methods_for_all_runs(within_model_df, within_model_cabc_df)
        logging.info("Saved %d three-methods summary plots", len(three_paths))

        heatmap_path = plot_core_groupA_heatmap(cabc_groups_df)
        if heatmap_path is not None:
            logging.info("Saved core Group A heatmap: %s", heatmap_path)
        else:
            logging.warning("Core Group A heatmap was not generated")

        fairness_path = plot_fairness_drift()
        if fairness_path is not None:
            logging.info("Saved fairness drift plot: %s", fairness_path)
        else:
            logging.warning("Fairness drift plot was not generated")

        logging.info("Saved cABC summary plots")

        elapsed = time.time() - start_time
        logging.info("=" * 80)
        logging.info("END SCRIPT: run_xai_agreement.py")
        logging.info("Status: SUCCESS")
        logging.info("Elapsed time: %.2f seconds", elapsed)
        logging.info("All XAI agreement outputs saved to: %s", AGREEMENT_DIR)
        logging.info("Execution log saved to: %s", log_path)
        logging.info("=" * 80)

    except Exception:
        elapsed = time.time() - start_time
        logging.exception("Status: FAILED after %.2f seconds", elapsed)
        logging.info("Execution log saved to: %s", log_path)
        raise


if __name__ == "__main__":
    run_xai_agreement_full()
