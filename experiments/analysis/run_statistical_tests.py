"""
Statistical Significance Tests (Paired)
- Use stability raw results to compare proposed models vs best baseline
- Provides permutation test p-value (no SciPy required)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from utils.project_paths import get_outputs_dir, get_log_file


DEFAULT_METRIC = "roc_auc"
N_PERMUTATIONS = 5000


def setup_logging():
    log_file = get_log_file("stats")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 60)
    logging.info("STATISTICAL TESTS STARTED")
    logging.info("=" * 60)


def permutation_test_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERMUTATIONS) -> float:
    """
    Paired permutation test on mean difference.
    H0: mean(a-b) == 0
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b

    observed = np.mean(d)
    count = 0

    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(d))
        perm_stat = np.mean(d * signs)
        if abs(perm_stat) >= abs(observed):
            count += 1

    return (count + 1) / (n_perm + 1)


def select_best_baseline(df: pd.DataFrame, metric: str) -> str:
    """
    Choose best baseline model by average metric.
    Baselines are those NOT starting with 'proposed_'.
    """
    baseline_df = df[~df["model"].str.startswith("proposed_")].copy()
    avg = baseline_df.groupby("model")[metric].mean().sort_values(ascending=False)
    return str(avg.index[0])


def main():
    setup_logging()

    in_dir = get_outputs_dir("analysis/stability")
    raw_path = in_dir / "stability_raw_all_datasets.csv"

    # If user only has per-dataset raw files, auto-merge
    if not raw_path.exists():
        files = sorted(in_dir.glob("stability_raw_*.csv"))
        if not files:
            raise FileNotFoundError("No stability_raw_*.csv found. Run run_model_stability_analysis.py first.")

        raw_df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        raw_path = in_dir / "stability_raw_all_datasets.csv"
        raw_df.to_csv(raw_path, index=False)
        logging.info(f"Auto-merged raw stability into: {raw_path}")
    else:
        raw_df = pd.read_csv(raw_path)

    metric = DEFAULT_METRIC
    out_rows = []

    for dataset in sorted(raw_df["dataset"].unique()):
        ds_df = raw_df[raw_df["dataset"] == dataset].copy()

        best_baseline = select_best_baseline(ds_df, metric)
        proposed_models = sorted([m for m in ds_df["model"].unique() if m.startswith("proposed_")])

        logging.info(f"Dataset={dataset} | Best baseline={best_baseline} | Proposed={proposed_models}")

        # Use paired values by seed
        base = ds_df[ds_df["model"] == best_baseline].sort_values("seed")
        base_seeds = base["seed"].to_numpy()

        for pm in proposed_models:
            prop = ds_df[ds_df["model"] == pm].sort_values("seed")

            # Align by seed intersection
            merged = pd.merge(
                base[["seed", metric]],
                prop[["seed", metric]],
                on="seed",
                suffixes=("_base", "_prop"),
                how="inner",
            )

            a = merged[f"{metric}_prop"].to_numpy()
            b = merged[f"{metric}_base"].to_numpy()

            if len(a) < 3:
                p = np.nan
                logging.warning(f"Not enough paired samples for dataset={dataset}, model={pm}")
            else:
                p = permutation_test_pvalue(a, b)

            out_rows.append({
                "dataset": dataset,
                "metric": metric,
                "baseline": best_baseline,
                "proposed": pm,
                "n_pairs": int(len(a)),
                "mean_proposed": float(np.mean(a)) if len(a) else np.nan,
                "mean_baseline": float(np.mean(b)) if len(b) else np.nan,
                "mean_diff": float(np.mean(a - b)) if len(a) else np.nan,
                "p_value_perm": float(p) if p == p else np.nan,
            })

    out_df = pd.DataFrame(out_rows)
    out_dir = get_outputs_dir("analysis/stats")
    out_path = out_dir / "paired_permutation_tests.csv"
    out_df.to_csv(out_path, index=False)

    logging.info(f"Saved statistical tests: {out_path}")
    logging.info("STATISTICAL TESTS COMPLETED")


if __name__ == "__main__":
    main()