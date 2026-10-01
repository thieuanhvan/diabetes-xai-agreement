"""
Checks for src/evaluation/agreement_metrics.py.

Run with either:
    python -m pytest tests/test_agreement_metrics.py
    python -m tests.test_agreement_metrics
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.agreement_metrics import (
    cabc_partition, compare, is_contradiction, jaccard, overlap_coefficient, rbo_ext,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def test_rbo_ext_bounds() -> None:
    r = list("abcdefghijklmnopq")  # 17 items, as in the BRFSS schema
    assert abs(rbo_ext(r, r, 0.9) - 1.0) < 1e-12
    assert abs(rbo_ext(r, r, 0.5) - 1.0) < 1e-12
    assert rbo_ext(r, r[::-1], 0.9) < 0.6
    # Swapping the head must cost more than swapping the tail.
    head = [r[1], r[0]] + r[2:]
    tail = r[:-2] + [r[-1], r[-2]]
    assert rbo_ext(r, head, 0.9) < rbo_ext(r, tail, 0.9)


def test_group_a_is_above_mean_random() -> None:
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(5, 40))
        v = pd.Series(rng.lognormal(0, 1.5, n), index=[f"f{i}" for i in range(n)])
        part = cabc_partition(v)
        assert set(part.group_a) == set(v[v > v.mean()].index)


def test_containment_vs_contradiction() -> None:
    a, b, c = ["x", "y"], ["x", "y", "z"], ["x", "w"]
    assert overlap_coefficient(a, b) == 1.0 and not is_contradiction(a, b)
    assert abs(jaccard(a, b) - 2 / 3) < 1e-12  # nesting alone lowers J_A
    assert overlap_coefficient(a, c) == 0.5 and is_contradiction(a, c)


def _load(year: str, model: str, method: str) -> pd.Series:
    ds = OUTPUT_DIR / f"cdc_brfss_diabetes_{year}"
    path = ds / "shap" / f"{model}_shap_feature_importance.csv" if method == "SHAP" \
        else ds / "fi" / f"{model}_feature_importance.csv"
    df = pd.read_csv(path)
    return pd.Series(df["importance"].abs().to_numpy(), index=df["feature"])


def test_reproduces_mapr_cabc_outputs() -> None:
    """Group A and J_A must match the frozen MAPR CSVs (tag mapr2026-v1.0)."""
    groups = pd.read_csv(OUTPUT_DIR / "xai_agreement" / "cabc_groups.csv")
    for _, row in groups.iterrows():
        method = "SHAP" if row["method"] == "SHAP" else "PI"
        part = cabc_partition(_load(str(row["year"]), row["model"], method))
        assert part.group_a == row["A"].split(","), (row["year"], row["model"], method)
    within = pd.read_csv(OUTPUT_DIR / "xai_agreement" / "within_model_cabc.csv")
    for _, row in within.iterrows():
        y, m = str(row["year"]), row["model"]
        got = compare(_load(y, m, "SHAP"), _load(y, m, "PI"))["J_A"]
        assert abs(round(got, 4) - row["J_A"]) < 1e-9, (y, m)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
