"""
Checks for the journal-extension analysis helpers.

Run with:
    python -m pytest tests/test_journal_analyses.py
    python -m tests.test_journal_analyses
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.run_variance_decomposition import balanced_anova
from src.explainability.global_attribution import build_model


def test_balanced_anova_partitions_total_sum_of_squares() -> None:
    rng = np.random.default_rng(0)
    rows = [dict(A=a, B=b, C=c, y=a + 0.5 * b + 0.2 * c + (a == 2) * (b == 1) + rng.normal(0, 0.1))
            for a in range(3) for b in range(2) for c in range(2) for _ in range(5)]
    d = pd.DataFrame(rows)
    ss = balanced_anova(d, ["A", "B", "C"], "y")
    parts = sum(v for k, v in ss.items() if k != "total")
    assert abs(parts - ss["total"]) < 1e-8
    assert ss["A"] > ss["B"] > ss["C"] > 0
    assert ss["A x B"] > ss["A x C"]
    cell_mean = d.groupby(["A", "B", "C"])["y"].transform("mean")
    assert abs(ss["residual (seed)"] - ((d["y"] - cell_mean) ** 2).sum()) < 1e-8


def test_class_weighting_options() -> None:
    lr = build_model("logistic_regression", 0, class_weighting="none").named_steps["model"]
    assert lr.class_weight is None
    rf = build_model("random_forest", 0).named_steps["model"]
    assert rf.class_weight == "balanced"
    xgb_default = build_model("xgboost", 0).named_steps["model"]
    assert xgb_default.get_params()["scale_pos_weight"] == 1.0
    xgb_bal = build_model("xgboost", 0, class_weighting="balanced", pos_ratio=0.2).named_steps["model"]
    assert abs(xgb_bal.get_params()["scale_pos_weight"] - 4.0) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
