"""
Checks for the v29 outcome-definition control analyses.

Run with:
    python -m pytest tests/test_v32.py
    python -m tests.test_v32
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import label_controls_core as C
from src.analysis.run_detection_power import inject, k_needed, metrics
from src.analysis.run_label_null import null_frames


def test_matched_labels_have_diag_prevalence() -> None:
    for outcome, lab in [("diabetes", "labM"), ("hypertension", "measM")]:
        for cycle in C.CYCLES:
            df = C.load_base(outcome, cycle)
            assert df[lab].sum() == int(round(len(df) * df["diag"].mean()))
            again = C.load_base(outcome, cycle)
            assert (df[lab] == again[lab]).all()


def test_null_labels_reproduce_the_real_two_by_two_table(tmp_path=None) -> None:
    import tempfile
    from pathlib import Path
    d = Path(tmp_path or tempfile.mkdtemp())
    frames = null_frames("diabetes", 3, d)
    for cycle in C.CYCLES:
        df = C.load_base("diabetes", cycle)
        real = pd.crosstab(df.diag, df.lab)
        for col in frames[cycle].columns:
            null = pd.crosstab(df.diag, frames[cycle][col])
            assert (real.to_numpy() == null.to_numpy()).all()


def test_split_ignores_the_alternative_label() -> None:
    df = C.load_base("diabetes", C.CYCLES[0])
    tr1, te1 = C.split(df, 3)
    df2 = df.copy()
    df2["labM"] = 1 - df2["labM"]
    tr2, te2 = C.split(df2, 3)
    assert list(tr1) == list(tr2) and list(te1) == list(te2)


def test_core_shift_reproduces_jbi_v1() -> None:
    for outcome, n in [("diabetes", 13), ("hypertension", 34)]:
        v = C.v1_vectors(outcome, labels=("diag", C.SECOND[outcome]))
        assert len(C.replicated(C.shift_table(v, "diag", C.SECOND[outcome]))) == n
        assert len(C.v1_replicated(outcome)) == n


def test_injection() -> None:
    p = pd.Series(np.array([0.35, 0.25, 0.15, 0.1, 0.1, 0.05]), index=list("abcdef"))
    q = inject(p, "b", 0.05)
    assert abs(q.sum() - 1) < 1e-12 and (q >= 0).all() and abs(q["b"] - 0.30) < 1e-12
    q0 = inject(p, "b", 0.0)
    assert np.allclose(q0, p)
    m = metrics(p, q0)
    assert m["J_A"] == 1.0 and m["spearman"] > 0.999
    assert inject(p, "d", -0.5)["d"] == 0.0


def test_binomial_rule() -> None:
    assert k_needed(10) == 3 and k_needed(5) == 2



def test_cabc_standard_uniform() -> None:
    """Published cABC limit (min of Pareto and BreakEven) on a uniform
    distribution keeps about 41% of items; the BreakEven rule keeps 50%."""
    import numpy as np
    import pandas as pd
    from src.evaluation.agreement_metrics import cabc_group_a_standard, cabc_partition
    u = pd.Series(np.linspace(1, 100, 100), index=range(100))
    assert len(cabc_partition(u).group_a) == 50
    assert 40 <= len(cabc_group_a_standard(u)) <= 43



def test_gamma_objective_reproduces_choice() -> None:
    import csv
    import pandas as pd
    from src.analysis import label_controls_core as C
    from src.analysis.run_label_null import GAMMA_USED, select_gamma
    for o in ["diabetes", "hypertension"]:
        rows = list(csv.reader(open(C.OUT_DIR / "label_null" / f"calibration_{o}.csv")))[3:]
        idx = pd.MultiIndex.from_tuples([(r[0], float(r[1])) for r in rows])
        summ = pd.DataFrame([[float(x) for x in r[2:]] for r in rows], index=idx)
        assert select_gamma(summ) == GAMMA_USED[o], o


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
