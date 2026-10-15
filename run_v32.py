"""
Runs the v29 outcome-definition control analyses in order (Spec A and B).
Right-click -> Run in PyCharm, or: python run_v32.py
Approximate time on one laptop CPU: B minutes; A.1/A.2/A.4 ~4 h; A.3 null (R=50) ~5 h.
Every fitting step is resumable: rerunning skips what is already in outputs/.
"""
import runpy
import sys

STEPS = [
    ("src.analysis.run_detection_power", []),
    ("src.analysis.run_label_controls", ["--stage", "all", "--outcome", "both"]),
    ("src.analysis.run_label_null", ["--outcome", "diabetes", "--R", "50"]),
    ("src.analysis.run_label_null", ["--outcome", "diabetes", "--R", "30", "--gamma", "0.3"]),
    ("src.analysis.run_label_null", ["--outcome", "hypertension", "--R", "30"]),
    ("src.analysis.run_label_null", ["--outcome", "hypertension", "--R", "30", "--gamma", "0.35"]),
    ("src.analysis.run_label_null", ["--outcome", "diabetes", "--R", "30", "--gamma", "0.4"]),
    ("src.analysis.run_label_null", ["--outcome", "hypertension", "--R", "30", "--gamma", "0.25"]),
    ("src.analysis.run_null_validation", ["--stage", "all"]),
    ("src.analysis.run_reviewer_checks", []),
    ("src.analysis.run_holdout_cycle", ["--outcome", "diabetes", "--stage", "all"]),
    ("src.analysis.run_holdout_cycle", ["--outcome", "hypertension", "--stage", "all"]),
    ("src.analysis.make_supplementary_tables", []),
    ("src.analysis.make_manuscript_outputs", []),
    ("tests.test_v32", []),
]

if __name__ == "__main__":
    for mod, args in STEPS:
        print(f"== {mod} {' '.join(args)}", flush=True)
        sys.argv = [mod] + args
        runpy.run_module(mod, run_name="__main__")
