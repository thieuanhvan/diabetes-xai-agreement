"""
Reproducibility table for the journal extension (requested by thầy Nghiệp,
09/05/2026: "thêm 1 table chi tiết cho reproducibility").

Collects, from the code constants and the stored outputs, every setting a
reader needs to rerun each analysis: data, sample sizes, seeds, models and
hyperparameters, attribution settings, thresholds, statistics, runtime and
package versions.

Writes outputs/reproducibility/settings.csv and settings.md
Usage:
    python -m src.analysis.build_reproducibility_table
"""

from __future__ import annotations

import importlib.metadata as md
import platform
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs"
RESULT_DIR = OUT / "reproducibility"

MODELS_TXT = ("LR: max_iter=1000; RF: n_estimators=200, unconstrained depth; "
              "XGB: n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, "
              "colsample_bytree=0.8; StandardScaler before every model; random_state = seed")
CW_TXT = "pipeline (MAPR): LR, RF class_weight=balanced, XGB unweighted; ablation: none / balanced"
PI_TXT = "sklearn permutation_importance, scoring=roc_auc, n_repeats=10, random_state=seed, signed mean kept"


def _perf(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _runtime(perf: pd.DataFrame | None) -> str:
    if perf is None:
        return "not run"
    cols = [c for c in ("fit_s", "shap_s", "pi_s") if c in perf]
    med = perf.groupby("model")[cols].median().sum(axis=1)
    return "; ".join(f"{m}: {v:.0f} s/run" for m, v in med.items())


def build() -> pd.DataFrame:
    rows = []
    rows.append(dict(
        analysis="MAPR baseline metric comparison (8a)",
        script="src.analysis.run_metric_comparison",
        data="BRFSS 2015/2021/2023, 17 common predictors; 18 stored attribution vectors (tag mapr2026-v1.0)",
        sample="253,680 / 236,378 / 272,769 rows; 80/20 split, seed 42 (MAPR)",
        seeds="1 (MAPR vectors, no retraining)",
        models=MODELS_TXT, attribution="SHAP (TreeSHAP 200 rows; LinearSHAP LR), PI as in MAPR",
        metrics="cABC J_A/J_B/J_C, overlap coefficient, size ratio, contradiction, top-K (5,6,7,10), "
                "Spearman, Kendall tau-b, weighted tau, RBO_ext (p=0.9/0.8/0.5)",
        statistics="45 pairs; descriptive", runtime="seconds"))
    rows.append(dict(
        analysis="Patient-level bootstrap of Group A",
        script="src.analysis.run_cabc_bootstrap",
        data="BRFSS per-patient SHAP (XGB, LR), 2,000 test patients per cohort",
        sample="2,000 patients x 3 cohorts", seeds="bootstrap B=2000, seed 42",
        models="fixed (MAPR models)", attribution="stored per-patient SHAP",
        metrics="P(|A|=k), XGB-LR J_A distribution, Spearman 95% interval",
        statistics="percentile bootstrap", runtime="about 10 s"))
    perf = _perf(OUT / "brfss_multiseed" / "pipeline" / "performance.csv")
    rows.append(dict(
        analysis="BRFSS multi-seed grid (MAPR factorial re-run)",
        script="src.analysis.run_brfss_multiseed",
        data="BRFSS 2015/2021/2023, 17 common predictors",
        sample="80/20 stratified split per seed",
        seeds="5 (0-4); class-weight ablation 3 (0-2)",
        models=MODELS_TXT + " | " + CW_TXT,
        attribution="SHAP on 2,000 test rows (XGB, LR) and 200 rows (RF, exact TreeSHAP ~5 s/row); "
                    + PI_TXT + "; PI on a stratified 10,000-row test subsample",
        metrics="as above, plus seed axis (noise floor)",
        statistics="mean and 2.5/97.5 percentiles over pairs", runtime=_runtime(perf)))
    perf = _perf(OUT / "label_axis" / "performance.csv")
    n_txt = "not run"
    if perf is not None:
        n = perf.groupby("cycle")[["n_train", "n_test"]].first()
        n_txt = "; ".join(f"{c}: train {r.n_train}, test {r.n_test}" for c, r in n.iterrows())
    rows.append(dict(
        analysis="NHANES label axis (diagnosis vs HbA1c vs total)",
        script="src.analysis.run_label_axis",
        data="NHANES 2017-Mar 2020 (P_*) and 2021-2023 (*_L), adults >= 18, borderline excluded, "
             "valid HbA1c, 14 BRFSS-counterpart features (nhanes-diabetes, branch label-axis-columns)",
        sample=n_txt + "; same split for all labels, stratified on (diagnosis, HbA1c) cell",
        seeds="10 (0-9)", models=MODELS_TXT + " | " + CW_TXT.split(";")[0],
        attribution="SHAP on all test rows; " + PI_TXT + "; unweighted and MEC-weighted (WTMEC) versions",
        metrics="as above; feature-level change in attribution share",
        statistics="Nadeau-Bengio corrected resampled t-test, Benjamini-Hochberg within "
                   "cycle x weighting x method x label pair (42 tests)",
        runtime=_runtime(perf)))
    perf_tw = _perf(OUT / "label_axis_trainweighted" / "performance.csv")
    rows.append(dict(
        analysis="NHANES label axis, survey-weighted training (sensitivity)",
        script="src.analysis.run_label_axis --train-weighted",
        data="as label axis", sample="as label axis", seeds="10 (0-9)",
        models="as label axis; WTMEC passed as sample_weight at fit time (combined with class weighting)",
        attribution="as label axis", metrics="as label axis",
        statistics="as label axis; compared with the unweighted run feature by feature",
        runtime=_runtime(perf_tw)))
    rows.append(dict(
        analysis="Fairness by label (NHANES)", script="src.analysis.run_label_fairness",
        data="as label axis", sample="as label axis", seeds="10 (0-9)", models=MODELS_TXT,
        attribution="none (predictions only)",
        metrics="TPR, FPR, AUC per group; TPR gap (weighted by WTMEC); groups: Income PIR<2 vs >=2, "
                "Education HS or less vs more, Sex, Age 18-44/45-64/65+",
        statistics="capacity-matched threshold (flag rate = training-label prevalence); "
                   "percentiles over seeds", runtime="about 1.5 min total"))
    rows.append(dict(
        analysis="Variance decomposition (model x method x label/year)",
        script="src.analysis.run_variance_decomposition",
        data="vectors from the label axis and the BRFSS multi-seed grid", sample="-",
        seeds="replicates = seeds", models="-", attribution="-",
        metrics="pooled eta^2 per source; response = attribution share and within-vector rank",
        statistics="balanced full-factorial ANOVA per feature, SS pooled over features",
        runtime="seconds"))
    rows.append(dict(
        analysis="Instance vs population agreement; univariate core check",
        script="src.analysis.run_instance_vs_population",
        data="BRFSS per-patient SHAP (XGB, LR); NHANES seed-0 split",
        sample="2,000 BRFSS patients per cohort; NHANES full test split", seeds="1",
        models="XGB, LR", attribution="per-patient |SHAP|",
        metrics="per-patient top-5 overlap and Spearman vs population values; univariate |AUC-0.5| ranking",
        statistics="descriptive (quartiles, share below population value)", runtime="about 30 s"))
    rows.append(dict(
        analysis="Correction of the MAPR Income-fairness result",
        script="src.analysis.run_brfss_income_harmonised",
        data="MAPR per-bin fairness outputs (n, n_positive, TPR)",
        sample="MAPR test split (seed 42)", seeds="1", models="MAPR models", attribution="-",
        metrics="delta-TPR, native bins (8/11/7) vs 5 common bins",
        statistics="pooled TP / pooled positives", runtime="seconds"))
    for cw in ["pipeline", "none"]:
        perf_h = _perf(OUT / f"label_axis_hypertension_{cw}" / "performance.csv")
        rows.append(dict(
            analysis=f"Hypertension transfer demonstration (class weighting: {cw})",
            script=f"src.analysis.run_label_axis --outcome hypertension --class-weighting {cw} --seeds 5",
            data="NHANES as label axis, without the HbA1c restriction; labels diag (BPQ020), measured "
                 "(mean oscillometric SBP >= 140 or DBP >= 90 mmHg), composite; diagnosed diabetes "
                 "replaces HighBP among the 14 features; blood pressure never a feature",
            sample="6,090 and 4,314 adults; same split for all labels", seeds="5 (0-4)",
            models=MODELS_TXT, attribution="as label axis",
            metrics="as label axis", statistics="corrected resampled t-test, BH; cross-cycle replication",
            runtime=_runtime(perf_h)))
    rows.append(dict(
        analysis="Sensitivity analyses", script="src.analysis.run_sensitivity_checks --n-perm 999",
        data="vectors and pairs of all grids; stored per-patient SHAP (BRFSS)", sample="-", seeds="-",
        models="-", attribution="-",
        metrics="crossed-seed and CLR decompositions, per-feature eta^2, permutation p (B = 999, "
                "p = (b + 1)/(B + 1)), seed-reference percentiles 1/5/10, exact sign-flip test, "
                "signed patient-level SHAP",
        statistics="as listed", runtime="about 1 min"))
    rows.append(dict(
        analysis="Feature dependence and parameter checks", script="src.analysis.run_dependence_checks",
        data="NHANES and BRFSS predictors; label-axis and BRFSS multi-seed vectors", sample="-", seeds="-",
        models="-", attribution="-",
        metrics="Spearman correlation between predictors; shares summed within six predictor domains; "
                "RBO p = 0.8, 0.9, 0.95; negative PI clipped, absolute or ranked",
        statistics="corrected resampled t-test, BH, cross-cycle replication (domain and PI checks)",
        runtime="about 3 min"))
    rows.append(dict(
        analysis="Comparison with standard disagreement metrics", script="src.analysis.run_soa_comparison",
        data="conference vectors; BRFSS multi-seed vectors (two regimes); NHANES label-axis vectors", sample="-",
        seeds="-", models="-", attribution="-",
        metrics="Krishna et al. feature agreement and rank agreement (top 5), rank correlation, pairwise rank agreement",
        statistics="means; share of label pairs below the seed 5th percentile", runtime="seconds"))
    return pd.DataFrame(rows)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    t = build()
    t.to_csv(RESULT_DIR / "settings.csv", index=False)
    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap"]
    env = ", ".join(f"{p} {md.version(p)}" for p in pkgs)
    lines = [f"# Reproducibility settings\n", f"Python {platform.python_version()}; {env}\n"]
    for _, r in t.iterrows():
        lines.append(f"## {r['analysis']}\n")
        for c in t.columns[1:]:
            lines.append(f"- **{c}**: {r[c]}")
        lines.append("")
    (RESULT_DIR / "settings.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULT_DIR / 'settings.csv'} and settings.md ({len(t)} analyses)")


if __name__ == "__main__":
    main()
