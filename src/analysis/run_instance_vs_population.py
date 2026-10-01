"""
Two reviewer questions answered from the same per-patient attributions.

(1) Population vs individual level (MAPR reviewer R#2). Global agreement
    between two explanations can hide disagreement for individual patients.
    For pairs of per-patient |SHAP| vectors on the SAME patients we compare
    population-level agreement (top-5 overlap and Spearman of the mean |SHAP|
    vectors) with the distribution of the same metrics computed patient by
    patient.
        BRFSS: XGBoost vs LR, existing per-patient SHAP
               (outputs/boundary_analysis/shap_per_patient, 2,000 test
               patients per cohort, seed 42)
        NHANES: XGBoost vs LR (model pair) and diag vs lab (label pair),
               computed here on the paired test split, seed 0

(2) Is the five-feature core trivial? (MAPR reviewer R#1, "tautological")
    Rank the features by a model-free univariate association with the label
    (|AUC - 0.5| of the single feature) and compare that ranking with the
    attribution-based core. If the univariate top-5 already equals the core,
    the core is recoverable without any model or XAI method.

Outputs (outputs/instance_vs_population/):
    instance_level.csv     per patient: top-5 overlap, Spearman
    instance_summary.csv   population value vs instance median/quartiles
    univariate_rank.csv    univariate |AUC - 0.5| ranking per dataset/label
    core_check.csv         overlap of univariate top-5 with the attribution core

Usage:
    python -m src.analysis.run_instance_vs_population
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.analysis.run_label_axis import FEATURES as NH_FEATURES
from src.analysis.run_label_axis import load_cycle
from src.explainability.global_attribution import _shap_values, build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs"
RESULT_DIR = OUT / "instance_vs_population"
CORE = {"Age", "BMI", "GenHlth", "HighBP", "HighChol"}
BRFSS_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income",
]


def _top(v: np.ndarray, k: int) -> set:
    return set(np.argsort(-v, kind="stable")[:k])


def compare_instances(A: np.ndarray, B: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Row-wise top-k overlap and Spearman between two |SHAP| matrices."""
    top = [len(_top(a, k) & _top(b, k)) / k for a, b in zip(A, B)]
    rho = [spearmanr(a, b).statistic if a.std() > 0 and b.std() > 0 else np.nan for a, b in zip(A, B)]
    return pd.DataFrame({"top5": top, "spearman": rho})


def population(A: np.ndarray, B: np.ndarray, k: int = 5) -> dict:
    a, b = A.mean(0), B.mean(0)
    return {"top5": len(_top(a, k) & _top(b, k)) / k, "spearman": float(spearmanr(a, b).statistic)}


def brfss_pairs() -> list[tuple]:
    base = OUT / "boundary_analysis" / "shap_per_patient"
    out = []
    for year in ["2015", "2021", "2023"]:
        d = base / f"cdc_brfss_diabetes_{year}"
        names = [x.strip() for x in (d / "feature_names.txt").read_text().splitlines() if x.strip()]
        A = np.abs(np.load(d / "xgboost_shap_values.npy")).astype(float)
        B = np.abs(np.load(d / "logistic_regression_shap_values.npy")).astype(float)
        out.append(("BRFSS", year, "model: XGB vs LR", A, B, names))
    return out


def nhanes_pairs(seed: int = 0) -> list[tuple]:
    out = []
    for cycle in ["2017-2020", "2021-2023"]:
        df = load_cycle(cycle)
        strata = df["diag"].astype(str) + df["lab"].astype(str)
        tr, te = train_test_split(df.index, test_size=0.2, random_state=seed, stratify=strata)
        S = {}
        for label in ["diag", "lab"]:
            for model in ["xgboost", "logistic_regression"]:
                pipe = build_model(model, seed).fit(df.loc[tr, NH_FEATURES], df.loc[tr, label])
                Xs = pipe.named_steps["scaler"].transform(df.loc[te, NH_FEATURES])
                S[(label, model)] = np.abs(_shap_values(pipe, model, Xs, Xs))
        out.append(("NHANES", cycle, "model: XGB vs LR (diag)", S[("diag", "xgboost")],
                    S[("diag", "logistic_regression")], NH_FEATURES))
        out.append(("NHANES", cycle, "label: diag vs lab (XGB)", S[("diag", "xgboost")],
                    S[("lab", "xgboost")], NH_FEATURES))
        out.append(("NHANES", cycle, "label: diag vs lab (LR)", S[("diag", "logistic_regression")],
                    S[("lab", "logistic_regression")], NH_FEATURES))
    return out


def univariate() -> pd.DataFrame:
    rows = []
    for year in ["2015", "2021", "2023"]:
        df = pd.read_csv(PROJECT_ROOT / "data" / f"cdc_brfss_diabetes_{year}.csv",
                         usecols=BRFSS_FEATURES + ["Diabetes_binary"])
        for f in BRFSS_FEATURES:
            rows.append(dict(dataset="BRFSS", cohort=year, label="diag", feature=f,
                             strength=abs(roc_auc_score(df["Diabetes_binary"], df[f]) - 0.5)))
    for cycle in ["2017-2020", "2021-2023"]:
        df = load_cycle(cycle)
        for label in ["diag", "lab", "total"]:
            for f in NH_FEATURES:
                rows.append(dict(dataset="NHANES", cohort=cycle, label=label, feature=f,
                                 strength=abs(roc_auc_score(df[label], df[f]) - 0.5)))
    u = pd.DataFrame(rows)
    u["rank"] = u.groupby(["dataset", "cohort", "label"])["strength"].rank(ascending=False)
    return u


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    inst_rows, summ_rows = [], []
    for dataset, cohort, pair, A, B, names in brfss_pairs() + nhanes_pairs():
        inst = compare_instances(A, B)
        inst.insert(0, "pair", pair)
        inst.insert(0, "cohort", cohort)
        inst.insert(0, "dataset", dataset)
        inst_rows.append(inst)
        pop = population(A, B)
        for metric in ["top5", "spearman"]:
            s = inst[metric].dropna()
            summ_rows.append(dict(dataset=dataset, cohort=cohort, pair=pair, metric=metric,
                                  n_patients=len(s), population=pop[metric],
                                  instance_q25=s.quantile(0.25), instance_median=s.median(),
                                  instance_q75=s.quantile(0.75),
                                  share_below_population=float((s < pop[metric]).mean())))
    pd.concat(inst_rows).round(4).to_csv(RESULT_DIR / "instance_level.csv", index=False)
    summ = pd.DataFrame(summ_rows)
    summ.round(4).to_csv(RESULT_DIR / "instance_summary.csv", index=False)
    logging.info("Population vs instance level:\n%s", summ.round(3).to_string(index=False))

    u = univariate()
    u.round(5).to_csv(RESULT_DIR / "univariate_rank.csv", index=False)
    core = (u[u["rank"] <= 5].groupby(["dataset", "cohort", "label"])["feature"]
            .apply(lambda s: sorted(s)).reset_index(name="univariate_top5"))
    core["overlap_with_core"] = core["univariate_top5"].map(lambda s: len(set(s) & CORE))
    core["univariate_top5"] = core["univariate_top5"].map(",".join)
    core.to_csv(RESULT_DIR / "core_check.csv", index=False)
    logging.info("Univariate top-5 vs attribution core:\n%s", core.to_string(index=False))


if __name__ == "__main__":
    main()
