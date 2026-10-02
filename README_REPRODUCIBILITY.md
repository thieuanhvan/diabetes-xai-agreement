# Reproducibility Guide

This document describes how to fully reproduce the numerical results
reported in the MAPR 2026 paper (see the How to cite section of
`README.md`). Stability and fairness outputs are included as
exploratory additions and are not required for the conference
results.

The reproduction has two stages. First, the per-cell pipeline trains
each of three models on each of three BRFSS cohorts and computes SHAP
and Permutation Importance attributions, producing 18 attribution CSVs
in total (9 model-cohort cells, 2 methods per cell). Second, the
agreement analysis aggregates the 18 rankings into cABC partitions and
per-group Jaccard tables. Fairness and per-cohort stability outputs
are produced as part of the per-cell pipeline.

## 1. Environment setup

```bash
git clone https://github.com/thieuanhvan/diabetes-xai-agreement.git
cd diabetes-xai-agreement
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12.10 (Anaconda) on Windows 10 (conference results, tag `mapr2026-v1.0`; the journal results in release `jbi-v1` were generated with Python 3.11.15 on Linux, see README). For byte-level
reproducibility, install the exact package versions recorded in
`outputs/cdc_brfss_diabetes_2015/reproducibility/environment.json` from
the reference run.

## 2. Verify input data

The harmonised cohort CSVs are committed to the repository:

```
data/cdc_brfss_diabetes_2015.csv    # 253,680 rows x 22 cols
data/cdc_brfss_diabetes_2021.csv    # 236,378 rows x 22 cols
data/cdc_brfss_diabetes_2023.csv    # 272,769 rows x 18 cols
```

Quick sanity check:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/cdc_brfss_diabetes_2015.csv').shape)"
# Expected output: (253680, 22)
```

The 2015 and 2021 cohorts contain 21 predictors plus the target; the
2023 cohort contains 17 predictors plus the target. The pipeline
restricts every cohort to the 17 predictors common to all three. No
raw `.XPT` files are required; the committed CSVs are the
authoritative inputs.

## 3. Run the per-cell pipeline

```bash
python -m src.pipelines.run_pipeline_all_combos
```

This iterates over the 9 (model, cohort) cells defined in
`src/datasets/dataset_registry.py`. Models are Logistic Regression,
Random Forest, and XGBoost; cohorts are BRFSS 2015, 2021, and 2023.

For each cell the pipeline produces:

```
outputs/<cohort_slug>/
    tables/model_comparison_table.csv          # AUC, accuracy, F1, etc.
    shap/<model>_shap_feature_importance.csv   # SHAP rankings
    fi/<model>_feature_importance.csv          # Permutation Importance
    confusion_matrix/<model>_confusion_matrix.png
    analysis/<model>_fairness_equalized_odds_summary.csv
    analysis/<model>_fairness_{age,sex,income}_detail.csv
    plots/...
    reproducibility/
        experiment_summary.json
        environment.json
```

## 4. Run the agreement and stability analysis

```bash
python -m src.analysis.run_xai_agreement
```

This reads the 18 attribution CSVs from Step 3 and writes the
aggregate tables and figures into `outputs/xai_agreement/`:

```
outputs/xai_agreement/
    cabc_groups.csv                    # Group A/B/C for each ranking
    within_model_agreement.csv         # SHAP vs PI cross-method (9 cells)
    cross_model_shap_agreement.csv     # SHAP cross-model (9 pairs)
    temporal_stability_cabc.csv        # Group A stability across cohort pairs
    temporal_stability.csv             # Legacy stability metrics (auxiliary)
    spearman_summary.csv               # Rank correlations (auxiliary)
    plots/cabc_partition.png           # Figure 1
    plots/crossmodel_jaccard.png       # Figure 2
```

## 5. Verify against the paper

After Step 4 completes, the following invariants should hold. They are
grouped by analytical axis. The acceptance test below covers the
agreement axis; stability and fairness outputs are exploratory.

### 5a. Agreement axis

Headline aggregates:

| Metric | Expected | Source file |
|---|---|---|
| Cross-method aggregate J_A     | 0.9127 | `within_model_agreement.csv` + cABC |
| Perfect cells (cross-method)   | 5 / 9  | same |
| Cross-model aggregate J_A      | 0.8307 | `cross_model_shap_agreement.csv` + cABC |
| Core feature set (18/18 in A)  | {Age, BMI, GenHlth, HighBP, HighChol} | `cabc_groups.csv` |

Per-cell cross-method J_A (SHAP vs PI):

| Cohort | XGBoost | Random Forest | Logistic Reg |
|---|---|---|---|
| 2015 | 1.0000 | 0.8333 | 1.0000 |
| 2021 | 1.0000 | 0.8333 | 0.8333 |
| 2023 | 1.0000 | 0.7143 | 1.0000 |

Per-pair cross-model J_A (SHAP, by cohort):

| Pair | 2015 | 2021 | 2023 | Mean |
|---|---|---|---|---|
| XGB-LR | 1.0000 | 0.8333 | 1.0000 | 0.9444 |
| XGB-RF | 0.8333 | 0.8333 | 0.7143 | 0.7936 |
| RF-LR  | 0.8333 | 0.7143 | 0.7143 | 0.7540 |

Test-set AUC (reported in the paper):

| Model | 2015 | 2021 | 2023 |
|---|---|---|---|
| XGBoost             | 0.8257 | 0.8228 | 0.8134 |
| Random Forest       | 0.7817 | 0.7832 | 0.7693 |
| Logistic Regression | 0.8185 | 0.8156 | 0.8061 |

### 5b. Temporal stability axis

Group-A Jaccard across cohort pairs (2015-2021, 2015-2023, 2021-2023),
per (model, method). Source:
`outputs/xai_agreement/temporal_stability_cabc.csv`, column `J_A`.

| Method                | Aggregate J_A | Perfect cells (J = 1.0) |
|-----------------------|---|---|
| SHAP                  | 0.9312 | 5 / 9 |
| Permutation Importance | 1.0000 | 9 / 9 |
| Aggregate (both)      | 0.9656 | 14 / 18 |

By model and method:

| Model | SHAP (3 cohort pairs) | Permutation Importance (3 cohort pairs) |
|-------|---|---|
| XGBoost              | 1.0000, 1.0000, 1.0000 | 1.0000, 1.0000, 1.0000 |
| Random Forest        | 1.0000, 0.8571, 0.8571 | 1.0000, 1.0000, 1.0000 |
| Logistic Regression  | 0.8333, 1.0000, 0.8333 | 1.0000, 1.0000, 1.0000 |

### 5c. Fairness axis

Equalised Odds violation per (model, cohort, protected attribute).
Source: `outputs/<cohort>/analysis/<model>_fairness_equalized_odds_summary.csv`,
column `eo_violation` (defined as max(delta_tpr, delta_fpr)).

Income:

| Model | 2015 | 2021 | 2023 |
|-------|------|------|------|
| XGBoost              | 0.2422 | 0.3153 | 0.3047 |
| Random Forest        | 0.1090 | 0.1923 | 0.2090 |
| Logistic Regression  | 0.4316 | 0.4914 | 0.4099 |

Age:

| Model | 2015 | 2021 | 2023 |
|-------|------|------|------|
| XGBoost              | 0.2365 | 0.2074 | 0.2122 |
| Random Forest        | 0.2502 | 0.2098 | 0.2291 |
| Logistic Regression  | 0.7902 | 0.7412 | 0.7278 |

Sex:

| Model | 2015 | 2021 | 2023 |
|-------|------|------|------|
| XGBoost              | 0.0054 | 0.0214 | 0.0056 |
| Random Forest        | 0.0182 | 0.0378 | 0.0207 |
| Logistic Regression  | 0.0620 | 0.0551 | 0.0667 |

If all values match, the reproduction is complete. If any value
differs by more than the last reported decimal, check that the Python
version is 3.12.x, all package versions match `requirements.txt`
lower bounds (and ideally match `environment.json` exactly), the
pipeline was run on the unmodified `data/cdc_brfss_diabetes_*.csv`
inputs, and `random_state = 42` is consistent across all stochastic
components.

## 6. Runtime

Approximate wall-clock on a standard desktop CPU, no GPU:

| Step | Time |
|---|---|
| Setup and data verification | under 1 minute |
| Per-cell pipeline (Step 3, 9 cells) | 2 to 3 hours |
| Agreement analysis (Step 4) | 5 to 10 minutes |

Random Forest SHAP and Permutation Importance dominate the wall
clock. The pipeline does not display a progress bar; one INFO log
line is emitted per sub-step. Tail the log file in `logs/` for live
progress:

```bash
# Linux / Mac
tail -f logs/pipeline_*.log

# Windows (PowerShell)
Get-Content logs/pipeline_*.log -Wait
```

## 7. Determinism notes

All splits, samples, and stochastic estimators use `random_state = 42`.
TreeSHAP attributions use a seeded 200-instance test-set sample;
LinearSHAP is closed-form and produces no stochasticity. Permutation
Importance is seeded and runs with `n_jobs = 1`. Using `n_jobs = -1`
on Windows with PyCharm has been observed to fail silently and
produce empty importance files, so the single-process setting is the
reliable choice. Minor wall-clock differences across runs (in
`shap_time_s`, `fi_time_s`) are expected and do not affect numerical
results.

## Contact

Questions about reproducing these results can be raised as a GitHub
issue on this repository.
