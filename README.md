# diabetes-xai-agreement

Research pipeline auditing explainable AI methods on population-scale
diabetes risk prediction from CDC BRFSS data (2015, 2021, 2023). This
repository reproduces the results reported in the MAPR 2026 paper listed
under [How to cite](#how-to-cite). Additional exploratory outputs are
included for completeness but are not required for the conference
results.

> **Runtime note.** A full reproduction from scratch takes approximately
> two to three hours on a standard desktop CPU; no GPU is required. The
> Random Forest SHAP and Permutation Importance stages dominate the wall
> clock. If only the headline numbers are needed, the canonical
> reference outputs are committed under `outputs/` and can be inspected
> directly without re-execution.

## How to cite

If you found our work useful, please cite us.

For the cABC-based XAI agreement audit and the harmonised multi-year
BRFSS evaluation protocol implemented in this repository, please cite:

Van Thieu, Hung-Nghiep Tran. Auditing Population-Level XAI Agreement
with cABC: Evidence from Diabetes Risk Prediction. *International
Conference on Multimedia Analysis and Pattern Recognition (MAPR)*, 2026.

```bibtex
@inproceedings{thieu2026auditing,
  title     = {Auditing Population-Level {XAI} Agreement with {cABC}:
               Evidence from Diabetes Risk Prediction},
  author    = {Thieu, Van and Tran, Hung-Nghiep},
  booktitle = {Proceedings of the 2026 International Conference on
               Multimedia Analysis and Pattern Recognition (MAPR)},
  year      = {2026},
  publisher = {IEEE},
  note      = {To appear}
}
```

The IEEE Xplore DOI will be added here once the proceedings are
published.

## What the pipeline computes

The codebase runs three classifiers (Logistic Regression, Random
Forest, XGBoost) on three BRFSS cohorts and computes two
feature-attribution methods per cell (SHAP and Permutation Importance),
producing eighteen attribution rankings in total. From these rankings
the pipeline derives four analytical axes:

1. Inter-method agreement (SHAP vs Permutation Importance) within each
   model-cohort cell, quantified via per-group Jaccard on a data-driven
   computed-ABC (cABC) partition of ranked importance.
2. Cross-model agreement (SHAP rankings across architecture pairs)
   within each cohort.
3. Temporal stability of Group A membership across cohort pairs
   (2015-2021, 2015-2023, 2021-2023).
4. Equalised Odds fairness summaries per (model, cohort) by Age, Sex,
   and Income, with automatic detection of the year-specific Income
   schema (BRFSS 2015 INCOME2, 2021 INCOME3, 2023 _INCOMG1).

The cABC partition follows Ultsch and Lötsch (2015), with the
breakpoint selected at the Lorenz-curve maximum gap; this removes the
arbitrary-K choice of top-K overlap.

## Factorial design

Three classifiers, three cohorts, two XAI methods gives 18 attribution
rankings.

| Dimension | Values |
|-----------|--------|
| Classifiers | Logistic Regression, Random Forest, XGBoost |
| Cohorts     | BRFSS 2015 (n = 253,680), 2021 (236,378), 2023 (272,769) |
| XAI methods | SHAP (TreeSHAP for trees, LinearSHAP for LR), Permutation Importance |
| Total N     | 762,827 respondents across the three cohorts |

The harmonised analysis is restricted to the 17 predictors common to
all three releases. BRFSS 2015 and 2021 each contain 21 features;
BRFSS 2023 contains 17 (the variables `AnyHealthcare`, `Fruits`,
`Veggies`, and `HvyAlcoholConsump` are not retained in the harmonised
2023 schema).

## Project structure

```
diabetes-xai-agreement/
├── configs/
│   └── default.yaml                  # Fixed hyperparameters (no test-set tuning)
├── data/
│   └── cdc_brfss_diabetes_{2015,2021,2023}.csv   # Committed, ~20 MB each
├── src/
│   ├── pipelines/
│   │   ├── run_pipeline.py           # Single (dataset, model) cell
│   │   ├── run_pipeline_all_combos.py  # Iterates 9 cells
│   │   └── step01..step10_*.py       # Pipeline stages
│   ├── analysis/
│   │   ├── run_xai_agreement.py      # Aggregates 18 rankings into cABC + Jaccard
│   │   └── fairness_analysis.py      # Equalised Odds per protected attribute
│   ├── datasets/                     # Loaders + cohort registry
│   ├── models/                       # Baseline + proposal estimators
│   ├── explainability/               # SHAP + Permutation Importance
│   ├── evaluation/                   # Classification metrics
│   └── reporting/                    # Auto-generated tables and plots
├── outputs/                          # Reference run committed
├── logs/                             # Generated during execution; not tracked
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Installation

```bash
git clone https://github.com/thieuanhvan/diabetes-xai-agreement.git
cd diabetes-xai-agreement
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12.10 (Anaconda) on Windows 10. The version
ranges in `requirements.txt` allow patch updates within the tested
major range; exact pins from the reference run are available in
`outputs/<cohort>/reproducibility/environment.json`.

## Running the pipeline

The harmonised CSVs in `data/` are the only inputs required. No raw
`.XPT` files or external downloads are needed to reproduce paper
results.

### Step 1: Train models and compute attributions (about 2 to 3 hours)

```bash
python -m src.pipelines.run_pipeline_all_combos
```

This iterates over the 9 (model, cohort) cells. For each cell, the
script writes a metrics table, SHAP and Permutation Importance CSVs,
confusion matrix plots, per-attribute fairness summaries, and a
reproducibility summary into `outputs/<cohort_slug>/`.

### Step 2: Aggregate agreement analysis (about 5 to 10 minutes)

```bash
python -m src.analysis.run_xai_agreement
```

This consumes the 18 attribution CSVs from Step 1 and writes the
aggregate tables and figures into `outputs/xai_agreement/`.

To verify a fresh run against the committed reference, diff the two
`outputs/` trees. CSVs should be byte-identical except for the
`*_time_s` fields in `experiment_summary.json`, where wall-clock
varies between runs while numerical results do not.

## Acceptance test

The invariants below are present in the committed `outputs/` and should
be reproduced by any clean run.

Only the **agreement axis** corresponds to headline numbers reported in
the MAPR 2026 paper. The **temporal stability** and **fairness** outputs
are exploratory additions produced by this pipeline; they are not audited
in the paper, which explicitly defers both to extension work. They are
included here because they are reproducible artefacts of the same run,
not because they are results of the paper.

### Agreement axis (reported in the paper)

Cross-method Group-A Jaccard (SHAP vs Permutation Importance, by
model-cohort cell):

| Cell                       | Expected J_A |
|----------------------------|---|
| XGBoost, BRFSS 2015        | 1.0000 |
| XGBoost, BRFSS 2021        | 1.0000 |
| XGBoost, BRFSS 2023        | 1.0000 |
| Random Forest, 2015        | 0.8333 |
| Random Forest, 2021        | 0.8333 |
| Random Forest, 2023        | 0.7143 |
| Logistic Regression, 2015  | 1.0000 |
| Logistic Regression, 2021  | 0.8333 |
| Logistic Regression, 2023  | 1.0000 |
| Aggregate (mean over 9 cells) | 0.9127 |

Cross-model SHAP Group-A Jaccard, averaged over the three cohorts:

| Pair      | Mean J_A |
|-----------|---|
| XGBoost, LR | 0.9444 |
| XGBoost, RF | 0.7936 |
| RF, LR      | 0.7540 |
| Aggregate   | 0.8307 |

Stable core set (Group A in all 18 rankings): `{Age, BMI, GenHlth,
HighBP, HighChol}`.

### Temporal stability axis (exploratory; not in the paper)

Group-A Jaccard across cohort pairs (2015 vs 2021, 2015 vs 2023, 2021
vs 2023), for each (model, method) combination. Source:
`outputs/xai_agreement/temporal_stability_cabc.csv`.

| Method | Aggregate J_A | Perfect cells |
|--------|---|---|
| SHAP                  | 0.9312 | 5 / 9 |
| Permutation Importance | 1.0000 | 9 / 9 |
| Aggregate (both methods) | 0.9656 | 14 / 18 |

### Fairness axis (exploratory; not in the paper)

Equalised Odds violation by protected attribute, per (model, cohort).
Source: `outputs/<cohort>/analysis/<model>_fairness_equalized_odds_summary.csv`.
Severity labels follow the convention in `fairness_analysis.py`:
`good` for EO < 0.05, `acceptable` for 0.05 to 0.10,
`concerning` for 0.10 to 0.20, `severe` for >= 0.20.

Sex shows minimal bias across all (model, cohort) cells (EO <= 0.07).
The dominant patterns are concentrated in Age and Income, with
Logistic Regression showing the most severe bias.

EO violation by Income (max of dTPR, dFPR):

| Model | 2015 | 2021 | 2023 |
|-------|------|------|------|
| XGBoost              | 0.2422 | 0.3153 | 0.3047 |
| Random Forest        | 0.1090 | 0.1923 | 0.2090 |
| Logistic Regression  | 0.4316 | 0.4914 | 0.4099 |

EO violation by Age (max of dTPR, dFPR):

| Model | 2015 | 2021 | 2023 |
|-------|------|------|------|
| XGBoost              | 0.2365 | 0.2074 | 0.2122 |
| Random Forest        | 0.2502 | 0.2098 | 0.2291 |
| Logistic Regression  | 0.7902 | 0.7412 | 0.7278 |

If a clean run reproduces these values, the pipeline is reproducible.
If any value differs, check that `random_state = 42` is honoured
throughout and that all packages match `requirements.txt`.

## Post-MAPR metric comparison (journal extension, work in progress)

The MAPR 2026 outputs under `outputs/xai_agreement/` are frozen at git tag
`mapr2026-v1.0`. Two scripts compare cABC Group-A Jaccard with the baselines
requested by the MAPR reviewers (top-K overlap, Spearman, Kendall, weighted
Kendall, extrapolated RBO) on the same 18 attribution vectors, without
retraining:

```bash
python -m src.analysis.run_metric_comparison   # 45 pairs -> outputs/metric_comparison/
python -m src.analysis.run_cabc_bootstrap      # patient-level bootstrap (XGB, LR)
python -m tests.test_agreement_metrics         # checks, incl. reproduction of MAPR cABC CSVs
```

Two corrections relative to the MAPR wording: (i) the cABC A|B boundary used
here selects exactly the features whose importance exceeds the mean
importance, so it is scale-free and data-adaptive but not threshold-free;
(ii) `rbo_score` in `run_xai_agreement.py` is a truncated sum bounded above by
1 - p^k (0.833 for 17 features at p = 0.9); new analyses use the extrapolated
RBO in `src/evaluation/agreement_metrics.py`.

## Reproducibility

All splits, samples, and stochastic estimators use `random_state = 42`.
Hyperparameters are fixed in `configs/default.yaml`; no test-set
tuning is performed. TreeSHAP attributions use a seeded sample of 200
test instances for tree models, while LinearSHAP is closed-form and
uses the full test set for Logistic Regression. Permutation Importance
uses `n_repeats = 10`, `scoring = "roc_auc"`, and `n_jobs = 1` on the
full test set. The single-process setting is intentional: with
`n_jobs = -1`, some Windows and PyCharm configurations fail silently
and produce empty importance files. Exact package versions for the
reference run are recorded in
`outputs/<cohort>/reproducibility/environment.json`. A separate
`README_REPRODUCIBILITY.md` provides the step-by-step recipe with
expected runtimes per stage.

## Key references

- Ultsch, A. and Lötsch, J. (2015). Computed ABC analysis for rational
  selection of most informative variables in multivariate data.
  *PLoS ONE*, 10(6), e0129767.
- Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to
  interpreting model predictions. *NeurIPS 30*.
- Lundberg, S. M. *et al.* (2020). From local explanations to global
  understanding with explainable AI for trees. *Nat. Mach. Intell.*,
  2(1), 56-67.
- Breiman, L. (2001). Random forests. *Mach. Learn.*, 45(1), 5-32.
- Fisher, A., Rudin, C., and Dominici, F. (2019). All models are wrong
  but many are useful. *JMLR*, 20(177).
- Hardt, M., Price, E., and Srebro, N. (2016). Equality of opportunity
  in supervised learning. *NeurIPS 29*.

The full reference list is in the accompanying paper.

## Funding

This research was funded by University of Information Technology, Vietnam
National University Ho Chi Minh City under grant number CS4-2026-80120.

## License

Code in this repository is released under the MIT License; see
[LICENSE](LICENSE). The BRFSS data files under `data/` are derived from
public-domain CDC survey releases; see [data/README.md](data/README.md)
for provenance.
