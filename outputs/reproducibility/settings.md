# Reproducibility settings

Python 3.11.15; numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, xgboost 3.2.0, shap 0.51.0

## MAPR baseline metric comparison (8a)

- **script**: src.analysis.run_metric_comparison
- **data**: BRFSS 2015/2021/2023, 17 common predictors; 18 stored attribution vectors (tag mapr2026-v1.0)
- **sample**: 253,680 / 236,378 / 272,769 rows; 80/20 split, seed 42 (MAPR)
- **seeds**: 1 (MAPR vectors, no retraining)
- **models**: LR: max_iter=1000; RF: n_estimators=200, unconstrained depth; XGB: n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8; StandardScaler before every model; random_state = seed
- **attribution**: SHAP (TreeSHAP 200 rows; LinearSHAP LR), PI as in MAPR
- **metrics**: cABC J_A/J_B/J_C, overlap coefficient, size ratio, contradiction, top-K (5,6,7,10), Spearman, Kendall tau-b, weighted tau, RBO_ext (p=0.9/0.8/0.5)
- **statistics**: 45 pairs; descriptive
- **runtime**: seconds

## Patient-level bootstrap of Group A

- **script**: src.analysis.run_cabc_bootstrap
- **data**: BRFSS per-patient SHAP (XGB, LR), 2,000 test patients per cohort
- **sample**: 2,000 patients x 3 cohorts
- **seeds**: bootstrap B=2000, seed 42
- **models**: fixed (MAPR models)
- **attribution**: stored per-patient SHAP
- **metrics**: P(|A|=k), XGB-LR J_A distribution, Spearman 95% interval
- **statistics**: percentile bootstrap
- **runtime**: about 10 s

## BRFSS multi-seed grid (MAPR factorial re-run)

- **script**: src.analysis.run_brfss_multiseed
- **data**: BRFSS 2015/2021/2023, 17 common predictors
- **sample**: 80/20 stratified split per seed
- **seeds**: 5 (0-4); class-weight ablation 3 (0-2)
- **models**: LR: max_iter=1000; RF: n_estimators=200, unconstrained depth; XGB: n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8; StandardScaler before every model; random_state = seed | pipeline (MAPR): LR, RF class_weight=balanced, XGB unweighted; ablation: none / balanced
- **attribution**: SHAP on 2,000 test rows (XGB, LR) and 200 rows (RF, exact TreeSHAP ~5 s/row); sklearn permutation_importance, scoring=roc_auc, n_repeats=10, random_state=seed, signed mean kept; PI on a stratified 10,000-row test subsample
- **metrics**: as above, plus seed axis (noise floor)
- **statistics**: mean and 2.5/97.5 percentiles over pairs
- **runtime**: logistic_regression: 2 s/run; random_forest: 1505 s/run; xgboost: 13 s/run

## NHANES label axis (diagnosis vs HbA1c vs total)

- **script**: src.analysis.run_label_axis
- **data**: NHANES 2017-Mar 2020 (P_*) and 2021-2023 (*_L), adults >= 18, borderline excluded, valid HbA1c, 14 BRFSS-counterpart features (nhanes-diabetes, branch label-axis-columns)
- **sample**: 2017-2020: train 4976, test 1245; 2021-2023: train 3347, test 837; same split for all labels, stratified on (diagnosis, HbA1c) cell
- **seeds**: 10 (0-9)
- **models**: LR: max_iter=1000; RF: n_estimators=200, unconstrained depth; XGB: n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8; StandardScaler before every model; random_state = seed | pipeline (MAPR): LR, RF class_weight=balanced, XGB unweighted
- **attribution**: SHAP on all test rows; sklearn permutation_importance, scoring=roc_auc, n_repeats=10, random_state=seed, signed mean kept; unweighted and MEC-weighted (WTMEC) versions
- **metrics**: as above; feature-level change in attribution share
- **statistics**: Nadeau-Bengio corrected resampled t-test, Benjamini-Hochberg within cycle x weighting x method x label pair (42 tests)
- **runtime**: logistic_regression: 1 s/run; random_forest: 77 s/run; xgboost: 3 s/run

## NHANES label axis, survey-weighted training (sensitivity)

- **script**: src.analysis.run_label_axis --train-weighted
- **data**: as label axis
- **sample**: as label axis
- **seeds**: 10 (0-9)
- **models**: as label axis; WTMEC passed as sample_weight at fit time (combined with class weighting)
- **attribution**: as label axis
- **metrics**: as label axis
- **statistics**: as label axis; compared with the unweighted run feature by feature
- **runtime**: logistic_regression: 1 s/run; random_forest: 93 s/run; xgboost: 6 s/run

## Fairness by label (NHANES)

- **script**: src.analysis.run_label_fairness
- **data**: as label axis
- **sample**: as label axis
- **seeds**: 10 (0-9)
- **models**: LR: max_iter=1000; RF: n_estimators=200, unconstrained depth; XGB: n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8; StandardScaler before every model; random_state = seed
- **attribution**: none (predictions only)
- **metrics**: TPR, FPR, AUC per group; TPR gap (weighted by WTMEC); groups: Income PIR<2 vs >=2, Education HS or less vs more, Sex, Age 18-44/45-64/65+
- **statistics**: capacity-matched threshold (flag rate = training-label prevalence); percentiles over seeds
- **runtime**: about 1.5 min total

## Variance decomposition (model x method x label/year)

- **script**: src.analysis.run_variance_decomposition
- **data**: vectors from the label axis and the BRFSS multi-seed grid
- **sample**: -
- **seeds**: replicates = seeds
- **models**: -
- **attribution**: -
- **metrics**: pooled eta^2 per source; response = attribution share and within-vector rank
- **statistics**: balanced full-factorial ANOVA per feature, SS pooled over features
- **runtime**: seconds

## Instance vs population agreement; univariate core check

- **script**: src.analysis.run_instance_vs_population
- **data**: BRFSS per-patient SHAP (XGB, LR); NHANES seed-0 split
- **sample**: 2,000 BRFSS patients per cohort; NHANES full test split
- **seeds**: 1
- **models**: XGB, LR
- **attribution**: per-patient |SHAP|
- **metrics**: per-patient top-5 overlap and Spearman vs population values; univariate |AUC-0.5| ranking
- **statistics**: descriptive (quartiles, share below population value)
- **runtime**: about 30 s

## Correction of the MAPR Income-fairness result

- **script**: src.analysis.run_brfss_income_harmonised
- **data**: MAPR per-bin fairness outputs (n, n_positive, TPR)
- **sample**: MAPR test split (seed 42)
- **seeds**: 1
- **models**: MAPR models
- **attribution**: -
- **metrics**: delta-TPR, native bins (8/11/7) vs 5 common bins
- **statistics**: pooled TP / pooled positives
- **runtime**: seconds

## Hypertension transfer demonstration (class weighting: pipeline)

- **script**: src.analysis.run_label_axis --outcome hypertension --class-weighting pipeline --seeds 5
- **data**: NHANES as label axis, without the HbA1c restriction; labels diag (BPQ020), measured (mean oscillometric SBP >= 140 or DBP >= 90 mmHg), composite; diagnosed diabetes replaces HighBP among the 14 features; blood pressure never a feature
- **sample**: 6,090 and 4,314 adults; same split for all labels
- **seeds**: 5 (0-4)
- **models**: LR: max_iter=1000; RF: n_estimators=200, unconstrained depth; XGB: n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8; StandardScaler before every model; random_state = seed
- **attribution**: as label axis
- **metrics**: as label axis
- **statistics**: corrected resampled t-test, BH; cross-cycle replication
- **runtime**: logistic_regression: 1 s/run; random_forest: 108 s/run; xgboost: 3 s/run

## Hypertension transfer demonstration (class weighting: none)

- **script**: src.analysis.run_label_axis --outcome hypertension --class-weighting none --seeds 5
- **data**: NHANES as label axis, without the HbA1c restriction; labels diag (BPQ020), measured (mean oscillometric SBP >= 140 or DBP >= 90 mmHg), composite; diagnosed diabetes replaces HighBP among the 14 features; blood pressure never a feature
- **sample**: 6,090 and 4,314 adults; same split for all labels
- **seeds**: 5 (0-4)
- **models**: LR: max_iter=1000; RF: n_estimators=200, unconstrained depth; XGB: n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8; StandardScaler before every model; random_state = seed
- **attribution**: as label axis
- **metrics**: as label axis
- **statistics**: corrected resampled t-test, BH; cross-cycle replication
- **runtime**: logistic_regression: 1 s/run; random_forest: 98 s/run; xgboost: 4 s/run

## Sensitivity analyses

- **script**: src.analysis.run_sensitivity_checks --n-perm 999
- **data**: vectors and pairs of all grids; stored per-patient SHAP (BRFSS)
- **sample**: -
- **seeds**: -
- **models**: -
- **attribution**: -
- **metrics**: crossed-seed and CLR decompositions, per-feature eta^2, permutation p (B = 999, p = (b + 1)/(B + 1)), seed-reference percentiles 1/5/10, exact sign-flip test, signed patient-level SHAP
- **statistics**: as listed
- **runtime**: about 1 min
