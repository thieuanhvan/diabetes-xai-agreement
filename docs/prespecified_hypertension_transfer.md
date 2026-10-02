# Pre-specified expectations: hypertension transfer demonstration

Written 2026-10-02 08:39 (UTC+7), before any hypertension model was fitted.

Design: NHANES 2017-March 2020 and 2021-2023 adults; labels diag (BPQ020, told high blood pressure), lab (mean measured SBP >= 140 or DBP >= 90 mmHg) and total (either); 14 predictors with diagnosed diabetes in place of HighBP; LR, RF, XGBoost; SHAP and PI; seeds 0-4; class weighting as in the diabetes analysis ("pipeline") and "none".

Expectations, stated in advance:
1. Depth versus contradiction: most label or model pairs with J_A < 1 are nested (OC_A = 1), so J_A again tracks depth.
2. Outcome definition: under the diag label, diagnosed comorbidities (Diabetes_self, HeartDiseaseorAttack, HighChol) gain attribution share relative to the lab label, as for diabetes.
3. Training configuration: the cross-model agreement of SHAP changes when RF class weighting is removed; the direction is not predicted.
A result contrary to expectation 2 will be reported as such.
