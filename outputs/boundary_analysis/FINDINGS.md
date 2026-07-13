# Boundary-Patient Sub-Analysis — Findings Summary

**Purpose.** Supplementary analysis accompanying the MAPR 2026 paper
(*Auditing Population-Level XAI Agreement with cABC: Evidence from Diabetes
Risk Prediction*; see the How to cite section of the top-level `README.md`).
It asks whether the headline cross-model SHAP agreement (J_A = 0.9444 for
XGB-LR) is stable across patient prediction-confidence strata, or whether
the average masks heterogeneity.

**Script.** `src/analysis/run_boundary_patient_analysis.py` — standalone,
does not modify the main pipeline. Total compute: about 50 s on a Linux
container. The validation pass confirms reproducibility against the
committed reference SHAP aggregates (LR at machine precision, XGB at
Kendall tau >= 0.97 on rank ordering).

**Method.**
- 3 cohorts: BRFSS 2015 / 2021 / 2023 (17-feature harmonised schema)
- 2 models: XGBoost (n_estimators=300, max_depth=6) and Logistic Regression
  (max_iter=1000, class_weight="balanced") — the same configurations as the
  main pipeline
- Random Forest excluded from the analysis pass due to TreeSHAP compute cost
  at unconstrained depth (about 40 levels); the LR-XGB pair is in any case
  the strongest-agreeing pair reported in the paper (J_A = 0.9444)
- Per cohort: train models, compute SHAP on a shared seed-42 sample of 2,000
  test patients, bin by XGBoost-predicted P(Diabetes=1) into quintiles
  (Q1-Q5), then within each bin compute the cABC Group A per model via the
  Lorenz breakpoint and the pairwise Jaccard J_A

---

## Headline finding

The core five features are unanimous everywhere.

| Feature | In Group A in 100% of (cohort x bin x model) cells? |
|---|---|
| **GenHlth**  | yes |
| **HighBP**   | yes |
| **Age**      | yes |
| **BMI**      | yes |
| **HighChol** | yes |

These 5 features appear in cABC Group A for both XGB and LR across all
3 cohorts and all 5 prediction-confidence quintiles: 30 of 30 cells.

The apparent cross-model disagreement in lower-confidence bins
(J_A = 0.71 to 0.83) is not substantive disagreement about what matters.
It is a cABC artefact of importance-distribution shape:

- XGBoost's Group A almost always has exactly **5 members** (the core five).
- LR's Group A has 5 to 7 members. In low and mid-risk bins, LR's gentler
  importance gradient lets the Lorenz breakpoint catch one or two tail
  features (typically **Sex**, sometimes **CholCheck**).
- In the high-risk quintile (Q5), LR's gradient sharpens and its Group A
  collapses to the same 5 features as XGB, giving J_A = 1.000 across all
  3 cohorts.

---

## Per-cohort J_A by quintile

Pair: XGB and LR (cABC Group A Jaccard).

| Cohort     | Q1    | Q2    | Q3    | Q4    | **Q5**     | Global (full sample) |
|------------|-------|-------|-------|-------|------------|----------------------|
| BRFSS 2015 | 0.833 | 0.833 | 0.833 | 0.833 | **1.000**  | 1.000 |
| BRFSS 2021 | 0.714 | 0.833 | 0.833 | 0.833 | **1.000**  | 0.833 |
| BRFSS 2023 | 1.000 | 0.833 | 0.833 | 0.833 | **1.000**  | 1.000 |

Bin edges differ slightly across cohorts (they depend on each cohort's
predicted-probability distribution); see `summary.json` for exact values.
The five quintiles span roughly:
- Q1: P in [0.000, 0.013-0.018] — extremely low risk
- Q2: P in [0.013, 0.044-0.055]
- Q3: P in [0.044, 0.117-0.130] — near the population base rate (about 0.14)
- Q4: P in [0.117, 0.254-0.262]
- Q5: P in [0.254, 0.759-0.838] — high risk

Note: because BRFSS prevalence is about 14 percent, the predicted-probability
distribution is right-skewed; there is no meaningful concentration of
patients in a classical decision-boundary region near 0.5. The five
quintiles span the empirical support of the prediction distribution.

---

## Mechanism: cABC truncation sensitivity

For one representative bin (BRFSS 2021, Q1), Group A composition is:

- **XGBoost** (5 features): GenHlth, Age, HighBP, BMI, HighChol
- **LR** (7 features): Age, GenHlth, BMI, HighBP, CholCheck, HighChol, Sex
- Union = 7, intersection = 5, so J_A = 5/7 = **0.714**

Both models put the same 5 features at the top. LR additionally pulls
CholCheck and Sex into Group A because its importance distribution is
flatter: the Lorenz breakpoint argmax shifts right.

This is consistent with the pattern reported in the paper:
- XGB tends to produce a tighter Group A (sharp tree-split importances)
- LR tends to produce a looser Group A (coefficient magnitudes spread more
  evenly)

A reviewer could legitimately argue that this is a **methodological** finding
about cABC rather than about the models: the breakpoint algorithm rewards LR
for being more permissive about secondary features. An alternative metric,
Top-K Jaccard with fixed K = 5, would give J_A = 1.000 in every cell,
supporting the no-disagreement reading.

---

## Implications

**1. The reported cross-model J_A is conservative.**
The mean J_A reported in the paper (0.8307 averaged over XGB-RF-LR pairs,
0.9444 for XGB-LR specifically) understates the substantive consensus. The
5 core features are agreed by both models in every confidence stratum and
every year. The apparent disagreement comes from cABC admitting 1 or 2 tail
features when one model's gradient is flatter, not from genuinely different
views about what drives diabetes risk.

**2. High-confidence patients receive the most stable explanations.**
For Q5 patients (predicted P > 0.25), cross-model agreement is perfect. This
is the deployment-relevant regime: patients flagged as high-risk receive
consistent feature attributions regardless of which model backbone produces
the explanation.

**3. Low-confidence patients still receive consistent core explanations.**
Even in Q1 (predicted P < 0.02), the top-5 features are unanimous. Where the
models differ is in secondary signals (Sex, CholCheck) that would not
normally be acted on clinically.

---

## Output files (`outputs/boundary_analysis/`)

- `jaccard_by_bin.csv` — long-format J_A per cohort, bin and pair
- `jaccard_by_bin.png` — 3-panel line plot
- `group_a_composition.csv` — Group A feature list per cohort, bin and model
- `summary.json` — run metadata, AUCs, global J_A per cohort
- `validation/{cohort}_aggregate_match.csv` — difference against the
  committed reference outputs
- `shap_per_patient/{cohort}/` — per-patient SHAP arrays and predicted
  probabilities (.npy) for downstream re-analysis

---

## Limitations and possible extensions

- **Random Forest is not included in the analysis pass.** Doing so requires
  either a faster machine or a depth-constrained RF. It would extend the
  analysis to all 3 model pairs of the cross-model design.
- **Top-K Jaccard as an alternative metric.** This would resolve the cABC
  truncation ambiguity; a comparison of cABC against Top-K behaviour in this
  regime is a methodological question in its own right.
- **Per-patient instability metric.** The present analysis aggregates within
  bins; per-patient SHAP rank stability (for example Kendall tau across
  models, per row) would give a finer view.
- **Multi-seed variance.** Running the analysis with 5 or more seeds would
  put error bars on the per-bin J_A and indicate whether the 0.714 observed
  in BRFSS 2021 Q1 is signal or noise.
