# data/

Three cleaned CDC BRFSS cohorts committed directly to this folder:

```
data/
├── cdc_brfss_diabetes_2015.csv    (253,680 rows x 22 cols)
├── cdc_brfss_diabetes_2021.csv    (236,378 rows x 22 cols)
└── cdc_brfss_diabetes_2023.csv    (272,769 rows x 18 cols)
```

These three files are the only inputs the pipeline consumes. The full
reproduction in `README_REPRODUCIBILITY.md` reads them as-is; no raw
`.XPT` ingestion is required.

## Dataset summary

|                 | BRFSS 2015 | BRFSS 2021 | BRFSS 2023 |
|-----------------|---|---|---|
| Records         | 253,680 | 236,378 | 272,769 |
| Features        | 21 (+ 1 target) | 21 (+ 1 target) | 17 (+ 1 target) |
| Target          | `Diabetes_binary` | `Diabetes_binary` | `Diabetes_binary` |
| Period          | Pre-pandemic baseline | In-pandemic peak | Post-pandemic |
| Source          | CDC BRFSS 2015 | CDC BRFSS 2021 | CDC BRFSS 2023 |

**On the 2023 schema reduction.** Four variables present in the 2015 and
2021 cohorts (`Fruits`, `Veggies`, `AnyHealthcare`, `HvyAlcoholConsump`)
are not carried into the 2023 CSV used here, which therefore has 17
features instead of 21. This is a property of the recoded cohort adopted
for this study, not a claim that the underlying CDC questionnaire items
were discontinued. The pipeline harmonises all three cohorts on the
17-feature common schema, so the analysis is unaffected either way.

All features are survey self-report. No clinical biomarkers (HbA1c,
fasting glucose, OGTT) are included.

## Provenance

- **Raw source:** CDC Behavioral Risk Factor Surveillance System (BRFSS),
  <https://www.cdc.gov/brfss/>
- **Recoding protocol:** Teboul (2022) Kaggle convention, faithfully
  reproduced and extended to BRFSS 2021 and BRFSS 2023 for this study
- **Reference Kaggle datasets** used for output validation during the
  cleaning step:
  - 2015: `alexteboul/diabetes-health-indicators-dataset`
  - 2021: `julnazz/diabetes-health-indicators-dataset`
  - 2023: `siamaktahmasbi/diabetes-2023-brfss-cdc`

## Switching the active cohort

Edit `src/datasets/dataset_registry.py` and set `ACTIVE_DATASET`:

```python
ACTIVE_DATASET = "cdc_brfss_diabetes_2015"   # pre-pandemic
ACTIVE_DATASET = "cdc_brfss_diabetes_2021"   # in-pandemic
ACTIVE_DATASET = "cdc_brfss_diabetes_2023"   # post-pandemic
```

Outputs are written to `outputs/<slug>/`. For the full 9-cell factorial
across all three cohorts and all three models, run:

```bash
python -m src.pipelines.run_pipeline_all_combos
```

## NHANES cohorts (label axis, journal extension)

```
data/
├── cdc_nhanes_diabetes_2017-2020.csv    (8,702 rows x 29 cols)
└── cdc_nhanes_diabetes_2021-2023.csv    (6,185 rows x 29 cols)
```

Built from the public CDC NHANES files (2017–March 2020 pre-pandemic
release `P_*`; August 2021–August 2023 release `*_L`) by the companion
repository `nhanes-diabetes` (branch `label-axis-columns`), adults aged 18+.
Used only by `src/analysis/run_label_axis.py`; the MAPR pipeline does not
read them.

Columns used by the label axis:

| Role | Columns |
|---|---|
| Labels | `Diabetes_self` (doctor-diagnosed), `Diabetes_lab_a1c` (HbA1c ≥ 6.5 %); "total" = either |
| Exclusion | `Diabetes_borderline` = 1 (DIQ010 = 3) |
| Features | HighBP, HighChol, BMI (measured), Smoker, Stroke, HeartDiseaseorAttack, PhysActivity_LTPA, AnyHealthcare, GenHlth, MentHlth, Sex, Age, Education, Income |
| Survey design | `SEQN`, `WTMEC`, `SDMVPSU`, `SDMVSTRA` |
| Not used as features | HbA1c, FastingGlucose, Diabetes_lab, Diabetes_treated, blood pressure |

Construct differences from BRFSS: BMI is measured, MentHlth is a PHQ-9
severity band expressed in days, AnyHealthcare (insurance) stands in for
NoDocbcCost, Income is the poverty–income ratio in 5 bands, and
PhysActivity_LTPA is not comparable across the two NHANES cycles
(48.5 % vs 75.4 % active; instrument change). CholCheck, PhysHlth and
DiffWalk have no counterpart in both cycles.
