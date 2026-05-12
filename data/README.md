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

**On the 2023 schema reduction:** CDC removed four lifestyle variables in
BRFSS 2023 (`Fruits`, `Veggies`, `AnyHealthcare`, `HvyAlcoholConsump`) due
to questionnaire modifications. The 2023 CSV therefore carries 17
features instead of 21. The pipeline harmonises all three cohorts on the
17-feature common schema.

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
