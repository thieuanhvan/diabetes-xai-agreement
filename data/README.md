# data/

Dataset files are **not tracked in git** (large files, >50MB each).
Download manually and place in `data/processed/` with the exact filenames below.

---

## Required Files

### Primary dataset — BRFSS 2021 (236K records)

**Filename:**
```
cdc_brfss_2021_236379x22_diabetes_binary_health_indicators_BRFSS2021.csv
```

**Download:** https://www.kaggle.com/datasets/julnazz/diabetes-health-indicators-dataset

**Config slug:** `cdc_brfss_2021_full`

---

### Comparison dataset — BRFSS 2015 (253K records)

**Filename:**
```
cdc_brfss_2015_253680x22_diabetes_binary_health_indicators_BRFSS2015.csv
```

**Download:** same Kaggle page above (separate file)

**Config slug:** `cdc_brfss_2015_full`

---

## Switching Datasets

Edit `src/datasets/dataset_registry.py`, change the last line:

```python
# For BRFSS 2021 (default)
ACTIVE_DATASET = "cdc_brfss_2021_full"

# For BRFSS 2015
ACTIVE_DATASET = "cdc_brfss_2015_full"
```

Then re-run the pipeline. Outputs will be saved under the corresponding slug folder
(e.g., `outputs/cdc_brfss_2015_full/`).

---

## Dataset Info

| | BRFSS 2021 | BRFSS 2015 |
|---|---|---|
| Records | 236,378 | 253,680 |
| Features | 21 (+ 1 target) | 21 (+ 1 target) |
| Target | `Diabetes_binary` | `Diabetes_binary` |
| Positive ratio | 14.2% | ~14% |
| Period | In-COVID peak | Pre-COVID baseline |
| Source | CDC BRFSS 2021 | CDC BRFSS 2015 |

All features are survey self-report (no clinical biomarkers such as HbA1c or glucose).

---

## Source

CDC Behavioral Risk Factor Surveillance System (BRFSS):
https://www.cdc.gov/brfss/
