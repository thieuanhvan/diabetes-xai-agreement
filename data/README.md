# data/

Dataset files are **not tracked in git** (large files, >50MB each).
Place the 3 required CSV files directly in this folder (`data/`) with the
exact filenames listed below.

These CSVs are produced by the companion repository
[`thieuanhvan/brfss-diabetes`](https://github.com/thieuanhvan/brfss-diabetes),
which handles raw `.XPT` ingestion, Teboul-protocol recoding, and listwise
deletion. This repository (`diabetes-xai-agreement`) does **not** build data
from raw — it consumes the cleaned CSVs only.

---

## Required Files

Place all 3 files directly in this `data/` folder:

```
data/
├── cdc_brfss_2015_rebuilt.csv  (253,680 rows × 22 cols)
├── cdc_brfss_2021_rebuilt.csv  (236,378 rows × 22 cols)
└── cdc_brfss_2023_rebuilt.csv  (272,769 rows × 18 cols)
```

---

## How to obtain the files

### Option A — Build from raw (recommended for reproducibility)

Use the `brfss-diabetes` toolkit:

```bash
# 1. Clone the toolkit repo
git clone https://github.com/thieuanhvan/brfss-diabetes.git
cd brfss-diabetes

# 2. Place CDC raw .XPT files in data/raw/cdc/
#    Download from: https://www.cdc.gov/brfss/annual_data/annual_data.htm
#      - LLCP2015.XPT
#      - LLCP2021.XPT
#      - LLCP2023.XPT

# 3. Build the cleaned CSVs
python src/brfss_diabetes/run_build_kaggle_csvs.py

# 4. Outputs are written to outputs/tabular/
#    Copy the 3 files into this repo's data/ folder
cp outputs/tabular/cdc_brfss_2015_rebuilt.csv  /path/to/diabetes-xai-agreement/data/
cp outputs/tabular/cdc_brfss_2021_rebuilt.csv  /path/to/diabetes-xai-agreement/data/
cp outputs/tabular/cdc_brfss_2023_rebuilt.csv  /path/to/diabetes-xai-agreement/data/
```

### Option B — Download pre-built CSVs from Kaggle

The same 3 cleaned CSVs are published as a Kaggle dataset:

**Kaggle dataset:** `https://www.kaggle.com/datasets/thieuanhvan/brfss-diabetes`
*(URL placeholder — to be updated after Kaggle release, planned 5/2026)*

**Kaggle DOI:** `10.34740/kaggle/dsv/XXXXXXXX`
*(DOI placeholder — assigned by Kaggle on publication)*

Download the dataset, unzip, and place the 3 CSVs in `data/`.

---

## Dataset Info

| | BRFSS 2015 | BRFSS 2021 | BRFSS 2023 |
|---|---|---|---|
| Records | 253,680 | 236,378 | 272,769 |
| Features | 21 (+ 1 target) | 21 (+ 1 target) | 17 (+ 1 target) |
| Target | `Diabetes_binary` | `Diabetes_binary` | `Diabetes_binary` |
| Period | Pre-COVID baseline | In-COVID peak | Post-COVID |
| Schema | Full (Teboul 2015) | Full (julnazz 2021) | Reduced (CDC dropped 4 vars) |
| Source | CDC BRFSS 2015 | CDC BRFSS 2021 | CDC BRFSS 2023 |
| Registry slug | `cdc_brfss_2015_rebuilt` | `cdc_brfss_2021_rebuilt` | `cdc_brfss_2023_rebuilt` |

**Note on 2023 schema reduction:** CDC removed 4 lifestyle variables in BRFSS 2023
(`Fruits`, `Veggies`, `AnyHealthcare`, `HvyAlcoholConsump`) due to questionnaire
modifications. The 2023 CSV therefore has 17 features instead of 21. For
cross-temporal analysis, the 2015 and 2021 datasets can be projected to the
common 17-feature schema as a separate post-processing step.

All features are survey self-report (no clinical biomarkers such as HbA1c or
fasting glucose).

---

## Switching the active dataset

Edit `src/datasets/dataset_registry.py`, change the `ACTIVE_DATASET` line:

```python
ACTIVE_DATASET = "cdc_brfss_2015_rebuilt"   # BRFSS 2015 — pre-COVID baseline
ACTIVE_DATASET = "cdc_brfss_2021_rebuilt"   # BRFSS 2021 — in-COVID peak
ACTIVE_DATASET = "cdc_brfss_2023_rebuilt"   # BRFSS 2023 — post-COVID (default)
```

Then re-run the pipeline. Outputs are saved under the corresponding slug folder
(e.g., `outputs/cdc_brfss_2023_rebuilt/`).

For batch runs across all 3 years, use:

```bash
python src/pipelines/run_pipeline_all_combos.py
```

---

## Source and provenance

- **Raw data:** CDC Behavioral Risk Factor Surveillance System (BRFSS),
  https://www.cdc.gov/brfss/
- **Recoding protocol:** Teboul (2022) Kaggle convention, faithfully
  reproduced and extended to 2023 by the `brfss-diabetes` toolkit
- **Reference Kaggle datasets** (used for output validation):
  - 2015: `alexteboul/diabetes-health-indicators-dataset`
  - 2021: `julnazz/diabetes-health-indicators-dataset`
  - 2023: `siamaktahmasbi/diabetes-2023-brfss-cdc`

For the full recoding methodology (PATTERN A vs PATTERN B encoding, multi-level
recodings, listwise deletion rationale), see
[`brfss-diabetes/docs/methodology.md`](https://github.com/thieuanhvan/brfss-diabetes/blob/main/docs/methodology.md).
