# CDC BRFSS 2021 – Rebuild 50/50 Dataset

This experiment rebuilds the balanced **50/50 diabetes dataset** from the original CDC BRFSS dataset (~236k records).

The goal is to demonstrate the dataset construction process used in the Kaggle dataset:

cdc_diabetes_binary_5050split_health_indicators_BRFSS2021.csv

This script reproduces the process from the **raw dataset** so that the experiment can be verified and reproduced.

---

## Source Dataset

Input file (raw):

data/raw/cdc_diabetes_012_health_indicators_BRFSS2021.csv

Size:

~236,000 records

Target column:

Diabetes_012

Label meaning:

| Value | Meaning |
|-----|------|
| 0 | No diabetes |
| 1 | Prediabetes |
| 2 | Diabetes |

---

## Dataset Construction Process

The dataset is rebuilt using the following steps:

### Step 1 — Remove Prediabetes

Records with
