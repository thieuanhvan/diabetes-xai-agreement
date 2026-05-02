README_REPRODUCIBILITY.md
# Reproducibility Guide

This document describes how to fully reproduce all results reported in the manuscript:

**"Explainable AI in Diabetes Risk Prediction: An Agreement, Stability, and Fairness Analysis on Population-Scale Data"**

---

## 📌 Overview

The pipeline consists of three main stages:

1. Dataset construction from raw BRFSS files
2. Model training and feature importance extraction
3. XAI agreement, stability, and fairness analysis

---

## ⚙️ Environment Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
📊 Step 1 — Build Datasets

Convert raw BRFSS .XPT files into Kaggle-style datasets.

python src/data_preparation/run_build_brfss_kaggle_style_datasets.py
Expected Output
data/processed/
    cdc_brfss_diabetes_2015.csv
    cdc_brfss_diabetes_2021.csv
    cdc_brfss_diabetes_2023.csv
🤖 Step 2 — Train Models

Run all combinations of datasets and models:

python src/pipelines/run_all_combos.py
Models
Logistic Regression
Random Forest
XGBoost
Datasets
BRFSS 2015
BRFSS 2021
BRFSS 2023
Outputs
outputs/
    cdc_brfss_diabetes_2015/
    cdc_brfss_diabetes_2021/
    cdc_brfss_diabetes_2023/

Each dataset folder contains:

model performance metrics
SHAP feature importance
permutation feature importance
🧠 Step 3 — XAI Agreement Analysis

Run agreement, stability, and fairness analysis:

python src/analysis/run_xai_agreement.py
Outputs
outputs/xai_agreement/

Includes:

within_model_agreement.csv
cross_model_shap_agreement.csv
temporal_stability.csv
cabc_groups.csv
Plots
cABC grouping per run
summary breakpoint plots
cross-model comparison
fairness drift analysis
📁 Logs

All scripts automatically generate execution logs:

logs/run_<script>_YYYYMMDD_HHMMSS.log

Logs include:

dataset size
model configuration
execution steps
output file locations
⏱ Runtime

Approximate runtime on a standard machine:

Step	Time
Dataset build	~5–10 minutes
Model training (all combos)	~2–3 hours
XAI agreement analysis	~5–15 minutes

Note: Random Forest SHAP computation is the most time-consuming step.

⚠️ Notes
Raw BRFSS data is not included in this repository
SHAP values are computed using a sample (default: 200 instances)
Results are deterministic with fixed random seeds
🔁 Reproducibility Guarantee

The results in the manuscript are reproducible given:

identical dataset construction
fixed random seed
consistent environment

Minor numerical variations may occur due to:

SHAP sampling
permutation importance randomness

However, all key findings and patterns remain stable.

📬 Contact

For questions or reproducibility issues, please open a GitHub issue.