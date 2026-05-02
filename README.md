📄 README.md (FINAL VERSION)
# diabetes-xai-agreement

A research project for evaluating **agreement, temporal stability, and fairness of explainable AI (XAI)** methods in diabetes risk prediction using population-scale BRFSS datasets.

---

## 📌 Overview

This project investigates how different XAI methods behave across:

- Multiple models (Logistic Regression, Random Forest, XGBoost)
- Multiple datasets (BRFSS 2015, 2021, 2023)
- Multiple evaluation dimensions:
  - Agreement (SHAP vs Permutation Importance)
  - Cross-model consistency
  - Temporal stability
  - Fairness

The goal is to provide a **robust evaluation framework for XAI** rather than relying on a single explanation method.

---

## 🎯 Key Contributions

1. A unified framework for evaluating XAI across **agreement, stability, and fairness**
2. Population-scale empirical analysis using **multi-year BRFSS datasets**
3. Application of **contribution-based grouping (cABC)** to reduce ranking instability
4. Integration of **fairness analysis** into XAI evaluation

>  This repository implements an evaluation framework combining agreement, stability, and fairness for XAI in diabetes risk prediction.

---

## 📊 Datasets

We use CDC BRFSS datasets:

- 2015
- 2021
- 2023

Each dataset contains:

- ~400,000 samples
- ~15–16% positive diabetes cases
- ~20 features (Kaggle-style schema)

⚠️ BRFSS 2024 is **not included** due to schema inconsistency (missing key variables such as HighBP, HighChol, CholCheck).

---

## ⚙️ Project Structure


diabetes-xai-agreement/
│
├── src/
│ ├── pipelines/
│ │ ├── run_pipeline.py
│ │ ├── run_all_combos.py
│ │
│ ├── analysis/
│ │ ├── run_xai_agreement.py
│ │
│ ├── data_preparation/
│ │ ├── run_build_brfss_kaggle_style_datasets.py
│
├── outputs/
│ ├── xai_agreement/
│ │ ├── cabc_groups.csv
│ │ ├── within_model_agreement.csv
│ │ ├── cross_model_shap_agreement.csv
│ │ ├── temporal_stability.csv
│ │ ├── plots/
│
├── logs/
│
└── README.md


---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Build datasets (from raw BRFSS)
python src/data_preparation/run_build_brfss_kaggle_style_datasets.py

Output:

data/processed/
    cdc_brfss_diabetes_2015.csv
    cdc_brfss_diabetes_2021.csv
    cdc_brfss_diabetes_2023.csv
3. Run all models (3 datasets × 3 models)
python src/pipelines/run_all_combos.py
4. Run XAI agreement analysis
python src/analysis/run_xai_agreement.py

Output:

outputs/xai_agreement/

Includes:

Agreement metrics
cABC grouping
Temporal stability
Visualization plots
📈 Key Outputs
Tables
within_model_agreement.csv
cross_model_shap_agreement.csv
temporal_stability.csv
cabc_groups.csv
Plots
cABC grouping (per run)
Summary breakpoint plots
Cross-model comparison
Temporal stability
Fairness drift analysis
🧪 Reproducibility

Each run script (run_*.py) automatically generates a log file:

logs/run_xxx_YYYYMMDD_HHMMSS.log

Logs include:

dataset size
model used
execution steps
output files
⚠️ Notes
Raw data (data/) is not included in this repository
Outputs may be large depending on experiment settings
SHAP computation for Random Forest can be slow (~50 minutes per run)
📄 Paper

This repository supports the research manuscript:

"Explainable AI in Diabetes Risk Prediction: An Agreement, Stability, and Fairness Analysis on Population-Scale Data"

Status: In preparation

👤 Author

Thiều Anh Vân
University of Information Technology (UIT), VNU-HCM

📬 Contact

For questions or collaboration, please open an issue or contact via email.

📜 License

This project is for academic and research purposes.

 

 
## ⭐ If you find this useful, please star the repository.

 