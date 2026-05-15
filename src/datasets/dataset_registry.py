from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Since we have brfss-diabetes which do task cleaning data
# we do not separate raw and processed folders anymore
#DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR = PROJECT_ROOT / "data"

DATASETS: dict[str, dict] = {


    # ============================================================
    # REBUILT — 17 predictors harmonized across all 3 years
    # 2015 & 2021 raw have 22 cols → list 17 predictors to keep
    # 2023 raw already has 18 cols (17 predictors + target) → None reads all
    # 4 features dropped from BRFSS 2023 questionnaire by CDC:
    #   Fruits, Veggies, AnyHealthcare, HvyAlcoholConsump
    # → reduce 2015 & 2021 to 17 for apples-to-apples cross-temporal comparison
    # ============================================================
    "cdc_brfss_diabetes_2015": {
        "path": DATA_DIR / "cdc_brfss_diabetes_2015.csv",
        "target": "Diabetes_binary",
        "slug": "cdc_brfss_diabetes_2015",
        "task": "classification",
        "include_columns": [
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
            "Stroke", "HeartDiseaseorAttack", "PhysActivity",
            "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth",
            "DiffWalk", "Sex", "Age", "Education", "Income",
        ],
    },

    "cdc_brfss_diabetes_2021": {
        "path": DATA_DIR / "cdc_brfss_diabetes_2021.csv",
        "target": "Diabetes_binary",
        "slug": "cdc_brfss_diabetes_2021",
        "task": "classification",
        "include_columns": [
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
            "Stroke", "HeartDiseaseorAttack", "PhysActivity",
            "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth",
            "DiffWalk", "Sex", "Age", "Education", "Income",
        ],
    },

    "cdc_brfss_diabetes_2023": {
        "path": DATA_DIR / "cdc_brfss_diabetes_2023.csv",
        "target": "Diabetes_binary",
        "slug": "cdc_brfss_diabetes_2023",
        "task": "classification",
        "include_columns": None,  # 2023 already has 17 predictors
    },

}


ACTIVE_DATASET = "cdc_brfss_diabetes_2015"
#ACTIVE_DATASET = "cdc_brfss_diabetes_2021"
#ACTIVE_DATASET = "cdc_brfss_diabetes_2023"


# Model used for XAI (step05, agreement analysis, tables)
# Change here to run a different model
#
# Notes:
#   SHAP     = explanation analysis (step05)           — only first 3 models supported
#   FI       = Permutation Importance (step05)         — model-agnostic, all models supported
#   Fairness = Equalized Odds (step07)                 — all 6 models supported
#   Paper    = used in paper (RQ2/RQ3)                 — only first 3 models

ACTIVE_MODEL = "xgboost"              # SHAP (TreeSHAP)   + Permutation FI | Fairness | Paper main model
#ACTIVE_MODEL = "logistic_regression"   # SHAP (LinearSHAP) + Permutation FI | Fairness | Paper consensus
#ACTIVE_MODEL = "random_forest"        # SHAP (TreeSHAP)   + Permutation FI | Fairness | Paper ablation

#ACTIVE_MODEL = "decision_tree"        # SHAP (TreeSHAP)   + Permutation FI | Fairness | Paper x high variance
#ACTIVE_MODEL = "knn"                  # SHAP x (too slow, skip) | Permutation FI | Fairness | Paper x
#ACTIVE_MODEL = "ann"                  # SHAP x (too slow, skip) | Permutation FI | Fairness | Paper x

# SHAP not supported for: KNeighborsClassifier, MLPClassifier (KernelSHAP > 4 hours)
# FI Permutation: all models supported (model-agnostic, Breiman 2001)

def _validate_cfg(cfg: dict) -> dict:
    if "path" not in cfg:
        raise KeyError("Dataset config missing 'path'.")
    if "target" not in cfg:
        raise KeyError("Dataset config missing 'target'.")
    if "slug" not in cfg:
        raise KeyError("Dataset config missing 'slug'.")
    if not cfg["path"].exists():
        raise FileNotFoundError(cfg["path"])
    return cfg


def get_active_dataset() -> dict:
    if ACTIVE_DATASET not in DATASETS:
        raise ValueError(f"ACTIVE_DATASET '{ACTIVE_DATASET}' not found in DATASETS.")
    return _validate_cfg(DATASETS[ACTIVE_DATASET])


def get_dataset_config(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' not found in DATASETS.")
    return _validate_cfg(DATASETS[name])


def get_all_dataset_names() -> list[str]:
    return list(DATASETS.keys())
