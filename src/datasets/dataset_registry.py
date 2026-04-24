from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

DATASETS: dict[str, dict] = {

    "kaggle_100000k": {
        "path": DATA_DIR / "kaggle_100000x32_diabetes_health_indicators_classification.csv",
        "target": "diagnosed_diabetes",
        "slug": "kaggle_100000k",
        "task": "classification",
        "include_columns": [
            "age", "gender", "ethnicity", "education_level", "income_level",
            "employment_status", "smoking_status", "alcohol_consumption_per_week",
            "physical_activity_minutes_per_week", "diet_score",
            "sleep_hours_per_day", "screen_time_hours_per_day",
            "family_history_diabetes", "hypertension_history",
            "cardiovascular_history"
            #, "bmi"
            #, "waist_to_hip_ratio",
            "systolic_bp", "diastolic_bp", "heart_rate"
            ,"cholesterol_total", "hdl_cholesterol", "ldl_cholesterol"
            ,"triglycerides"
            , "glucose_fasting"
        ],
    },

    "kaggle_640": {
        "path": DATA_DIR / "kaggle_640x16_early_stage_diabetes_risk_prediction.csv",
        "target": "class",
        "slug": "kaggle_640",
        "task": "classification",
        "include_columns": None,
    },

    "cdc_brfss_2015": {
        "path": DATA_DIR / "cdc_brfss_2015_70692x22_diabetes_binary_5050split_health_indicators_BRFSS2015.csv.csv",
        "target": "Diabetes_binary",
        "slug": "cdc_brfss_2015",
        "task": "classification",
        "include_columns": None,
    },
    "cdc_brfss_2015_full": {
        "path": DATA_DIR / "cdc_brfss_2015_253680x22_diabetes_binary_health_indicators_BRFSS2015.csv",
        "target": "Diabetes_binary",
        "slug": "cdc_brfss_2015_full",
        "task": "classification",
        "include_columns": None,
    },

    "cdc_brfss_2021": {
        "path": DATA_DIR / "cdc_brfss_2021_67136x22_diabetes_health_indicators_classification.csv",
        "target": "Diabetes_binary",
        "slug": "cdc_brfss_2021",
        "task": "classification",
        "include_columns": None,
    },

    "cdc_brfss_2021_full": {
        "path": DATA_DIR / "cdc_brfss_2021_236379x22_diabetes_binary_health_indicators_BRFSS2021.csv",
        "target": "Diabetes_binary",
        "slug": "cdc_brfss_2021_full",
        "task": "classification",
        "include_columns": None,
    },

    "sklearn_diabetes": {
        "path": DATA_DIR / "sklearn_442x10_diabetes_progression_regression.csv",
        "target": "disease_progression",
        "slug": "sklearn_diabetes",
        "task": "regression_to_binary",
        "include_columns": None,
    },
    "cdc_brfss_2015_rebuilt": {
        "path": DATA_DIR / "cdc_brfss_2015_rebuilt.csv",
        "target": "Diabetes_binary",
        "slug": "cdc_brfss_2015_rebuilt",
        "task": "classification",
        "include_columns": None,
    },

    "cdc_brfss_2021_rebuilt": {
        "path": DATA_DIR / "cdc_brfss_2021_rebuilt.csv",
        "target": "Diabetes_binary",
        "slug": "cdc_brfss_2021_rebuilt",
        "task": "classification",
        "include_columns": None,
    },

    "cdc_brfss_2023_rebuilt": {
        "path": DATA_DIR / "cdc_brfss_2023_rebuilt.csv",
        "target": "Diabetes_binary",
        "slug": "cdc_brfss_2023_rebuilt",
        "task": "classification",
        "include_columns": None,
    },

}
#ACTIVE_DATASET = "sklearn_diabetes"
#ACTIVE_DATASET = "kaggle_100000k"
#ACTIVE_DATASET = "kaggle_640"

#ACTIVE_DATASET = "cdc_brfss_2015"
#ACTIVE_DATASET = "cdc_brfss_2015_full"

#ACTIVE_DATASET = "cdc_brfss_2021"
#ACTIVE_DATASET = "cdc_brfss_2021_full"

ACTIVE_DATASET = "cdc_brfss_2015_rebuilt"
#ACTIVE_DATASET = "cdc_brfss_2021_rebuilt"
#ACTIVE_DATASET = "cdc_brfss_2023_rebuilt"



# Model dùng cho XAI (step05, agreement analysis, tables)
# Đổi tại đây nếu muốn chạy với model khác
#
# Ghi chú:
#   SHAP     = phân tích giải thích (step05)           — chỉ 3 models đầu hỗ trợ
#   FI       = Permutation Importance (step05)         — model-agnostic, mọi model đều chạy được
#   Fairness = Equalized Odds (step07)                 — tất cả 6 models đều chạy được
#   Paper    = dùng trong paper (RQ2/RQ3)              — chỉ 3 models đầu

ACTIVE_MODEL = "xgboost"              # SHAP (TreeSHAP)   + Permutation FI | Fairness | Paper model chính
#ACTIVE_MODEL = "random_forest"        # SHAP (TreeSHAP)   + Permutation FI | Fairness | Paper ablation
#ACTIVE_MODEL = "logistic_regression"   # SHAP (LinearSHAP) + Permutation FI | Fairness | Paper consensus
#ACTIVE_MODEL = "decision_tree"        # SHAP (TreeSHAP)   + Permutation FI | Fairness | Paper ✗ variance cao

#ACTIVE_MODEL = "knn"                  # SHAP ✗ (quá chậm, skip) | Permutation FI | Fairness | Paper ✗
#ACTIVE_MODEL = "ann"                  # SHAP ✗ (quá chậm, skip) | Permutation FI | Fairness | Paper ✗

# Không hỗ trợ SHAP: KNeighborsClassifier, MLPClassifier (KernelSHAP > 4 giờ)
# FI Permutation: mọi model đều chạy được (model-agnostic, Breiman 2001)

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