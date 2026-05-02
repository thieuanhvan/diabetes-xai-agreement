from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Since we have brfss-diabetes which do task cleaning data
# we do not separate raw and processed folders anymore
#DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR = PROJECT_ROOT / "data"

DATASETS: dict[str, dict] = {


    # ============================================================
    # REBUILT — 17 predictors thống nhất cho cả 3 năm
    # 2015 & 2021 raw có 22 cols → liệt kê 17 predictors cần giữ
    # 2023 raw đã 18 cols (17 predictors + target) → None đọc hết
    # 4 features bị CDC drop khỏi BRFSS 2023 questionnaire:
    #   Fruits, Veggies, AnyHealthcare, HvyAlcoholConsump
    # → reduce 2015 & 2021 xuống 17 để cross-temporal apples-to-apples
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
        "include_columns": None,  # 2023 đã 17 predictors sẵn
    },

}


ACTIVE_DATASET = "cdc_brfss_diabetes_2015"
#ACTIVE_DATASET = "cdc_brfss_diabetes_2021"
#ACTIVE_DATASET = "cdc_brfss_diabetes_2023"


# Model dùng cho XAI (step05, agreement analysis, tables)
# Đổi tại đây nếu muốn chạy với model khác
#
# Ghi chú:
#   SHAP     = phân tích giải thích (step05)           — chỉ 3 models đầu hỗ trợ
#   FI       = Permutation Importance (step05)         — model-agnostic, mọi model đều chạy được
#   Fairness = Equalized Odds (step07)                 — tất cả 6 models đều chạy được
#   Paper    = dùng trong paper (RQ2/RQ3)              — chỉ 3 models đầu

ACTIVE_MODEL = "xgboost"              # SHAP (TreeSHAP)   + Permutation FI | Fairness | Paper model chính
#ACTIVE_MODEL = "logistic_regression"   # SHAP (LinearSHAP) + Permutation FI | Fairness | Paper consensus
#ACTIVE_MODEL = "random_forest"        # SHAP (TreeSHAP)   + Permutation FI | Fairness | Paper ablation

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
