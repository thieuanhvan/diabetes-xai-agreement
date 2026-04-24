"""
XAI Rank Stability Analysis
============================
Mục tiêu: Kiểm tra xem Agreement Gap giữa SHAP và Gain-based FI
có ổn định qua nhiều random seeds không.

Output:
  outputs/<dataset_slug>/analysis/xai_rank_stability/
    ├── xgboost_raw_ranks_all_seeds.csv        # rank từng feature, từng seed, từng method
    ├── xgboost_rank_mean_std.csv              # mean ± std rank theo method
    └── xgboost_rank_correlation_per_seed.csv  # Spearman correlation SHAP vs FI mỗi seed
"""

import logging
import numpy as np
import pandas as np_pd
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr

import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

# ============================================================
# CONFIG — chỉnh tại đây nếu cần
# ============================================================

DATASET_PATH = "data/processed/cdc_brfss_2021_236379x22_diabetes_binary_health_indicators_BRFSS2021.csv"
TARGET_COL   = "Diabetes_binary"
DATASET_SLUG = "cdc_brfss_2021_full"
MODEL_NAME   = "xgboost"

SEEDS        = [42, 0, 123]       # 3 seeds là đủ cho paper Q2
TEST_SIZE    = 0.2
SHAP_SAMPLE  = 200                # số mẫu dùng cho SHAP (giữ nhỏ để nhanh)

XGBOOST_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    # random_state sẽ được set theo từng seed trong vòng lặp
)

# ============================================================
# SETUP LOGGING
# ============================================================

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"xai_rank_stability_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

# ============================================================
# HELPER: build pipeline
# ============================================================

def build_pipeline(X, seed):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols   = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    xgb = XGBClassifier(**XGBOOST_PARAMS, random_state=seed)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", xgb),
    ])

    return pipeline

# ============================================================
# HELPER: extract gain-based FI
# ============================================================

def get_gain_fi(pipeline, feature_names):
    model_core = pipeline.named_steps["model"]
    importances = model_core.feature_importances_  # gain-based (XGBoost default)

    df = pd.DataFrame({
        "feature":    feature_names,
        "gain_fi":    importances,
    })
    df["rank_gain_fi"] = df["gain_fi"].rank(ascending=False, method="min")
    return df

# ============================================================
# HELPER: extract SHAP importance
# ============================================================

def get_shap_fi(pipeline, X_test, feature_names, sample_size=SHAP_SAMPLE):
    preprocessor = pipeline.named_steps["preprocessor"]
    model_core   = pipeline.named_steps["model"]

    X_processed = preprocessor.transform(X_test)
    X_processed = pd.DataFrame(X_processed, columns=[str(c) for c in feature_names])

    if len(X_processed) > sample_size:
        X_sample = X_processed.sample(sample_size, random_state=42)
    else:
        X_sample = X_processed

    explainer   = shap.TreeExplainer(model_core)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        "feature":        feature_names,
        "shap_importance": mean_abs_shap,
    })
    df["rank_shap"] = df["shap_importance"].rank(ascending=False, method="min")
    return df

# ============================================================
# MAIN LOOP
# ============================================================

def run_xai_rank_stability():
    setup_logging()
    logging.info("=" * 60)
    logging.info("XAI RANK STABILITY ANALYSIS STARTED")
    logging.info("=" * 60)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    logging.info(f"Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    logging.info(f"Shape: {df.shape}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = X.columns.tolist()

    # ----------------------------------------------------------
    # Output dir
    # ----------------------------------------------------------
    out_dir = Path("outputs") / DATASET_SLUG / "analysis" / "xai_rank_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Vòng lặp seeds
    # ----------------------------------------------------------
    all_rows    = []   # raw ranks mỗi seed mỗi feature
    corr_rows   = []   # spearman correlation mỗi seed

    for seed in SEEDS:
        logging.info(f"--- SEED {seed} ---")

        # Train/test split (stratify giữ nguyên như pipeline chính)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y,
        )

        # Build & train
        pipeline = build_pipeline(X_train, seed)
        pipeline.fit(X_train, y_train)
        logging.info(f"  Model trained (seed={seed})")

        # Gain-based FI
        fi_df = get_gain_fi(pipeline, feature_names)

        # SHAP FI
        shap_df = get_shap_fi(pipeline, X_test, feature_names)

        # Merge
        merged = fi_df.merge(shap_df, on="feature")
        merged["seed"] = seed

        all_rows.append(merged)

        # Spearman correlation giữa rank_shap và rank_gain_fi
        corr, pval = spearmanr(merged["rank_shap"], merged["rank_gain_fi"])
        corr_rows.append({
            "seed":       seed,
            "spearman_r": round(corr, 4),
            "p_value":    round(pval, 4),
        })
        logging.info(f"  Spearman r(SHAP, GainFI) = {corr:.4f} (p={pval:.4f})")

    # ----------------------------------------------------------
    # Tổng hợp
    # ----------------------------------------------------------
    raw_df = pd.concat(all_rows, ignore_index=True)

    # mean ± std rank theo từng feature, từng method
    rank_summary = (
        raw_df.groupby("feature")
        .agg(
            rank_shap_mean  =("rank_shap",    "mean"),
            rank_shap_std   =("rank_shap",    "std"),
            rank_gain_mean  =("rank_gain_fi", "mean"),
            rank_gain_std   =("rank_gain_fi", "std"),
            shap_imp_mean   =("shap_importance", "mean"),
            gain_imp_mean   =("gain_fi",       "mean"),
        )
        .reset_index()
    )

    # Tính rank gap trung bình (SHAP rank - GainFI rank, âm = SHAP xếp cao hơn)
    rank_summary["rank_gap"] = rank_summary["rank_shap_mean"] - rank_summary["rank_gain_mean"]
    rank_summary = rank_summary.sort_values("rank_shap_mean")

    corr_df = pd.DataFrame(corr_rows)

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    raw_path  = out_dir / f"{MODEL_NAME}_raw_ranks_all_seeds.csv"
    sum_path  = out_dir / f"{MODEL_NAME}_rank_mean_std.csv"
    corr_path = out_dir / f"{MODEL_NAME}_rank_correlation_per_seed.csv"

    raw_df.to_csv(raw_path,   index=False)
    rank_summary.to_csv(sum_path,  index=False)
    corr_df.to_csv(corr_path, index=False)

    logging.info(f"Saved: {raw_path}")
    logging.info(f"Saved: {sum_path}")
    logging.info(f"Saved: {corr_path}")

    # ----------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------
    logging.info("\n=== RANK SUMMARY (sorted by SHAP rank) ===")
    logging.info("\n" + rank_summary[
        ["feature", "rank_shap_mean", "rank_shap_std",
         "rank_gain_mean", "rank_gain_std", "rank_gap"]
    ].to_string(index=False))

    logging.info("\n=== SPEARMAN CORRELATION PER SEED ===")
    logging.info("\n" + corr_df.to_string(index=False))

    logging.info("XAI RANK STABILITY ANALYSIS COMPLETED")


if __name__ == "__main__":
    run_xai_rank_stability()