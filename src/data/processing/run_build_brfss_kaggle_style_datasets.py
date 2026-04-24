from __future__ import annotations

from pathlib import Path
import sys
import logging
from datetime import datetime

import pandas as pd
import numpy as np

"""
Build BRFSS Kaggle-style datasets from CDC raw XPT files.

Years:
    2015, 2021, 2023

Input:
    data/raw/cdc/LLCP2015.XPT
    data/raw/cdc/LLCP2021.XPT
    data/raw/cdc/LLCP2023.XPT

Output:
    data/processed/cdc_brfss_2015_rebuilt.csv
    data/processed/cdc_brfss_2021_rebuilt.csv
    data/processed/cdc_brfss_2023_rebuilt.csv
"""


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cdc"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = {
    2015: RAW_DIR / "LLCP2015.XPT",
    2021: RAW_DIR / "LLCP2021.XPT",
    2023: RAW_DIR / "LLCP2023.XPT",
}

OUTPUT_FILES = {
    2015: OUTPUT_DIR / "cdc_brfss_2015_rebuilt.csv",
    2021: OUTPUT_DIR / "cdc_brfss_2021_rebuilt.csv",
    2023: OUTPUT_DIR / "cdc_brfss_2023_rebuilt.csv",
}

KAGGLE_COLUMNS = [
    "Diabetes_binary",
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]


# ============================================================
# LOGGING
# ============================================================

def setup_logger(script_name: str = "run_build_brfss_kaggle_style_datasets") -> Path:
    """
    Configure timestamped logging for this run script.

    Each execution creates one separate log file under logs/.
    Logs are written to both console and file.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{script_name}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicated logs when rerunning from IDE.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("=" * 80)
    logging.info("START SCRIPT: %s.py", script_name)
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Raw dir: %s", RAW_DIR)
    logging.info("Output dir: %s", OUTPUT_DIR)
    logging.info("Log file: %s", log_file)
    logging.info("=" * 80)

    return log_file


# ============================================================
# IO
# ============================================================

def read_xpt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    logging.info("Reading XPT file: %s", path)
    df = pd.read_sas(path, format="xport", encoding="utf-8")
    df.columns = [str(c).strip().upper() for c in df.columns]

    logging.info("Loaded raw XPT shape: %s", df.shape)
    return df


# ============================================================
# HELPERS
# ============================================================

def clean_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace(
        [
            7, 8, 9,
            77, 88, 99,
            777, 888, 999,
            7777, 8888, 9999,
        ],
        np.nan,
    )
    return s


def binary_yes_no(series: pd.Series) -> pd.Series:
    """
    Common BRFSS yes/no convention:
        1 = yes
        2 = no
    """
    s = clean_numeric(series)
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s == 1] = 1.0
    out[s == 2] = 0.0
    return out


def pick_existing_column(df: pd.DataFrame, candidates: list[str], field_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            logging.info("Mapped field '%s' to raw column '%s'", field_name, c)
            return c

    raise KeyError(f"{field_name} not found. Tried {candidates}")


def pick_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fill_mode(series: pd.Series, default_value: float = 0.0) -> pd.Series:
    """
    Fill NaN with mode if available, otherwise use default.
    """
    s = series.copy()
    non_null = s.dropna()

    if len(non_null) == 0:
        return s.fillna(default_value)

    mode_val = non_null.mode()

    if len(mode_val) == 0:
        return s.fillna(default_value)

    return s.fillna(mode_val.iloc[0])


def optional_binary_field(df: pd.DataFrame, candidates: list[str], field_name: str) -> pd.Series:
    col = pick_optional_column(df, candidates)

    if col is None:
        logging.warning(
            "%s missing in raw dataset. Filling whole output column with 0. Tried candidates: %s",
            field_name,
            candidates,
        )
        return pd.Series(0.0, index=df.index, dtype="float")

    logging.info("Mapped optional field '%s' to raw column '%s'", field_name, col)
    return binary_yes_no(df[col])


# ============================================================
# DEBUG
# ============================================================

def print_debug_columns(df: pd.DataFrame, year: int) -> None:
    logging.info("--- DEBUG COLUMN SEARCH FOR %s ---", year)

    patterns = [
        "HYPE", "BPHIGH", "TOLDHI", "CHOLCHK",
        "FRT", "FRUIT", "VEG", "VEGET",
        "RFDRH", "HLTHPLN", "MEDCOST",
    ]

    for patt in patterns:
        matches = [c for c in df.columns if patt in c]
        logging.info("%s: %s", patt, matches)


# ============================================================
# TARGET
# ============================================================

def build_target(df: pd.DataFrame, year: int) -> pd.Series:
    col = "DIABETE3" if year == 2015 else "DIABETE4"

    if col not in df.columns:
        raise KeyError(f"Target column {col} not found for year {year}")

    logging.info("Using target column for %s: %s", year, col)

    s = clean_numeric(df[col])

    # Kaggle-style binary target:
    # positive: diabetes OR prediabetes/borderline
    # negative: no diabetes OR gestational-only
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s.isin([1, 4])] = 1.0
    out[s.isin([2, 3])] = 0.0

    return out


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(df: pd.DataFrame, year: int) -> pd.DataFrame:
    logging.info("Building Kaggle-style dataset for year %s", year)

    out = pd.DataFrame(index=df.index)

    out["Diabetes_binary"] = build_target(df, year)

    # HighBP
    col = pick_existing_column(
        df,
        ["BPHIGH4", "BPHIGH6", "BPHIGH7", "HIGHBP", "_RFHYPE5", "_RFHYPE6", "_RFHYPE7", "_RFHYPE8", "_RFHYPE9"],
        "HighBP",
    )
    out["HighBP"] = binary_yes_no(df[col])

    # HighChol
    col = pick_existing_column(df, ["TOLDHI2", "TOLDHI3"], "HighChol")
    out["HighChol"] = binary_yes_no(df[col])

    # CholCheck
    col = pick_existing_column(df, ["CHOLCHK", "CHOLCHK3"], "CholCheck")
    out["CholCheck"] = binary_yes_no(df[col])

    # BMI
    col = pick_existing_column(df, ["_BMI5"], "BMI")
    out["BMI"] = clean_numeric(df[col]) / 100.0

    # Smoker
    col = pick_existing_column(df, ["SMOKE100"], "Smoker")
    out["Smoker"] = binary_yes_no(df[col])

    # Stroke
    col = pick_existing_column(df, ["CVDSTRK3"], "Stroke")
    out["Stroke"] = binary_yes_no(df[col])

    # HeartDiseaseorAttack
    col = pick_existing_column(df, ["CVDCRHD4"], "HeartDiseaseorAttack")
    out["HeartDiseaseorAttack"] = binary_yes_no(df[col])

    # PhysActivity
    col = pick_existing_column(df, ["EXERANY2"], "PhysActivity")
    out["PhysActivity"] = binary_yes_no(df[col])

    # Fruits
    out["Fruits"] = optional_binary_field(
        df,
        ["_FRTLT1A", "_FRTLT1", "_FRUITEX", "_FRUITE1", "FRUIT2", "FRUIT1"],
        "Fruits",
    )

    # Veggies
    out["Veggies"] = optional_binary_field(
        df,
        ["_VEGLT1A", "_VEGLT1", "_VEGETEX", "_VEGETE1", "_RFVEG23", "VEGETAB1", "VEGETAB2"],
        "Veggies",
    )

    # HvyAlcoholConsump
    col = pick_existing_column(df, ["_RFDRHV5", "_RFDRHV7", "_RFDRHV8", "_RFDRHV9"], "HvyAlcoholConsump")
    out["HvyAlcoholConsump"] = binary_yes_no(df[col])

    # AnyHealthcare
    out["AnyHealthcare"] = optional_binary_field(
        df,
        ["HLTHPLN1", "_HLTHPLN", "_HLTHPL2"],
        "AnyHealthcare",
    )

    # NoDocbcCost
    col = pick_existing_column(df, ["MEDCOST", "MEDCOST1"], "NoDocbcCost")
    out["NoDocbcCost"] = binary_yes_no(df[col])

    # GenHlth
    out["GenHlth"] = clean_numeric(df["GENHLTH"])

    # MentHlth
    out["MentHlth"] = clean_numeric(df["MENTHLTH"])

    # PhysHlth
    out["PhysHlth"] = clean_numeric(df["PHYSHLTH"])

    # DiffWalk
    out["DiffWalk"] = binary_yes_no(df["DIFFWALK"])

    # Sex
    col = pick_existing_column(df, ["SEX", "SEXVAR", "_SEX"], "Sex")
    out["Sex"] = clean_numeric(df[col])

    # Age
    col = pick_existing_column(df, ["_AGEG5YR", "CAGEG"], "Age")
    out["Age"] = clean_numeric(df[col])

    # Education
    out["Education"] = clean_numeric(df["EDUCA"])

    # Income
    col = pick_existing_column(df, ["INCOME2", "INCOME3"], "Income")
    out["Income"] = clean_numeric(df[col])

    # Keep schema fixed
    out = out[KAGGLE_COLUMNS].copy()

    before_drop = len(out)

    # Drop rows with missing target only
    out = out.dropna(subset=["Diabetes_binary"]).copy()

    after_drop = len(out)
    logging.info(
        "Dropped rows with missing target for %s: %d -> %d (dropped=%d)",
        year,
        before_drop,
        after_drop,
        before_drop - after_drop,
    )

    # Impute remaining missing values without using -1
    bmi_median = out["BMI"].median()
    out["BMI"] = out["BMI"].fillna(bmi_median)
    logging.info("BMI median imputation value for %s: %.4f", year, bmi_median)

    # Binary / ordinal / categorical-like columns -> mode
    for c in out.columns:
        if c in {"Diabetes_binary", "BMI"}:
            continue

        missing_before = int(out[c].isna().sum())
        out[c] = fill_mode(out[c], default_value=0.0)
        missing_after = int(out[c].isna().sum())

        if missing_before > 0:
            logging.info(
                "Imputed column '%s' for %s: missing %d -> %d",
                c,
                year,
                missing_before,
                missing_after,
            )

    # Cast integer columns
    for c in out.columns:
        if c != "BMI":
            out[c] = out[c].astype(int)

    logging.info("Final dataset shape for %s: %s", year, out.shape)

    return out


# ============================================================
# SUMMARY
# ============================================================

def log_dataset_summary(out: pd.DataFrame, year: int) -> None:
    logging.info("Dataset summary for %s", year)
    logging.info("Rows: %d", len(out))
    logging.info("Columns: %d", out.shape[1])

    counts = out["Diabetes_binary"].value_counts().sort_index()
    ratios = out["Diabetes_binary"].value_counts(normalize=True).sort_index()

    logging.info("Target counts:\n%s", counts.to_string())
    logging.info("Target ratios:\n%s", ratios.to_string())

    logging.info("Missing values after processing: %d", int(out.isna().sum().sum()))

    logging.info("Column list: %s", list(out.columns))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log_file = setup_logger()
    start_time = datetime.now()

    try:
        for year, path in INPUT_FILES.items():
            logging.info("=" * 80)
            logging.info("Processing year: %s", year)
            logging.info("Input file: %s", path)
            logging.info("Output file: %s", OUTPUT_FILES[year])
            logging.info("=" * 80)

            df = read_xpt(path)
            print_debug_columns(df, year)

            out = build_dataset(df, year)
            log_dataset_summary(out, year)

            out.to_csv(OUTPUT_FILES[year], index=False)
            logging.info("Saved processed dataset: %s", OUTPUT_FILES[year])

        elapsed = (datetime.now() - start_time).total_seconds()

        logging.info("=" * 80)
        logging.info("DONE SUCCESS")
        logging.info("Elapsed time: %.2f seconds", elapsed)
        logging.info("Log file: %s", log_file)
        logging.info("=" * 80)

    except Exception:
        elapsed = (datetime.now() - start_time).total_seconds()
        logging.exception("FAILED after %.2f seconds", elapsed)
        logging.info("Log file: %s", log_file)
        sys.exit(1)


if __name__ == "__main__":
    main()