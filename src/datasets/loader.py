from __future__ import annotations

import os
import sys
import pandas as pd

# ------------------------------------------------------------
# Allow running this file directly (no PyCharm settings needed)
# ------------------------------------------------------------
if __package__ is None or __package__ == "":
    # Running as a script: python src/datasets/loader.py
    # -> add <project_root>/src to sys.path so absolute imports work
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    SRC_PATH = os.path.join(PROJECT_ROOT, "src")
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    # Use absolute import when executed as script
    from datasets.dataset_registry import (
        DATASETS,
        get_active_dataset,
        get_dataset_config,
    )
else:
    # Normal package import
    from .dataset_registry import (
        DATASETS,
        get_active_dataset,
        get_dataset_config,
    )


def _apply_include_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = cfg.get("include_columns")
    if cols is None:
        return df
    # Ensure target is included
    target = cfg["target"]
    final_cols = list(cols)
    if target not in final_cols:
        final_cols.append(target)
    return df[final_cols]


def load_active_dataset():
    """
    Load dataset based on ACTIVE_DATASET in registry.
    Returns (df, cfg).
    """
    cfg = get_active_dataset()
    df = pd.read_csv(cfg["path"])
    df = _apply_include_columns(df, cfg)
    return df, cfg


def load_dataset_by_name(name: str):
    """
    Load dataset by registry key name (e.g. 'cdc_brfss_2021').
    Returns (df, cfg).
    """
    cfg = get_dataset_config(name)
    df = pd.read_csv(cfg["path"])
    df = _apply_include_columns(df, cfg)
    return df, cfg


def load_all_datasets():
    """
    Convenience loader: returns dict[name] = (df, cfg)
    """
    out = {}
    for name in DATASETS.keys():
        out[name] = load_dataset_by_name(name)
    return out


# Optional: quick self-test when running directly
#if __name__ == "__main__":
#    df, cfg = load_active_dataset()
#    print("Loaded ACTIVE dataset:", cfg.get("name", "(unknown)"))
#    print("Shape:", df.shape)
#    print("Columns:", list(df.columns)[:10], "...")