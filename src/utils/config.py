"""
Config loader — single source of truth for hyperparameters and pipeline settings.

Usage:
    from src.utils.config import load_config

    cfg = load_config()                    # loads configs/default.yaml
    cfg = load_config("quick_test")        # loads configs/quick_test.yaml

    n_estimators = cfg["models"]["random_forest"]["n_estimators"]
    sample_size = cfg["xai"]["shap"]["sample_size"]

Note (2026-05-01):
    This file is currently used as a documentation source — the YAML mirrors
    the hardcoded values in the pipeline so reviewers can inspect the full
    configuration at a glance. A subsequent refactor will wire `load_config()`
    into baselines.py / proposal.py / step02_preprocessing.py to remove the
    hardcoded duplicates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.project_paths import CONFIG_DIR


def load_config(name: str = "default") -> dict[str, Any]:
    """Load YAML config from configs/{name}.yaml.

    Parameters
    ----------
    name : str
        Config file stem (without .yaml). Defaults to "default".

    Returns
    -------
    dict
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If configs/{name}.yaml does not exist.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    path: Path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Expected at {CONFIG_DIR}/."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
