from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

# Ensure UTF-8
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASETS = [
    "cdc_brfss_2015_rebuilt",
    "cdc_brfss_2021_rebuilt",
    "cdc_brfss_2023_rebuilt",
]

DEFAULT_MODELS = [
    "xgboost",
    "random_forest",
    "logistic_regression",
]

_BOOTSTRAP = r'''
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

REPO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "src")

for p in (REPO, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

ACTIVE_DATASET = "__DATASET__"
ACTIVE_MODEL   = "__MODEL__"

import src.datasets.dataset_registry as _a
import datasets.dataset_registry     as _b

for _m in (_a, _b):
    _m.ACTIVE_DATASET = ACTIVE_DATASET
    _m.ACTIVE_MODEL   = ACTIVE_MODEL

from src.utils.logging_config import configure_logging
from src.pipelines.run_pipeline import run_all_steps

configure_logging()

import logging
logging.info("=" * 78)
logging.info(f"[BOOTSTRAP] DATASET={ACTIVE_DATASET} MODEL={ACTIVE_MODEL}")
logging.info("=" * 78)

run_all_steps()
'''


def run_one(dataset: str, model: str):
    bootstrap_file = REPO_ROOT / f".tmp_{dataset}_{model}.py"
    bootstrap_file.write_text(
        _BOOTSTRAP.replace("__DATASET__", dataset).replace("__MODEL__", model),
        encoding="utf-8"
    )

    logging.info("#" * 78)
    logging.info(f"START {dataset} x {model}")
    logging.info("#" * 78)

    t0 = time.time()

    result = subprocess.run(
        [sys.executable, str(bootstrap_file)],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONUTF8": "1"},
    )

    elapsed = time.time() - t0

    if result.returncode == 0:
        logging.info(f"END {dataset} x {model} OK ({elapsed:.1f}s)")
    else:
        logging.error(f"END {dataset} x {model} FAILED")

    bootstrap_file.unlink(missing_ok=True)


def main():
    from src.utils.logging_config import configure_logging
    configure_logging()

    logging.info("=" * 78)
    logging.info("RUN ALL COMBOS (CLEAN VERSION)")
    logging.info("=" * 78)

    for dataset, model in product(DEFAULT_DATASETS, DEFAULT_MODELS):
        run_one(dataset, model)

    logging.info("=" * 78)
    logging.info("ALL COMBOS DONE")
    logging.info("=" * 78)


if __name__ == "__main__":
    main()