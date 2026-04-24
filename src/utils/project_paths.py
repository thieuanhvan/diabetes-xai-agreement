from __future__ import annotations

from pathlib import Path
from datetime import datetime
from datasets.dataset_registry import ACTIVE_DATASET
from pathlib import Path

# =========================================================
# Detect project root
# =========================================================
def get_project_root() -> Path:
    """
    Dynamically detect the project root.

    The project root is defined as the directory that contains
    the 'src' folder.

    This works regardless of where the script is executed from.
    """

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "src").exists():
            return parent

    raise RuntimeError("Project root not found. Folder containing 'src' is required.")


# =========================================================
# Core project paths
# =========================================================
PROJECT_ROOT = get_project_root()

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "configs"
LOG_DIR = PROJECT_ROOT / "logs"


# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Output helpers
# =========================================================
def get_outputs_dir(subfolder: str | None = None) -> Path:
    """
    Return outputs directory or a subfolder inside outputs.

    Example:
        outputs/shap
        outputs/analysis
        outputs/statistics
    """

    # base = OUTPUT_DIR
    base = OUTPUT_DIR / ACTIVE_DATASET

    if subfolder:
        path = base / subfolder
        path.mkdir(parents=True, exist_ok=True)
        return path

    return base


# =========================================================
# Logs
# =========================================================
def get_logs_dir() -> Path:
    """
    Return logs directory.
    """
    return LOG_DIR


def get_log_file(run_type: str) -> Path:
    """
    Create a timestamped log file.

    Example filename:
        pipeline_20260312143022.log
    """

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{run_type}_{timestamp}.log"

    return LOG_DIR / filename


# =========================================================
# Debug helper
# =========================================================
def print_project_structure() -> None:
    """
    Debug helper to verify resolved paths.
    """

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("SRC_DIR:", SRC_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("CONFIG_DIR:", CONFIG_DIR)
    print("LOG_DIR:", LOG_DIR)

    # utils/path_utils.py

