from __future__ import annotations

import pandas as pd
from pathlib import Path
from utils.project_paths import get_outputs_dir


def get_analysis_output_dir() -> Path:
    return get_outputs_dir("analysis")


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    path = get_analysis_output_dir() / filename
    df.to_csv(path, index=False)
    print("✔ Saved:", path)
    return path