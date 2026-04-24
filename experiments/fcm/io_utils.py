from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from utils.project_paths import get_outputs_dir


def get_fcm_output_dir() -> Path:
    return get_outputs_dir("fcm")


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    path = get_fcm_output_dir() / filename
    df.to_csv(path, index=False)
    print("✔ Saved:", path)
    return path


def save_matrix_npy(matrix: np.ndarray, filename: str) -> Path:
    path = get_fcm_output_dir() / filename
    np.save(path, matrix)
    print("✔ Saved:", path)
    return path