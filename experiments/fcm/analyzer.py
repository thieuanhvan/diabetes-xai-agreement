from __future__ import annotations

import numpy as np
import pandas as pd


def compute_fcm_strength(W: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Compute FCM node strength as sum of absolute outgoing weights.

    Returns
    -------
    DataFrame with columns:
        - feature
        - fcm_strength
    """
    strength = np.sum(np.abs(W), axis=1)

    return pd.DataFrame({
        "feature": feature_names,
        "fcm_strength": strength
    })