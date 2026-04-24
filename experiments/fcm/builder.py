"""
Build FCM weight matrix using correlation.
"""

import numpy as np


def build_correlation_fcm_weights(X):

    """
    Build correlation-based weight matrix.

    Parameters
    ----------
    X : numpy array
        dataset matrix (features + target)

    Returns
    -------
    W : numpy array
        correlation weight matrix
    """

    corr = np.corrcoef(X.T)

    np.fill_diagonal(corr, 0)

    return corr