"""Linear fitting for dye-alone calibration."""

from typing import Tuple

import numpy as np


def linear_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Perform simple linear regression.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Dependent variable values.

    Returns
    -------
    Tuple[float, float, float, float]
        (slope, intercept, r_squared, rmse)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError('Linear regression requires matching 1-D concentration and signal arrays')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Linear regression requires finite concentrations and signals')
    n = len(x)
    if n < 2:
        raise ValueError('Need at least 2 points for linear regression')
    if np.all(x == x[0]):
        raise ValueError('Need at least 2 distinct concentrations to estimate slope and intercept')

    # Use numpy's polyfit for numerical stability
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]

    # Calculate predictions and metrics
    y_pred = slope * x + intercept
    residuals = y - y_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals**2))

    return slope, intercept, r_squared, rmse
