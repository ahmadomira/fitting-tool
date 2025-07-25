import numpy as np
from scipy.stats import t


def prediction_interval(data, avg_value):
    n = len(data)
    mean = np.mean(data)
    if n > 1:
        std_dev = np.std(data, ddof=1)
        margin_of_error = std_dev * np.sqrt(1 + 1 / n) * t.ppf(0.975, n - 1)
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        lower_bound = min(lower_bound, avg_value * 0.75)
        upper_bound = max(upper_bound, avg_value * 1.25)
    else:
        margin_of_error = "not applicable"
        std_dev = "not applicable"
        lower_bound = "not applicable"
        upper_bound = "not applicable"
    return mean, lower_bound, upper_bound, std_dev


def round_to_sigfigs(value, sigfigs=4):
    if isinstance(value, (int, float)):
        return float(f"{value:.{sigfigs}g}")
    return value


def detect_outliers_per_point(data, reference, relative_threshold):
    """
    Detect outliers based on relative threshold compared to reference values.

    Parameters:
    data (array-like): Data points to check for outliers
    reference (array-like): Reference values to compare against
    relative_threshold (float): Threshold as a fraction of reference values

    Returns:
    numpy.ndarray: Indices of outlier data points
    """
    deviations = np.abs(data - reference)
    outlier_indices = np.where(deviations > relative_threshold * reference)[0]
    return outlier_indices
