import numpy as np
from scipy.stats import t

def prediction_interval(data, avg_value):
    n = len(data)
    mean = np.mean(data)
    if n > 1:
        std_dev = np.std(data, ddof=1)
        margin_of_error = std_dev * np.sqrt(1 + 1/n) * t.ppf(0.975, n-1)
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
