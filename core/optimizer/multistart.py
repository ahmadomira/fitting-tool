"""Multi-start optimization for nonlinear fitting.

This module provides a robust multi-start L-BFGS-B optimizer for fitting
binding assay models. The multi-start approach helps avoid local minima
by running the optimizer from multiple random initial points.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from core.optimizer.scaling import ParamScaler


@dataclass
class FitAttempt:
    """Result from a single optimization attempt.

    Attributes
    ----------
    params : np.ndarray
        Fitted parameter values.
    cost : float
        Final cost function value (sum of squared residuals).
    rmse : float
        Root mean squared error.
    r_squared : float
        Coefficient of determination (R²).
    success : bool
        Whether the optimizer converged.
    """

    params: np.ndarray
    cost: float
    rmse: float
    r_squared: float
    success: bool


def generate_initial_guesses(
    n_trials: int,
    bounds: List[Tuple[float, float]],
    log_scale_params: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """Generate random initial parameter guesses within bounds.

    Parameters
    ----------
    n_trials : int
        Number of initial guesses to generate.
    bounds : List[Tuple[float, float]]
        (lower, upper) bounds for each parameter.
    log_scale_params : List[int], optional
        Indices of parameters that should be sampled in log space.
        This is useful for parameters like Ka that span many orders of magnitude.

    Returns
    -------
    List[np.ndarray]
        List of initial parameter arrays.
    """
    if log_scale_params is None:
        log_scale_params = []

    initial_guesses = []
    for _ in range(n_trials):
        guess = np.empty(len(bounds))
        for i, (lower, upper) in enumerate(bounds):
            if i in log_scale_params and lower > 0 and upper > 0:
                # Sample in log space for parameters spanning orders of magnitude
                log_lower = np.log10(lower)
                log_upper = np.log10(upper)
                guess[i] = 10 ** np.random.uniform(log_lower, log_upper)
            else:
                guess[i] = np.random.uniform(lower, upper)
        initial_guesses.append(guess)

    return initial_guesses


def multistart_minimize(
    objective: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    n_trials: int = 100,
    initial_guesses: Optional[List[np.ndarray]] = None,
    log_scale_params: Optional[List[int]] = None,
    compute_metrics: Optional[Callable[[np.ndarray], Tuple[float, float]]] = None,
    scaler: Optional[ParamScaler] = None,
) -> List[FitAttempt]:
    """Run L-BFGS-B optimizer from multiple starting points.

    Parameters
    ----------
    objective : Callable[[np.ndarray], float]
        Raw-space objective; should return sum of squared residuals.
    bounds : List[Tuple[float, float]]
        (lower, upper) bounds for each parameter, in raw space.
    n_trials : int
        Number of optimization runs.
    initial_guesses : List[np.ndarray], optional
        Pre-computed raw-space initial guesses. If None, generated randomly.
    log_scale_params : List[int], optional
        Indices of parameters to sample in log space for random initialization.
    compute_metrics : Callable, optional
        Function that takes raw params and returns (rmse, r_squared).
        If None, RMSE and R² are NaN: a scalar cost does not provide the
        observation count needed to convert a sum of squares into RMSE.
    scaler : ParamScaler, optional
        If provided, optimizer runs in the scaler's internal (tilded) space;
        callers still pass and receive raw parameters.

    Returns
    -------
    List[FitAttempt]
        Raw-space attempts sorted by raw cost (ascending).
    """
    if initial_guesses is None:
        initial_guesses = generate_initial_guesses(n_trials, bounds, log_scale_params)

    opt_bounds, opt_objective, opt_initial = bounds, objective, initial_guesses
    if scaler is not None:
        opt_bounds = scaler.bounds_to_internal(bounds)
        opt_objective = scaler.wrap_objective(objective)
        opt_initial = [scaler.to_internal(g) for g in initial_guesses]

    results = []
    for opt_guess in opt_initial:
        try:
            result = minimize(opt_objective, opt_guess, method='L-BFGS-B', bounds=opt_bounds)

            if scaler is not None:
                params_raw = scaler.to_external(result.x)
                cost_raw = float(result.fun) * scaler.loss_scale
            else:
                params_raw = result.x
                cost_raw = float(result.fun)

            if compute_metrics is not None:
                rmse, r_squared = compute_metrics(params_raw)
            else:
                rmse = np.nan
                r_squared = np.nan

            results.append(
                FitAttempt(
                    params=params_raw,
                    cost=cost_raw,
                    rmse=rmse,
                    r_squared=r_squared,
                    success=result.success,
                )
            )
        except Exception:
            # Skip failed optimizations
            continue

    # Sort by cost (best first)
    results.sort(key=lambda x: x.cost)
    return results
