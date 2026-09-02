"""Optimization utilities for fitting binding assay models."""

from core.optimizer.ensemble import (
    EnsembleResult,
    collapse,
    describe,
    describe_log10,
    select_representative_index,
)
from core.optimizer.filters import (
    calculate_fit_metrics,
    filter_by_r_squared,
    filter_by_rmse,
    select_valid_fits,
)
from core.optimizer.linear_fit import linear_regression
from core.optimizer.multistart import FitAttempt, generate_initial_guesses, multistart_minimize

__all__ = [
    # multistart.py
    'FitAttempt',
    'generate_initial_guesses',
    'multistart_minimize',
    # filters.py
    'filter_by_rmse',
    'filter_by_r_squared',
    'select_valid_fits',
    'calculate_fit_metrics',
    # ensemble.py
    'EnsembleResult',
    'collapse',
    'select_representative_index',
    'describe',
    'describe_log10',
    # linear_fit.py
    'linear_regression',
]
