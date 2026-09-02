"""Ensemble collapse: turn a pool of valid fits into one reported result.

This module is the **single source of operation** for collapsing the
multi-start (or per-replica) pool of valid fits into the values the app
reports. It does three things, each in one place:

1. **Select the representative** — :func:`select_representative_index`
   picks the single real fit that drives the plotted curve and the
   reported value/RMSE/R² (default: highest R², which is identical to
   lowest RMSE on a fixed dataset).
2. **Collapse the pool** — :func:`collapse` bundles the per-parameter
   sample pool, the per-trial quality pool, and the representative index
   into an :class:`EnsembleResult`.
3. **Summarise the pool** — :func:`describe` returns every statistic the
   app reports (centre, spread, and the two intervals) in one pass, so no
   caller has to pick an aggregation mode.

The module is pure ``numpy`` — it operates on arrays, not assays, so both
pipeline paths and the GUI reuse it. To change which fit is reported edit
:func:`select_representative_index`; to report a new statistic add a key
to :func:`describe`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Primitive per-parameter statistics — defined once, reused everywhere.
# ---------------------------------------------------------------------------


def _median(samples: np.ndarray) -> float:
    return float(np.median(samples))


def _mad(samples: np.ndarray) -> float:
    """Median absolute deviation — robust dispersion around the median."""
    med = np.median(samples)
    return float(np.median(np.abs(samples - med)))


def _mean(samples: np.ndarray) -> float:
    return float(np.mean(samples))


def _std(samples: np.ndarray) -> float:
    """Sample standard deviation; 0 for a single sample (no spread defined)."""
    return float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0


@dataclass
class EnsembleResult:
    """Mode-independent structural collapse of a valid-fit pool.

    Attributes
    ----------
    representative_index : int
        Index (into every ``parameter_samples`` / ``quality_samples``
        array) of the representative fit — the real fit that drives the
        curve and the reported value/RMSE/R².
    parameter_samples : dict[str, np.ndarray]
        One flat array per parameter key holding every valid trial's
        value, aligned by index.
    quality_samples : dict[str, np.ndarray]
        ``{'rmse': ..., 'r_squared': ...}`` — per-trial fit quality,
        aligned to ``parameter_samples`` by index.
    """

    representative_index: int
    parameter_samples: dict[str, np.ndarray]
    quality_samples: dict[str, np.ndarray]

    @property
    def representative_params(self) -> np.ndarray:
        """Representative parameter vector, ordered as ``parameter_samples``."""
        i = self.representative_index
        return np.array([arr[i] for arr in self.parameter_samples.values()], dtype=float)


def select_representative_index(quality: dict[str, np.ndarray]) -> int:
    """Index of the representative fit — the one place this criterion lives.

    Selects the highest R², which is identical to the lowest RMSE on a
    fixed dataset (``R² = 1 − SS_res/SS_tot``, ``RMSE = √(SS_res/n)`` are
    both monotone in ``SS_res``). R² is the more intuitive label.

    Parameters
    ----------
    quality : dict[str, np.ndarray]
        Must contain ``'r_squared'`` and ``'rmse'``, one value per valid
        trial (RMSE breaks R² ties).

    Returns
    -------
    int
        Index of the representative trial.
    """
    # Highest R², ties broken by lowest RMSE. On a fixed dataset R² and RMSE
    # are monotone (argmax R² == argmin RMSE); the tiebreak only matters when
    # R² collapses (e.g. constant y → ss_tot == 0 → every R² == 0), where RMSE
    # still identifies the genuinely best fit.
    r2 = np.asarray(quality['r_squared'], dtype=float)
    rmse = np.asarray(quality['rmse'], dtype=float)
    return int(np.lexsort((rmse, -r2))[0])


def collapse(
    param_matrix: np.ndarray,
    rmse: np.ndarray,
    r_squared: np.ndarray,
    parameter_keys: list[str],
) -> EnsembleResult:
    """Bundle a valid-fit pool into an :class:`EnsembleResult`.

    Parameters
    ----------
    param_matrix : np.ndarray
        ``(n_valid, n_params)`` array of valid-trial parameter vectors.
    rmse, r_squared : np.ndarray
        ``(n_valid,)`` per-trial quality, aligned to ``param_matrix`` rows.
    parameter_keys : list[str]
        Parameter names, ordered to match ``param_matrix`` columns.

    Returns
    -------
    EnsembleResult
        Pool, quality, and representative index.
    """
    param_matrix = np.asarray(param_matrix, dtype=float)
    quality = {
        'rmse': np.asarray(rmse, dtype=float),
        'r_squared': np.asarray(r_squared, dtype=float),
    }
    parameter_samples = {key: param_matrix[:, i].copy() for i, key in enumerate(parameter_keys)}
    return EnsembleResult(
        representative_index=select_representative_index(quality),
        parameter_samples=parameter_samples,
        quality_samples=quality,
    )


def describe(samples: np.ndarray) -> dict[str, float]:
    """Full per-parameter summary of a 1-D pool: center, spread, and range.

    Returns ``{'median', 'mad', 'mean', 'std', 'p16', 'p84', 'min', 'max'}``
    — the median/MAD and mean/SD pairs, the central-68% interval (16th and
    84th percentiles, the distribution-free analog of ±1 SD), and the observed
    range, all computed directly from *samples*.
    """
    s = np.asarray(samples, dtype=float)
    p16, p84 = np.percentile(s, [16, 84])
    return {
        'median': _median(s),
        'mad': _mad(s),
        'mean': _mean(s),
        'std': _std(s),
        'p16': float(p16),
        'p84': float(p84),
        'min': float(np.min(s)),
        'max': float(np.max(s)),
    }


def describe_log10(samples: np.ndarray) -> dict[str, float]:
    """Summary of ``log10(samples)`` — transform FIRST, then aggregate.

    The statistics of ``log10(Ka)`` must be computed from the per-fit log10
    values, never by taking ``log10`` of a Ka spread statistic:
    ``log10(MAD_Ka)`` is meaningless and ``log10(mean Ka)`` is Jensen-biased.
    Requires strictly positive samples (Ka > 0) — raises rather than emit a
    silent ``-inf`` / ``nan``.
    """
    s = np.asarray(samples, dtype=float)
    if s.size == 0 or float(np.min(s)) <= 0.0:
        raise ValueError('describe_log10 requires strictly positive samples (Ka > 0).')
    return describe(np.log10(s))
