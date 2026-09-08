"""Minimalist preprocessing pipeline for MeasurementSet.

Provides a tiny registry of named preprocessing steps (simple callables)
and a helper to apply a sequence of steps.  Mirrors the I/O registry
pattern: dict-based, no decorators, no metaclasses.

Usage::

    from core.data_processing.preprocessing import apply_preprocessing

    steps = [
        {"name": "zscore_replica_filter", "params": {"threshold": 2.5}},
    ]
    apply_preprocessing(ms, steps)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Protocol, runtime_checkable

import numpy as np

from core.data_processing.measurement_set import MeasurementSet

logger = logging.getLogger(__name__)


# ======================================================================
# Protocol
# ======================================================================


@runtime_checkable
class PreprocessingStep(Protocol):
    """Interface for a preprocessing step.

    Implementations modify a ``MeasurementSet`` **in-place** (typically
    the ``_active_mask`` and ``processing_log``).  Raw data is never
    mutated.
    """

    name: str

    def __call__(self, ms: MeasurementSet) -> None: ...


# ======================================================================
# Registry
# ======================================================================

# Maps string name → factory(params_dict) → PreprocessingStep
PREPROCESSING_STEPS: Dict[str, Callable[[Dict[str, Any]], PreprocessingStep]] = {}


def register_step(name: str, factory: Callable[[Dict[str, Any]], PreprocessingStep]) -> None:
    """Register a preprocessing step factory under *name*.

    Parameters
    ----------
    name : str
        Lookup key (e.g. ``"zscore_replica_filter"``).
    factory : callable
        ``factory(params_dict) -> PreprocessingStep``.
    """
    PREPROCESSING_STEPS[name] = factory


def get_step(name: str, params: Dict[str, Any] | None = None) -> PreprocessingStep:
    """Instantiate a registered preprocessing step.

    Parameters
    ----------
    name : str
        Registered step name.
    params : dict, optional
        Keyword arguments forwarded to the factory.

    Returns
    -------
    PreprocessingStep

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in PREPROCESSING_STEPS:
        available = list(PREPROCESSING_STEPS.keys())
        raise KeyError(f"Unknown preprocessing step '{name}'. Available: {available}")
    return PREPROCESSING_STEPS[name](params or {})


# ======================================================================
# Runner
# ======================================================================


def apply_preprocessing(
    ms: MeasurementSet,
    steps: List[Dict[str, Any]],
) -> None:
    """Apply a sequence of preprocessing steps to *ms* in order.

    Each entry in *steps* is a dict with ``"name"`` (str) and optional
    ``"params"`` (dict).  Example::

        [
            {"name": "zscore_replica_filter", "params": {"threshold": 2.5}},
        ]

    Parameters
    ----------
    ms : MeasurementSet
        Modified in-place (active mask and processing log).
    steps : list[dict]
        Ordered list of step specifications.
    """
    for spec in steps:
        name = spec['name']
        params = spec.get('params', {})
        step = get_step(name, params)
        logger.info("Applying preprocessing step '%s' with params %s", name, params)
        step(ms)


# ======================================================================
# Built-in step: Z-score replica filter
# ======================================================================


class ZScoreReplicaFilter:
    """Mark replicas as inactive if any point is a z-score outlier.

    Uses **robust statistics** (median and MAD) to avoid the masking
    effect where a single extreme outlier inflates the mean and std,
    making its own z-score artificially low.

    At each concentration point the median and MAD (median absolute
    deviation) are computed across *active* replicas.  The modified
    z-score is:

    .. math::

        z_i = 0.6745 \\cdot \\frac{|x_i - \\text{median}|}{\\text{MAD}}

    The 0.6745 factor normalises MAD to be comparable with the standard
    deviation of a normal distribution.

    If any point in a replica has ``|z| > threshold``, the **entire
    replica** is deactivated.

    Parameters
    ----------
    threshold : float
        Modified z-score threshold (default 3.5, a conventional choice
        for the modified z-score method).
    min_replicas : int
        Minimum number of active replicas required; if fewer, the step
        is skipped with a warning (default 3).
    """

    name: str = 'zscore_replica_filter'

    def __init__(self, threshold: float = 3.5, min_replicas: int = 3) -> None:
        if threshold <= 0:
            raise ValueError(f'threshold must be positive, got {threshold}')
        if min_replicas < 2:
            raise ValueError(f'min_replicas must be >= 2, got {min_replicas}')
        self.threshold = threshold
        self.min_replicas = min_replicas

    def __call__(self, ms: MeasurementSet) -> None:
        """Apply modified z-score filtering to *ms* in-place."""
        active_ids = ms.active_replica_ids

        if len(active_ids) < self.min_replicas:
            msg = f'zscore_replica_filter skipped: only {len(active_ids)} active replicas (minimum {self.min_replicas} required)'
            logger.warning(msg)
            ms.processing_log.append(
                {
                    'step': self.name,
                    'status': 'skipped',
                    'reason': msg,
                    'threshold': self.threshold,
                    'min_replicas': self.min_replicas,
                    'n_active': len(active_ids),
                }
            )
            return

        # Gather active signals → (n_active, n_points)
        active_indices = [i for i, a in enumerate(ms._active_mask) if a]
        active_signals = ms.signals[active_indices]

        # Robust location and scale per point
        median = np.median(active_signals, axis=0)
        abs_dev = np.abs(active_signals - median)
        mad = np.median(abs_dev, axis=0)

        # Modified z-score: 0.6745 normalises MAD to σ-equivalent
        # A zero MAD can also arise from a tied majority. This policy assigns z = 0
        # to all replicas at that point, including any deviations from that majority.
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.where(mad > 0, 0.6745 * abs_dev / mad, 0.0)

        # Maximum |z| per replica
        max_z = z_scores.max(axis=1)  # shape (n_active,)

        dropped: list[str] = []
        for local_idx, global_idx in enumerate(active_indices):
            if max_z[local_idx] > self.threshold:
                rid = ms.replica_ids[global_idx]
                ms.set_active(rid, False)
                dropped.append(rid)

        kept = ms.active_replica_ids
        logger.info(
            'zscore_replica_filter: dropped %d replica(s) %s, kept %d',
            len(dropped),
            dropped,
            len(kept),
        )

        ms.processing_log.append(
            {
                'step': self.name,
                'status': 'applied',
                'threshold': self.threshold,
                'dropped': dropped,
                'kept': kept,
            }
        )


# Auto-register on import
register_step(
    'zscore_replica_filter',
    lambda params: ZScoreReplicaFilter(**params),
)
