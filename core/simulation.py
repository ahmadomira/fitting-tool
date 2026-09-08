"""Forward simulation of titration data.

This is the inverse of the fitting pipeline: given an assay, its conditions, and
explicit parameter *values*, evaluate the **same forward model the fitter uses**
over a chosen concentration vector — no fitting, no divergent math.  The GUI
simulation applet uses this for experiment design (choosing concentration ranges,
point counts, tolerable noise before bench time).

All math is reused from :mod:`core.assays` / :mod:`core.models`; this module only
builds the concentration grid, drives the assay's ``forward_model``, and optionally
adds measurement noise.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Mapping, Sequence, Type

import numpy as np

from core.assays.base import BaseAssay
from core.data_processing.measurement_set import MeasurementSet
from core.units import Q_

__all__ = ['build_concentration_vector', 'simulate_signal', 'simulate_species', 'simulate_dataset']


def _as_count(n: Any) -> int:
    """Coerce *n* to an int point-count of at least 2, else raise ``ValueError``."""
    n_int = int(n)
    if n_int < 2:
        raise ValueError(f'Number of points must be at least 2 (got {n}).')
    return n_int


def build_concentration_vector(mode: str, **kwargs: Any) -> np.ndarray:
    """Build a titrant concentration vector (Molar) from a chosen input mode.

    Concentrations are physical quantities and **cannot be negative**; every mode
    rejects ``x < 0`` (``x = 0`` is allowed — the forward models handle it).  All
    inputs are interpreted in base units (Molar); unit conversion happens at the
    GUI boundary.

    Parameters
    ----------
    mode : str
        One of ``'explicit'``, ``'linear'``, ``'step'``, ``'log'``.
    **kwargs
        Mode-specific arguments (all concentrations in M):

        * ``explicit`` — ``values``: a sequence of concentrations.
        * ``linear``   — ``start``, ``stop``, ``n``: ``np.linspace(start, stop, n)``.
        * ``step``     — ``start``, ``step``, ``n``: ``start + step * arange(n)``.
        * ``log``      — ``start``, ``stop``, ``n``: log-spaced from ``start`` to ``stop``.

    Returns
    -------
    np.ndarray
        1-D float array of concentrations in M.

    Raises
    ------
    ValueError
        On an unknown mode, ``n < 2``, ``stop <= start``, non-positive bounds for
        log spacing, a non-positive step, an empty/non-finite vector, or any
        negative concentration.
    """
    if mode == 'explicit':
        values: Sequence[float] = kwargs['values']
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            raise ValueError('Concentration vector is empty — enter at least one value.')
    elif mode == 'linear':
        start, stop = float(kwargs['start']), float(kwargs['stop'])
        if not stop > start:
            raise ValueError(f'Linear range needs stop > start (got start={start:g}, stop={stop:g}).')
        arr = np.linspace(start, stop, _as_count(kwargs['n']))
    elif mode == 'step':
        start, step = float(kwargs['start']), float(kwargs['step'])
        if not step > 0:
            raise ValueError(f'Step size must be positive (got {step:g}).')
        arr = start + step * np.arange(_as_count(kwargs['n']))
    elif mode == 'log':
        start, stop = float(kwargs['start']), float(kwargs['stop'])
        if not (start > 0 and stop > 0):
            raise ValueError('Log spacing needs start > 0 and stop > 0.')
        if not stop > start:
            raise ValueError(f'Log range needs stop > start (got start={start:g}, stop={stop:g}).')
        arr = np.logspace(np.log10(start), np.log10(stop), _as_count(kwargs['n']))
    else:
        raise ValueError(f'Unknown concentration mode: {mode!r}.')

    if not np.all(np.isfinite(arr)):
        raise ValueError('Concentration vector contains non-finite values.')
    if np.any(arr < 0):
        raise ValueError('Concentrations cannot be negative.')
    return arr.astype(float)


def _build_assay(
    assay_cls: Type[BaseAssay],
    conditions: Mapping[str, Any],
    x_vector: np.ndarray,
) -> BaseAssay:
    """Construct an assay for forward evaluation over *x_vector* (M).

    Mirrors :meth:`MeasurementSet.to_assay`: the titrant grid is ``x_data`` and the
    (unused) ``y_data`` is a zero placeholder of matching shape.  ``conditions`` are
    the same Quantity-valued kwargs ``AssayConfigPanel.current_conditions`` produces
    (plus ``mode`` for DBA).
    """
    x = np.asarray(x_vector, dtype=float)
    return assay_cls(
        x_data=Q_(x, 'M'),
        y_data=Q_(np.zeros_like(x), 'au'),
        **conditions,
    )


def simulate_signal(
    assay_cls: Type[BaseAssay],
    conditions: Mapping[str, Any],
    parameters: Mapping[str, float],
    x_vector: np.ndarray,
) -> np.ndarray:
    """Evaluate an assay's forward model at *x_vector* from explicit parameters.

    Uses the assay's own ``forward_model`` — identical to the fit path — so the
    predicted signal is exactly what a fit with these parameters would reproduce.

    Parameters
    ----------
    assay_cls : Type[BaseAssay]
        Concrete assay class (e.g. ``GDAAssay``).
    conditions : Mapping[str, Any]
        Quantity-valued conditions (``Ka_dye``, ``h0``, ``fixed_conc``, …) plus
        ``mode`` for DBA — as returned by ``AssayConfigPanel.current_conditions``.
    parameters : Mapping[str, float]
        Base-unit parameter values keyed by the assay's ``parameter_keys``.
    x_vector : np.ndarray
        Titrant concentrations (M).

    Returns
    -------
    np.ndarray
        Predicted signal (au) magnitudes, same length as *x_vector*.
    """
    assay = _build_assay(assay_cls, conditions, x_vector)
    y = assay.forward_model(assay.params_from_dict(dict(parameters)))
    return np.asarray(getattr(y, 'magnitude', y), dtype=float)


def simulate_species(
    assay_cls: Type[BaseAssay],
    conditions: Mapping[str, Any],
    parameters: Mapping[str, float],
    x_vector: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate an assay's equilibrium speciation at *x_vector* from explicit parameters.

    Uses the assay's own ``species()`` accessor — the same solve the forward model
    is built from — so the returned per-species concentrations are exactly those
    underlying :func:`simulate_signal` for the same inputs.

    Parameters
    ----------
    assay_cls : Type[BaseAssay]
        Concrete assay class (e.g. ``GDAAssay``).
    conditions : Mapping[str, Any]
        Quantity-valued conditions (plus ``mode`` for DBA), as for
        :func:`simulate_signal`.
    parameters : Mapping[str, float]
        Base-unit parameter values keyed by the assay's ``parameter_keys``.
    x_vector : np.ndarray
        Titrant concentrations (M).

    Returns
    -------
    dict[str, np.ndarray]
        Species name → concentration array (M magnitudes), insertion order giving
        the intended plotting order; NaN where the equilibrium solve fails.
    """
    assay = _build_assay(assay_cls, conditions, x_vector)
    return assay.species(assay.params_from_dict(dict(parameters)))


def _signal_span(y: np.ndarray) -> float:
    """Return a positive scale for noise: the signal's peak-to-peak range.

    Falls back to the peak magnitude (then to 1.0) for flat curves so the noise
    scale is never zero.
    """
    span = float(np.max(y) - np.min(y))
    if span > 0:
        return span
    mag = float(np.max(np.abs(y)))
    return mag if mag > 0 else 1.0


def simulate_dataset(
    assay_cls: Type[BaseAssay],
    conditions: Mapping[str, Any],
    parameters: Mapping[str, float],
    x_vector: np.ndarray,
    *,
    noise_frac: float = 0.0,
    n_replicas: int = 1,
    rng: np.random.Generator | None = None,
) -> MeasurementSet:
    """Simulate a multi-replica :class:`MeasurementSet` from explicit parameters.

    The clean signal is sampled at *x_vector*; when ``noise_frac > 0`` each replica
    gets independent Gaussian noise with standard deviation
    ``noise_frac × (signal peak-to-peak range)`` — a scale-invariant fraction, so
    the same ``noise_frac`` is meaningful across assays with very different signal
    magnitudes.

    Parameters
    ----------
    noise_frac : float
        Finite, nonnegative noise standard deviation as a fraction of the signal range (0 = clean).
    n_replicas : int
        Positive integer number of replicate rows (>= 1).
    rng : np.random.Generator, optional
        Source of randomness; defaults to ``np.random.default_rng()``.

    Returns
    -------
    MeasurementSet
        Loadable/fittable via the existing pipeline.  ``metadata`` records the
        ground-truth parameters/conditions for provenance (note: the measurement
        writers do not persist metadata — the settings JSON is the durable record).
    """
    if not isinstance(noise_frac, Real) or not np.isfinite(noise_frac) or noise_frac < 0:
        raise ValueError('noise_frac must be finite and nonnegative')
    if isinstance(n_replicas, (bool, np.bool_)) or not isinstance(n_replicas, Integral) or n_replicas < 1:
        raise ValueError('n_replicas must be a positive integer')
    n_replicas = int(n_replicas)

    assay = _build_assay(assay_cls, conditions, x_vector)
    y = assay.forward_model(assay.params_from_dict(dict(parameters)))
    y_clean = np.asarray(getattr(y, 'magnitude', y), dtype=float)

    x = np.asarray(x_vector, dtype=float)
    if noise_frac and noise_frac > 0:
        if rng is None:
            rng = np.random.default_rng()
        sd = float(noise_frac) * _signal_span(y_clean)
        signals = y_clean[None, :] + rng.normal(0.0, sd, size=(n_replicas, x.size))
    else:
        signals = np.tile(y_clean, (n_replicas, 1))

    cond_plain = {k: (float(v.magnitude) if hasattr(v, 'magnitude') else v) for k, v in assay.get_conditions().items()}
    metadata = {
        'source_file': 'simulation',
        'assay_type': assay.assay_type.name,
        'simulation': {
            'parameters': dict(parameters),
            'conditions': cond_plain,
            'noise_frac': float(noise_frac),
        },
    }
    return MeasurementSet(
        concentrations=x,
        signals=signals,
        replica_ids=tuple(f'sim{i + 1}' for i in range(n_replicas)),
        metadata=metadata,
    )
