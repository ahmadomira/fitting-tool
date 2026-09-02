"""Fitting pipeline orchestration.

This module provides the main entry point for fitting assay data. The
pipeline orchestrates:
1. Loading/preparing data
2. Running multi-start optimization
3. Filtering to the valid-fit pool by quality metrics
4. Collapsing the pool to a representative real fit (see
   :mod:`core.optimizer.ensemble`)
5. Summarising the pool for reporting (see :func:`summarize_parameters`)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from core.assays.base import BaseAssay
from core.assays.dye_alone import DyeAloneAssay
from core.data_processing.measurement_set import MeasurementSet
from core.optimizer.ensemble import collapse, describe, describe_log10
from core.optimizer.filters import calculate_fit_metrics, select_valid_fits
from core.optimizer.multistart import multistart_minimize
from core.optimizer.scaling import ParamScaler
from core.units import Q_, Quantity

logger = logging.getLogger(__name__)

# The fit uses the measured concentrations; the plotted/exported curve is
# evaluated on this denser grid so the line is smooth between data points.
_FIT_CURVE_POINTS = 300


def _dense_fit_curve(assay: BaseAssay, params: np.ndarray) -> tuple[Quantity, Quantity]:
    """Evaluate the forward model on a dense grid spanning the data range.

    Same parameters and conditions as the fit — only the sampling resolution of
    the displayed curve changes (no effect on the fit, parameters, or metrics).
    """
    x_mag = assay.x_data.magnitude
    x_dense = np.linspace(float(x_mag.min()), float(x_mag.max()), _FIT_CURVE_POINTS)
    return Q_(x_dense, 'M'), assay.forward_model(params, x=x_dense)


@dataclass
class FitResult:
    """Serializable container for fitting results.

    All fitted parameter values are stored as ``pint.Quantity`` objects with
    proper units. Spread is not stored: every reported interval is derived
    on demand from ``parameter_samples`` by :func:`summarize_parameters`.

    Attributes
    ----------
    parameters : dict[str, Quantity]
        Representative-fit parameter name → Quantity value. This is an
        *actual* fit from the valid pool (the one with the highest R²),
        not a synthetic per-parameter aggregate — so it lies on the
        model's degenerate manifold and reconstructs ``y_fit`` exactly.
    rmse : float
        RMSE of the representative fit.
    r_squared : float
        R² of the representative fit.
    n_passing : int
        Number of fits in the valid pool.
    n_total : int
        Total number of fit attempts.
    x_fit : Quantity
        Dense concentration grid spanning the data range, for plotting a
        smooth fit curve (not the fitting grid).
    y_fit : Quantity
        Model prediction at ``x_fit`` (the smooth curve).
    assay_type : str
        Assay type name (e.g. ``"GDA"``, ``"IDA"``).
    model_name : str
        Model identifier (e.g. ``"equilibrium_4param"``, ``"linear"``).
    conditions : dict[str, Any]
        Assay conditions used (Quantity values for ``Ka_dye``, ``h0``, etc.).
    fit_config : dict[str, Any]
        Snapshot of the fitting configuration.
    measurement_set_id : str | None
        UUID of the source MeasurementSet (``None`` when fitted directly).
    source_file : str | None
        Original filename, if known.
    id : str
        Auto-generated UUID (hex).
    timestamp : str
        ISO-8601 creation timestamp.
    metadata : dict[str, Any]
        Additional metadata about the fit.
    uncertainty_source : str
        What the pool in ``parameter_samples`` spans. ``"optimizer"``
        (default): the multistart passing trials on a single signal.
        ``"replicate"``: the passing trials pooled across every active
        replica (populated by :func:`fit_measurement_set_per_replica`). The
        string literal ``"replicate"`` is the on-disk value in exported
        fit-result JSON, so it is part of that file format.
    replica_fits : list[FitResult] | None
        Per-replica fit results when this is an aggregate from
        :func:`fit_measurement_set_per_replica`; ``None`` for single-signal
        fits.  Retained for diagnostic inspection only.
    parameter_samples : dict[str, np.ndarray] | None
        One flat array per parameter key holding every valid trial's
        parameter value (length == ``n_passing``), aligned by index.  In
        average mode this is the multistart pool on the single averaged
        signal; in per-replica mode it is the concatenation of valid
        trials across every active replica.  Every reported spread — the
        annotation's range, the table's columns, the exports — is computed
        from this pool via :func:`summarize_parameters`.
    quality_samples : dict[str, np.ndarray] | None
        ``{"rmse": ..., "r_squared": ...}`` — per-trial fit quality for the
        pool, aligned to ``parameter_samples`` by index.  Drives the
        RMSE/R² distribution plots and representative selection.
    representative_index : int | None
        Index (into the ``parameter_samples``/``quality_samples`` arrays)
        of the representative fit reported in ``parameters``/``rmse``/
        ``r_squared``/``y_fit``.
    """

    parameters: Dict[str, Quantity]
    rmse: float
    r_squared: float
    n_passing: int
    n_total: int
    x_fit: Quantity
    y_fit: Quantity
    assay_type: str
    model_name: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    fit_config: Dict[str, Any] = field(default_factory=dict)
    measurement_set_id: Optional[str] = None
    source_file: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    uncertainty_source: str = 'optimizer'
    replica_fits: Optional[List['FitResult']] = None
    parameter_samples: Optional[Dict[str, np.ndarray]] = None
    quality_samples: Optional[Dict[str, np.ndarray]] = None
    representative_index: Optional[int] = None

    @property
    def success(self) -> bool:
        """Whether the fit was successful (at least one passing fit)."""
        return self.n_passing > 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-safe dictionary.

        ``Quantity`` fields are converted to ``{value, unit}`` pairs.
        ``np.ndarray`` / Quantity array fields are converted to plain
        Python lists.

        Returns
        -------
        dict[str, Any]
        """
        # Serialize parameter magnitudes, emitting each unit token from the
        # Quantity itself (not a registry lookup) so the JSON is self-describing
        # and correct even when assay_type is unknown to the registry — the
        # parameters already carry authoritative units.
        params_serial = {}
        units = {}
        for k, v in self.parameters.items():
            params_serial[k] = float(v.magnitude)
            units[k] = str(v.units)

        # Serialize conditions, keeping each Quantity's unit token so the
        # conditions survive a round-trip as Quantities (not bare floats).
        cond_serial = {}
        cond_units = {}
        for k, v in self.conditions.items():
            if isinstance(v, Quantity):
                cond_serial[k] = float(v.magnitude)
                cond_units[k] = str(v.units)
            else:
                cond_serial[k] = v

        # Serialize x_fit / y_fit
        x_list = self.x_fit.magnitude.tolist()
        y_list = self.y_fit.magnitude.tolist()

        replica_fits_serial: Optional[List[Dict[str, Any]]] = None
        if self.replica_fits is not None:
            replica_fits_serial = [rf.to_dict() for rf in self.replica_fits]

        parameter_samples_serial: Optional[Dict[str, List[float]]] = None
        if self.parameter_samples is not None:
            parameter_samples_serial = {k: [float(v) for v in arr] for k, arr in self.parameter_samples.items()}

        quality_samples_serial: Optional[Dict[str, List[float]]] = None
        if self.quality_samples is not None:
            quality_samples_serial = {k: [float(v) for v in arr] for k, arr in self.quality_samples.items()}

        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'measurement_set_id': self.measurement_set_id,
            'source_file': self.source_file,
            'assay_type': self.assay_type,
            'model_name': self.model_name,
            'conditions': cond_serial,
            'fit_config': self.fit_config,
            'parameters': params_serial,
            'parameter_units': units,
            'condition_units': cond_units,
            'rmse': self.rmse,
            'r_squared': self.r_squared,
            'n_passing': self.n_passing,
            'n_total': self.n_total,
            'x_fit': x_list,
            'y_fit': y_list,
            'x_fit_unit': str(self.x_fit.units),
            'y_fit_unit': str(self.y_fit.units),
            'metadata': self.metadata,
            'uncertainty_source': self.uncertainty_source,
            'replica_fits': replica_fits_serial,
            'parameter_samples': parameter_samples_serial,
            'quality_samples': quality_samples_serial,
            'representative_index': self.representative_index,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FitResult':
        """Reconstruct a ``FitResult`` from a dictionary.

        Parameters
        ----------
        d : dict
            As produced by :meth:`to_dict`.

        Returns
        -------
        FitResult
        """
        from core.assays.registry import ASSAY_REGISTRY, AssayType

        # Look up units from the file's own tokens first, then the registry.
        param_units = d.get('parameter_units') or {}
        if not param_units:
            try:
                at = AssayType[d['assay_type']]
                param_units = dict(ASSAY_REGISTRY[at].units)
            except KeyError:
                param_units = {}

        def _unit_for(key: str) -> str:
            unit = param_units.get(key)
            if unit is None:
                # Fail loud rather than silently loading a real unit (e.g. 1/M)
                # as dimensionless, which would drop it from the result with no
                # signal. The GUI import path turns this into a user warning.
                raise ValueError(
                    f"FitResult.from_dict: no unit for parameter '{key}' "
                    f'(assay_type={d.get("assay_type")!r}). The file predates '
                    'parameter_units or uses an unknown assay type, so it cannot '
                    'be loaded without corrupting units.'
                )
            return unit

        # Reconstruct Quantity parameters
        parameters = {k: Q_(v, _unit_for(k)) for k, v in d['parameters'].items()}

        # Reconstruct x_fit / y_fit as Quantities from their own stored unit
        # tokens (older files without them fall back to the M/au convention).
        x_fit = Q_(np.asarray(d['x_fit']), d.get('x_fit_unit', 'M'))
        y_fit = Q_(np.asarray(d['y_fit']), d.get('y_fit_unit', 'au'))

        replica_fits_data = d.get('replica_fits')
        replica_fits: Optional[List['FitResult']] = None
        if replica_fits_data is not None:
            replica_fits = [cls.from_dict(rf) for rf in replica_fits_data]

        parameter_samples_data = d.get('parameter_samples')
        parameter_samples: Optional[Dict[str, np.ndarray]] = None
        if parameter_samples_data is not None:
            parameter_samples = {k: np.asarray(v, dtype=float) for k, v in parameter_samples_data.items()}

        quality_samples_data = d.get('quality_samples')
        quality_samples: Optional[Dict[str, np.ndarray]] = None
        if quality_samples_data is not None:
            quality_samples = {k: np.asarray(v, dtype=float) for k, v in quality_samples_data.items()}

        # Normalise the pool index so a malformed/legacy JSON can't crash the
        # GUI later on an out-of-range lookup.
        representative_index = d.get('representative_index')
        if representative_index is not None:
            pool_size = len(next(iter(parameter_samples.values()))) if parameter_samples else 0
            if not 0 <= representative_index < pool_size:
                representative_index = None

        # Re-wrap conditions that carried a unit back into Quantities (older
        # files without condition_units load as bare floats, as before).
        condition_units = d.get('condition_units', {})
        conditions = {}
        for k, v in d.get('conditions', {}).items():
            unit = condition_units.get(k)
            conditions[k] = Q_(v, unit) if unit is not None else v

        return cls(
            parameters=parameters,
            rmse=d['rmse'],
            r_squared=d['r_squared'],
            n_passing=d['n_passing'],
            n_total=d['n_total'],
            x_fit=x_fit,
            y_fit=y_fit,
            assay_type=d['assay_type'],
            model_name=d['model_name'],
            conditions=conditions,
            fit_config=d.get('fit_config', {}),
            measurement_set_id=d.get('measurement_set_id'),
            source_file=d.get('source_file'),
            id=d.get('id', uuid.uuid4().hex),
            timestamp=d.get('timestamp', datetime.now(timezone.utc).isoformat()),
            metadata=d.get('metadata', {}),
            uncertainty_source=d.get('uncertainty_source', 'optimizer'),
            replica_fits=replica_fits,
            parameter_samples=parameter_samples,
            quality_samples=quality_samples,
            representative_index=representative_index,
        )


@dataclass
class FitConfig:
    """Configuration for the fitting pipeline.

    ``rmse_threshold_factor`` is an *optional* extra trim on top of the
    absolute R² floor (``min_r_squared``), and is off by default
    (``None``). When set, fits with ``RMSE > best_valid_RMSE * factor`` are
    also dropped from the valid pool.
    """

    n_trials: int = 100
    rmse_threshold_factor: Optional[float] = None
    min_r_squared: float = 0.9
    log_scale_params: Optional[List[str]] = None
    custom_bounds: Optional[Dict[str, Tuple[Quantity, Quantity]]] = None
    rescale_parameters: bool = True
    per_replica: bool = True


def _model_name_for_assay(assay: BaseAssay) -> str:
    """Derive a model name string from an assay instance.

    Each assay family declares its own ``model_name`` class attribute; the
    base default (``'equilibrium_4param'``) covers the 1:1 models.
    """
    return assay.model_name


def _config_to_dict(config: FitConfig) -> Dict[str, Any]:
    """Serialize a FitConfig to a plain dict."""
    # Provenance only (never deserialized back): keep the unit token so a bound
    # entered in µM / MM⁻¹ is not stored as a unit-ambiguous bare number.
    custom_bounds: Optional[Dict[str, list]] = None
    if config.custom_bounds is not None:
        custom_bounds = {}
        for key, (lo, hi) in config.custom_bounds.items():
            custom_bounds[key] = [float(lo.magnitude), float(hi.magnitude), str(lo.units)]
    return {
        'n_trials': config.n_trials,
        'rmse_threshold_factor': config.rmse_threshold_factor,
        'min_r_squared': config.min_r_squared,
        'log_scale_params': config.log_scale_params,
        'custom_bounds': custom_bounds,
        'rescale_parameters': config.rescale_parameters,
        # JSON key kept as 'per_replicate' for backward compat with
        # previously exported fit-result files.
        'per_replicate': config.per_replica,
    }


def _resolve_bounds(
    assay: BaseAssay,
    custom_bounds: Optional[Dict[str, Tuple[Quantity, Quantity]]],
) -> Dict[str, Tuple[Quantity, Quantity]]:
    """Merge user overrides with registry defaults -> named bounds dict."""
    bounds_dict = assay.get_default_bounds()
    if custom_bounds is not None:
        unknown = set(custom_bounds) - set(assay.parameter_keys)
        if unknown:
            raise ValueError(
                f'Unknown parameter(s) in custom_bounds: {sorted(unknown)}. Valid keys: {list(assay.parameter_keys)}'
            )
        bounds_dict.update(custom_bounds)
    return bounds_dict


def _resolve_log_scale(
    assay: BaseAssay,
    log_scale_params: Optional[List[str]],
) -> List[int]:
    """Convert parameter names -> positional indices for the optimizer."""
    if log_scale_params is None:
        names = list(assay.registry_metadata.log_scale_keys)
    else:
        names = list(log_scale_params)

    keys = assay.parameter_keys
    indices: List[int] = []
    for name in names:
        if name not in keys:
            raise ValueError(f"Unknown log-scale parameter '{name}'. Valid keys: {list(keys)}")
        indices.append(keys.index(name))
    return indices


def _wrap_params_as_quantities(
    params: np.ndarray,
    assay: BaseAssay,
) -> Dict[str, Quantity]:
    """Wrap optimizer float params into named Quantity dict."""
    from core.assays.registry import ASSAY_REGISTRY

    units = ASSAY_REGISTRY[assay.assay_type].units
    result = {}
    for key, value in zip(assay.parameter_keys, params):
        unit = units.get(key)
        if unit is None:
            raise KeyError(
                f"No registry unit declared for parameter '{key}' of {assay.assay_type.name}; "
                'cannot attach units to the fit result.'
            )
        result[key] = Q_(float(value), unit)
    return result


def representative_view(
    assay: BaseAssay,
    params: np.ndarray,
) -> Tuple[float, float, Quantity, Quantity]:
    """Self-consistent reported fields for one real parameter vector.

    The single source for "a parameter vector → its RMSE/R² and plotted
    curve against *assay*'s data", used both when first reporting a fit and
    when the user re-selects a different representative. Because *params* is
    a real fit (on the model's degenerate manifold), the curve and metrics
    reconstruct correctly — unlike a synthetic per-parameter aggregate.

    Returns
    -------
    tuple[float, float, Quantity, Quantity]
        ``(rmse, r_squared, x_fit, y_fit)``.
    """
    y_pred = assay.forward_model(params)
    rmse, r_squared = calculate_fit_metrics(assay.y_data.magnitude, y_pred.magnitude)
    x_fit, y_fit = _dense_fit_curve(assay, params)
    return rmse, r_squared, x_fit, y_fit


@dataclass(frozen=True)
class ParameterSummary:
    """One reported row: a fitted parameter, or its log₁₀ twin.

    Attributes
    ----------
    key : str
        Parameter key (e.g. ``"Ka_dye"``). A log twin repeats its source key.
    is_log : bool
        ``True`` for a log₁₀ twin row. ``estimate`` and ``stats`` are then
        statistics of ``log10(parameter)`` and carry no unit — but ``unit``
        still names the underlying parameter's unit, so a label can read
        ``log₁₀(Ka / M⁻¹)``.
    estimate : float
        The representative fit's value for this row.
    unit : str
        Unit of the underlying parameter (``""`` when it has none).
    stats : dict[str, float] | None
        :func:`core.optimizer.ensemble.describe` output over the valid-fit
        pool, or ``None`` when the result carries no pool (dye-alone linear
        fits, and imports of results exported before pools were stored).
    """

    key: str
    is_log: bool
    estimate: float
    unit: str
    stats: Optional[Dict[str, float]]


def summarize_parameters(result: FitResult) -> List[ParameterSummary]:
    """Reported rows for *result* — the single source for every display.

    The fit summary table, the plot annotation and the TXT/CSV exports all
    render from this, so they cannot drift apart. A log₁₀ twin row follows
    each association constant (the assay's ``log_scale_keys``); its statistics
    come from :func:`~core.optimizer.ensemble.describe_log10`, which transforms
    the pool *first* — never ``log10`` of a spread.

    Each unit comes from the parameter's own ``Quantity``, which stays correct
    even when the assay type is missing from the registry; only the choice of
    which parameters are log-scale is registry metadata.
    """
    from core.assays.registry import ASSAY_REGISTRY, AssayType

    try:
        log_keys = set(ASSAY_REGISTRY[AssayType[result.assay_type]].log_scale_keys)
    except KeyError:
        log_keys = set()

    samples = result.parameter_samples or {}
    rows: List[ParameterSummary] = []
    for key, value in result.parameters.items():
        unit = str(value.units) if isinstance(value, Quantity) else ''
        magnitude = float(value.magnitude) if isinstance(value, Quantity) else float(value)
        raw_pool = samples.get(key)
        pool = np.asarray(raw_pool, dtype=float) if raw_pool is not None else None

        rows.append(
            ParameterSummary(
                key=key,
                is_log=False,
                estimate=magnitude,
                unit=unit,
                stats=describe(pool) if pool is not None else None,
            )
        )
        if key in log_keys and pool is not None:
            rows.append(
                ParameterSummary(
                    key=key,
                    is_log=True,
                    estimate=float(np.log10(magnitude)),
                    unit=unit,
                    stats=describe_log10(pool),
                )
            )
    return rows


def select_representative(result: FitResult, assay: BaseAssay, index: int) -> None:
    """Re-point *result* at a different valid fit from its pool.

    Mutates *result* in place: sets ``representative_index`` to *index* and
    rebuilds ``parameters``/``rmse``/``r_squared``/``x_fit``/``y_fit`` from
    that pooled trial via :func:`representative_view`. The pool is unchanged —
    only which real fit is reported. Used by the GUI when the user picks a fit
    from the distribution plot or the selector.

    Parameters
    ----------
    result : FitResult
        A successful fit carrying a ``parameter_samples`` pool.
    assay : BaseAssay
        Assay rebuilt from the same data (the averaged signal for
        per-replica results) so curve/metrics match the reported data.
    index : int
        Index into the pooled samples of the fit to report.
    """
    if not result.parameter_samples:
        raise ValueError('Cannot select a representative: result has no parameter_samples pool.')

    pool_size = len(next(iter(result.parameter_samples.values())))
    if not 0 <= index < pool_size:
        raise ValueError(f'Representative index {index} out of range for pool of size {pool_size}.')

    rep_params = np.array([result.parameter_samples[k][index] for k in assay.parameter_keys], dtype=float)
    rmse, r_squared, x_fit, y_fit = representative_view(assay, rep_params)
    result.representative_index = index
    result.parameters = _wrap_params_as_quantities(rep_params, assay)
    result.rmse = rmse
    result.r_squared = r_squared
    result.x_fit = x_fit
    result.y_fit = y_fit


def fit_assay(
    assay: BaseAssay,
    config: Optional[FitConfig] = None,
    *,
    measurement_set_id: Optional[str] = None,
    source_file: Optional[str] = None,
) -> FitResult:
    """Fit an assay using multi-start optimisation.

    Parameters
    ----------
    assay : BaseAssay
        The assay data to fit (with Quantity x_data/y_data).
    config : FitConfig, optional
        Fitting configuration.
    measurement_set_id : str, optional
        UUID of the source ``MeasurementSet`` for traceability.
    source_file : str, optional
        Original filename, if known.

    Returns
    -------
    FitResult
        With Quantity parameters, x_fit, y_fit, and the valid-fit pool.
    """
    if config is None:
        config = FitConfig()

    from core.assays.registry import ASSAY_REGISTRY

    # Resolve bounds (Quantity dicts)
    bounds_dict = _resolve_bounds(assay, config.custom_bounds)

    # Extract float bounds for scipy, converting each bound to the parameter's
    # canonical unit first. This keeps the Quantity->float boundary unit-safe: a
    # custom bound supplied in a compatible-but-non-canonical unit (e.g. a Ka in
    # 'MM^-1' or a concentration in 'µM') is converted correctly instead of
    # having its face magnitude taken, and a dimensionally wrong bound fails fast.
    canonical_units = ASSAY_REGISTRY[assay.assay_type].units
    float_bounds = [
        (
            float(bounds_dict[k][0].to(canonical_units[k]).magnitude),
            float(bounds_dict[k][1].to(canonical_units[k]).magnitude),
        )
        for k in assay.parameter_keys
    ]

    # Resolve log-scale parameters (names -> indices)
    log_scale = _resolve_log_scale(assay, config.log_scale_params)

    # y_data magnitude for metrics
    y_data_mag = assay.y_data.magnitude

    # Define objective function
    def objective(params: np.ndarray) -> float:
        return assay.sum_squared_residuals(params)

    # Define metrics function
    def compute_metrics(params: np.ndarray) -> Tuple[float, float]:
        y_pred = assay.forward_model(params)
        return calculate_fit_metrics(y_data_mag, y_pred.magnitude)

    scaler: Optional[ParamScaler] = None
    if config.rescale_parameters:
        scaler = ParamScaler.from_assay(assay)

    # Run multi-start optimization
    all_attempts = multistart_minimize(
        objective=objective,
        bounds=float_bounds,
        n_trials=config.n_trials,
        log_scale_params=log_scale,
        compute_metrics=compute_metrics,
        scaler=scaler,
    )

    # Select the valid pool: absolute R² floor (primary) + optional RMSE trim.
    valid_attempts = select_valid_fits(
        all_attempts,
        min_r_squared=config.min_r_squared,
        rmse_threshold_factor=config.rmse_threshold_factor,
    )

    # Handle case where no fits pass
    if not valid_attempts:
        if all_attempts:
            best = all_attempts[0]
            logger.warning(
                'No fits passed filtering (n_trials=%d, min_r2=%.2f, rmse_factor=%s). Best attempt: RMSE=%.4e, R2=%.4f.',
                config.n_trials,
                config.min_r_squared,
                config.rmse_threshold_factor,
                best.rmse,
                best.r_squared,
            )
            hint = (
                'All fit attempts were rejected by the quality filter. Try: (1) lowering min_r_squared, '
                '(2) increasing n_trials, or (3) reviewing parameter bounds.'
            )
            if config.rmse_threshold_factor is not None:
                hint += ' The optional RMSE-factor trim is also active — disabling it may help.'
            diag = {
                'error': 'No fits passed filtering criteria',
                'best_attempt_rmse': best.rmse,
                'best_attempt_r_squared': best.r_squared,
                'best_attempt_params': assay.params_to_dict(best.params),
                'hint': hint,
            }
        else:
            logger.warning(
                'All %d fit attempts failed (optimizer did not converge).',
                config.n_trials,
            )
            diag = {'error': 'All fit attempts failed to converge'}

        return FitResult(
            parameters={},
            rmse=np.inf,
            r_squared=0.0,
            n_passing=0,
            n_total=len(all_attempts) if all_attempts else config.n_trials,
            x_fit=assay.x_data,
            y_fit=Q_(np.full(len(assay.x_data), np.nan), 'au'),
            assay_type=assay.assay_type.name,
            model_name=_model_name_for_assay(assay),
            conditions=assay.get_conditions(),
            fit_config=_config_to_dict(config),
            measurement_set_id=measurement_set_id,
            source_file=source_file,
            metadata=diag,
        )

    # Collapse the pool: keep every valid trial, report a real representative fit.
    keys = list(assay.parameter_keys)
    param_matrix = np.array([a.params for a in valid_attempts], dtype=float)
    rmse_pool = np.array([a.rmse for a in valid_attempts], dtype=float)
    r2_pool = np.array([a.r_squared for a in valid_attempts], dtype=float)
    ens = collapse(param_matrix, rmse_pool, r2_pool, keys)

    # Reported result = the representative real fit (on-manifold, self-consistent).
    rep_params = ens.representative_params
    rmse, r_squared, x_fit, y_fit = representative_view(assay, rep_params)
    params_q = _wrap_params_as_quantities(rep_params, assay)

    return FitResult(
        parameters=params_q,
        rmse=rmse,
        r_squared=r_squared,
        n_passing=len(valid_attempts),
        n_total=len(all_attempts),
        x_fit=x_fit,
        y_fit=y_fit,
        assay_type=assay.assay_type.name,
        model_name=_model_name_for_assay(assay),
        conditions=assay.get_conditions(),
        fit_config=_config_to_dict(config),
        measurement_set_id=measurement_set_id,
        source_file=source_file,
        parameter_samples=ens.parameter_samples,
        quality_samples=ens.quality_samples,
        representative_index=ens.representative_index,
    )


def fit_linear_assay(
    assay: DyeAloneAssay,
    *,
    measurement_set_id: Optional[str] = None,
    source_file: Optional[str] = None,
) -> FitResult:
    """Fit a dye-alone assay using simple linear regression.

    Parameters
    ----------
    assay : DyeAloneAssay
        The dye-alone calibration data.
    measurement_set_id : str, optional
        UUID of the source ``MeasurementSet`` for traceability.
    source_file : str, optional
        Original filename, if known.

    Returns
    -------
    FitResult
    """
    slope, intercept, r_squared, rmse = assay.fit_linear()
    y_fit = assay.forward_model(np.array([slope.magnitude, intercept.magnitude]))

    return FitResult(
        parameters={
            'slope': slope,
            'intercept': intercept,
        },
        rmse=float(rmse.magnitude),
        r_squared=r_squared,
        n_passing=1,
        n_total=1,
        x_fit=assay.x_data,
        y_fit=y_fit,
        assay_type=assay.assay_type.name,
        model_name='linear',
        conditions=assay.get_conditions(),
        fit_config={},
        measurement_set_id=measurement_set_id,
        source_file=source_file,
        metadata={'method': 'linear_regression'},
    )


def bounds_from_dye_alone(
    result: FitResult,
    margin: float = 0.2,
) -> Dict[str, Tuple[Quantity, Quantity]]:
    """Derive signal-coefficient bounds from a dye-alone calibration.

    Returns Quantity bound tuples suitable for ``FitConfig.custom_bounds``.

    Parameters
    ----------
    result : FitResult
        A *successful* dye-alone fit (``model_name == "linear"``).
    margin : float
        Fractional margin (default 0.2 = +/-20%).

    Returns
    -------
    Dict[str, Tuple[Quantity, Quantity]]
    """
    if result.model_name != 'linear':
        raise ValueError(f"Expected a dye-alone (linear) FitResult, got model_name='{result.model_name}'")
    if not result.success:
        raise ValueError('Cannot derive bounds from a failed fit')

    slope_q = result.parameters['slope']
    intercept_q = result.parameters['intercept']

    def _window(value: Quantity, margin_frac: float) -> Tuple[Quantity, Quantity]:
        mag = float(value.magnitude)
        half = abs(mag) * margin_frac if abs(mag) > 0 else margin_frac
        lo = max(0.0, mag - half)
        hi = max(0.0, mag + half)
        return (Q_(lo, value.units), Q_(hi, value.units))

    return {
        'I_dye_free': _window(slope_q, margin),
        'I0': _window(intercept_q, margin),
    }


def fit_measurement_set(
    ms: MeasurementSet,
    assay_cls: Type[BaseAssay],
    conditions: Dict[str, Any],
    config: Optional[FitConfig] = None,
    *,
    use_average: bool = True,
    replica_id: Optional[str] = None,
) -> FitResult:
    """Convenience: build an assay from a MeasurementSet and fit it.

    When ``config.per_replica`` is ``True`` and neither ``use_average=False``
    nor an explicit ``replica_id`` is provided, dispatches to
    :func:`fit_measurement_set_per_replica`, which fits each active replica
    independently and aggregates parameters across replicas.

    Parameters
    ----------
    ms : MeasurementSet
        Source data.
    assay_cls : Type[BaseAssay]
        Concrete assay class.
    conditions : dict
        Assay-specific conditions as Quantity values.
    config : FitConfig, optional
        Fitting configuration.
    use_average : bool
        Use the averaged active-replica signal (default ``True``).
    replica_id : str, optional
        Fit a specific replica instead of the average.

    Returns
    -------
    FitResult
    """
    if config is not None and config.per_replica and replica_id is None and use_average:
        return fit_measurement_set_per_replica(ms, assay_cls, conditions, config)

    assay = ms.to_assay(
        assay_cls,
        conditions=conditions,
        use_average=use_average,
        replica_id=replica_id,
    )

    if isinstance(assay, DyeAloneAssay):
        return fit_linear_assay(
            assay,
            measurement_set_id=ms.id,
            source_file=ms.metadata.get('source_file'),
        )

    return fit_assay(
        assay,
        config=config,
        measurement_set_id=ms.id,
        source_file=ms.metadata.get('source_file'),
    )


class PerReplicaFitError(RuntimeError):
    """Raised when per-replica fitting cannot produce any surviving fit.

    Carries a ``failures`` mapping of ``replica_id -> reason`` so the GUI
    layer can surface a no-silent-fallbacks warning listing every failed
    replica.
    """

    def __init__(self, message: str, failures: Dict[str, str]):
        super().__init__(message)
        self.failures = dict(failures)


def fit_measurement_set_per_replica(
    ms: MeasurementSet,
    assay_cls: Type[BaseAssay],
    conditions: Dict[str, Any],
    config: Optional[FitConfig] = None,
) -> FitResult:
    """Fit every active replica independently and pool passing trials.

    Each active replica is fit with its own multistart run (the same
    ``config`` — including ``rescale_parameters`` — is forwarded unchanged
    to every per-replica call).  Because the parameter rescaler is an
    exact affine bijection [core/optimizer/scaling.py:20-23], every
    per-replica ``FitResult`` emerges in physical units regardless of
    its per-replica scaler, so parameter values from every replica
    live in the same raw-unit space.

    **Aggregation semantics (pooled):** every replica's valid trials are
    concatenated into a single flat pool (per parameter key) stored on the
    returned ``FitResult.parameter_samples``, aligned with a pooled
    ``quality_samples`` whose RMSE/R² are recomputed against the averaged
    active-replica signal.  A replica with more valid trials contributes
    proportionally more samples to the pool.

    The reported ``parameters``/``rmse``/``r_squared``/``y_fit`` come from
    the **representative** pooled trial — the one with the highest R²
    against the averaged signal, a real on-manifold fit, not a synthetic
    per-parameter aggregate.  The pooled trials from every replica are kept
    in ``parameter_samples``, which every reported spread derives from.

    Failure handling: a replica that raises (degenerate scaler input,
    convergence failure, …) is skipped and recorded in the returned
    FitResult's metadata.  If no replica survives, a
    :class:`PerReplicaFitError` is raised with the full failure map so
    the GUI can report every bad replica.

    Parameters
    ----------
    ms : MeasurementSet
        Source data with one or more active replicas.
    assay_cls : Type[BaseAssay]
        Concrete assay class.
    conditions : dict
        Assay-specific conditions as Quantity values.
    config : FitConfig, optional
        Fitting configuration.  ``per_replica`` is ignored (this function
        *is* the per-replica path) but every other field — including
        ``rescale_parameters`` — is forwarded to each per-replica fit.

    Returns
    -------
    FitResult
        Aggregate result whose ``parameters``/``rmse``/``r_squared`` are
        the representative pooled trial,
        ``uncertainty_source == "replicate"`` (string kept for
        backward compat with previously exported JSONs),
        ``parameter_samples``/``quality_samples`` hold the pool, and
        ``replica_fits`` holds the per-replica successful fits for
        diagnostic inspection.
    """
    if config is None:
        config = FitConfig()

    active_ids = ms.active_replica_ids
    if not active_ids:
        raise ValueError('No active replicas to fit.')

    per_call_config = replace(config, per_replica=False)

    replica_fits: List[FitResult] = []
    succeeded_ids: List[str] = []
    failures: Dict[str, str] = {}
    for rid in active_ids:
        try:
            rr = fit_measurement_set(
                ms,
                assay_cls,
                conditions,
                per_call_config,
                use_average=False,
                replica_id=rid,
            )
        except Exception as exc:
            failures[rid] = f'{type(exc).__name__}: {exc}'
            logger.warning('Per-replica fit failed for replica %s: %s', rid, exc)
            continue

        if not rr.success:
            failures[rid] = 'No trials passed the quality filter.'
            continue

        if rr.parameter_samples is None:
            failures[rid] = 'Per-replica fit returned no parameter_samples pool to aggregate.'
            continue

        rr.metadata.setdefault('replica_id', rid)
        replica_fits.append(rr)
        succeeded_ids.append(rid)

    if not replica_fits:
        raise PerReplicaFitError(
            f'Per-replica fitting failed on all {len(active_ids)} active replica(s).',
            failures,
        )

    # Pool every replica's valid trials (per parameter key), aligned by index.
    param_keys = tuple(replica_fits[0].parameter_samples.keys())
    for rr in replica_fits[1:]:
        if tuple(rr.parameter_samples.keys()) != param_keys:
            raise ValueError(f'Replicate parameter keys differ: {param_keys} vs {tuple(rr.parameter_samples.keys())}.')
    pool: Dict[str, np.ndarray] = {
        k: np.concatenate([rr.parameter_samples[k] for rr in replica_fits]) for k in param_keys
    }
    pool_size = len(next(iter(pool.values())))

    template_assay = ms.to_assay(
        assay_cls,
        conditions=conditions,
        use_average=True,
    )
    keys = list(template_assay.parameter_keys)

    # Recompute each pooled trial's quality against the averaged signal so the
    # pooled representative and its reported RMSE/R² stay self-consistent.
    param_matrix = np.column_stack([pool[k] for k in keys])
    y_avg = template_assay.y_data.magnitude
    rmse_pool = np.empty(pool_size)
    r2_pool = np.empty(pool_size)
    for j in range(pool_size):
        y_pred = template_assay.forward_model(param_matrix[j])
        rmse_pool[j], r2_pool[j] = calculate_fit_metrics(y_avg, y_pred.magnitude)
    ens = collapse(param_matrix, rmse_pool, r2_pool, keys)

    # Reported result = the representative pooled trial (a real, on-manifold fit).
    rep_params = ens.representative_params
    rmse, r_squared, x_fit, y_fit = representative_view(template_assay, rep_params)
    params_q = _wrap_params_as_quantities(rep_params, template_assay)

    n_total_pool = sum(rr.n_total for rr in replica_fits)

    metadata: Dict[str, Any] = {
        'n_replicas_total': len(active_ids),
        'n_replicas_fit': len(replica_fits),
        'replica_ids_fit': list(succeeded_ids),
        'pool_size': pool_size,
        'per_replica_n_passing': {rid: rf.n_passing for rid, rf in zip(succeeded_ids, replica_fits)},
    }
    if failures:
        metadata['replica_failures'] = failures

    return FitResult(
        parameters=params_q,
        rmse=rmse,
        r_squared=r_squared,
        n_passing=pool_size,
        n_total=n_total_pool,
        x_fit=x_fit,
        y_fit=y_fit,
        assay_type=template_assay.assay_type.name,
        model_name=_model_name_for_assay(template_assay),
        conditions=template_assay.get_conditions(),
        fit_config=_config_to_dict(config),
        measurement_set_id=ms.id,
        source_file=ms.metadata.get('source_file'),
        metadata=metadata,
        uncertainty_source='replicate',
        replica_fits=replica_fits,
        parameter_samples=ens.parameter_samples,
        quality_samples=ens.quality_samples,
        representative_index=ens.representative_index,
    )
