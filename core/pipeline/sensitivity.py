"""Ka input-sensitivity analysis (pure compute layer, no Qt).

A fitted association constant (Ka) is only as trustworthy as the *inputs* fed
into the model — the titrant concentration vector, the fixed conditions
(``h0`` / ``d0`` / …) and any non-fitted binding constant (``Ka_dye``). Each
carries measurement error, and the fit silently compensates by shifting the
fitted Ka. This module re-runs the fit many times with those inputs perturbed
within a user-defined ``±%`` interval and reports the resulting spread of Ka.

Three analysis modes:

- **JOINT** — every selected input perturbed together each sample → one Ka
  histogram (overall sensitivity to combined concentration error).
- **OAT** — each selected input perturbed alone → one histogram per input
  (ranks which input Ka is most sensitive to).
- **HEATMAP** — sweep two chosen input offsets on a grid, fit at each cell →
  a 2D Ka surface.

Every perturbed fit runs on the **averaged** signal so exactly one Ka comes out
per perturbation, isolating input sensitivity from replica/optimizer scatter.

Layering: this module must not import from ``gui.*``. Perturbable inputs are
classified by pint dimensionality of the passed condition Quantities
(``M`` → concentration, ``1/M`` → binding constant); pretty display labels are
derived in the GUI layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional

import numpy as np

from core.assays.registry import ASSAY_REGISTRY, AssayType
from core.pipeline.fit_pipeline import FitConfig, fit_assay
from core.units import Q_, Quantity

# Sentinel input key for the titrant (the varying x-axis concentration vector).
TITRANT = '__titrant__'


class SensitivityMode(Enum):
    """Analysis mode for a sensitivity run."""

    JOINT = 'joint'
    OAT = 'oat'
    HEATMAP = 'heatmap'


@dataclass(frozen=True)
class SensitivityConfig:
    """User-facing configuration for a sensitivity run.

    Attributes
    ----------
    mode : SensitivityMode
        Which analysis to run.
    n_samples : int
        Number of random draws per histogram (JOINT/OAT). Unused by HEATMAP.
    delta_pct : dict[str, float]
        Input key (``TITRANT`` or a condition key) → ``±`` percent bound. A
        value of ``0`` (or an absent key) holds that input fixed.
    seed : int or None
        Seed for :func:`numpy.random.default_rng`. ``None`` is
        nondeterministic.
    x_key : str or None
        HEATMAP only — input key swept along the X axis.
    y_key : str or None
        HEATMAP only — input key swept along the Y axis.
    n_steps : int
        HEATMAP only — grid resolution per axis (odd includes the 0 offset).
    """

    mode: SensitivityMode
    n_samples: int
    delta_pct: dict[str, float]
    seed: Optional[int] = None
    x_key: Optional[str] = None
    y_key: Optional[str] = None
    n_steps: int = 11


@dataclass
class SensitivityResult:
    """Outcome of a sensitivity run.

    Attributes
    ----------
    mode : SensitivityMode
        The mode that produced this result.
    assay_type : str
        ``AssayType.name`` of the fitted assay.
    ka_keys : tuple[str, ...]
        Association-constant parameter keys (registry ``log_scale_keys``).
    baseline_ka : dict[str, float]
        Unperturbed representative Ka per key, in ``1/M``.
    histograms : dict[str, dict[str, np.ndarray]] or None
        JOINT/OAT only. Label (``'joint'`` or an input key) → ka_key → array
        of collected Ka samples (``1/M``). ``None`` for HEATMAP.
    x_offsets_pct : np.ndarray or None
        HEATMAP only — X-axis offset grid, in percent.
    y_offsets_pct : np.ndarray or None
        HEATMAP only — Y-axis offset grid, in percent.
    ka_grid : dict[str, np.ndarray] or None
        HEATMAP only — ka_key → 2D array ``[iy, ix]`` in ``1/M``; ``NaN``
        where the fit failed.
    n_success : int
        Perturbed fits that converged.
    n_total : int
        Perturbed fits attempted (excludes the baseline).
    config : SensitivityConfig
        The configuration that produced this result.
    """

    mode: SensitivityMode
    assay_type: str
    ka_keys: tuple[str, ...]
    baseline_ka: dict[str, float]
    histograms: Optional[dict[str, dict[str, np.ndarray]]]
    x_offsets_pct: Optional[np.ndarray]
    y_offsets_pct: Optional[np.ndarray]
    ka_grid: Optional[dict[str, np.ndarray]]
    n_success: int
    n_total: int
    config: SensitivityConfig


def perturbable_inputs(assay_type: AssayType, base_conditions: dict) -> list[tuple[str, str]]:
    """List the inputs that can be perturbed for an assay.

    Parameters
    ----------
    assay_type : AssayType
        The assay type (kept for API symmetry; classification is by
        dimensionality of the condition values).
    base_conditions : dict
        Assay conditions as passed to the assay constructor. Values that are
        :class:`pint.Quantity` are classified by dimensionality; non-Quantity
        values (e.g. the ``mode`` string) are skipped.

    Returns
    -------
    list[tuple[str, str]]
        Ordered ``(key, kind)`` pairs. ``TITRANT`` comes first with kind
        ``'titrant'``; then each Quantity condition as ``'concentration'``
        (``M``) or ``'binding_constant'`` (``1/M``).
    """
    inputs: list[tuple[str, str]] = [(TITRANT, 'titrant')]
    for key, value in base_conditions.items():
        if not isinstance(value, Quantity):
            continue
        if value.is_compatible_with('M'):
            inputs.append((key, 'concentration'))
        elif value.is_compatible_with('1/M'):
            inputs.append((key, 'binding_constant'))
    return inputs


def estimated_fit_count(sens_config: SensitivityConfig) -> int:
    """Number of perturbed fits a config will run.

    Excludes the baseline fit and the ``n_trials`` multiplier inside each fit.

    Parameters
    ----------
    sens_config : SensitivityConfig
        The configuration to size.

    Returns
    -------
    int
        JOINT → ``n_samples``; OAT → ``n_samples`` × (#keys with ``delta > 0``);
        HEATMAP → ``n_steps ** 2``.
    """
    if sens_config.mode is SensitivityMode.JOINT:
        return sens_config.n_samples
    if sens_config.mode is SensitivityMode.OAT:
        n_active = sum(1 for v in sens_config.delta_pct.values() if v > 0)
        return sens_config.n_samples * n_active
    return sens_config.n_steps**2


def _build_assay(assay_cls, x_base: np.ndarray, y_avg: np.ndarray, base_conditions: dict, offsets_pct: dict):
    """Build a perturbed assay directly (mirrors ``MeasurementSet.to_assay``).

    No ``MeasurementSet`` mutation, so this is thread-safe. ``Quantity × scalar``
    scales a condition in place for both ``M`` and ``1/M`` alike.
    """
    x = x_base * (1 + offsets_pct.get(TITRANT, 0) / 100)
    conds = {k: (v * (1 + offsets_pct[k] / 100) if k in offsets_pct else v) for k, v in base_conditions.items()}
    return assay_cls(x_data=Q_(x, 'M'), y_data=Q_(y_avg, 'au'), **conds)


def _fit_one(assay_cls, x_base, y_avg, base_conditions, offsets_pct, fit_config, ka_keys):
    """Fit one perturbation → ``{ka_key: value_in_1/M}`` or ``None`` on failure."""
    assay = _build_assay(assay_cls, x_base, y_avg, base_conditions, offsets_pct)
    result = fit_assay(assay, replace(fit_config, per_replica=False))
    if not result.success:
        return None
    return {k: result.parameters[k].to('1/M').magnitude for k in ka_keys}


def run_sensitivity(
    ms,
    assay_cls,
    base_conditions: dict,
    fit_config: FitConfig,
    sens_config: SensitivityConfig,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> SensitivityResult:
    """Run a Ka input-sensitivity analysis.

    Parameters
    ----------
    ms : MeasurementSet
        Source data; its averaged active signal and concentration grid are
        pulled once up front.
    assay_cls : type[BaseAssay]
        Concrete assay class to build each perturbed fit with.
    base_conditions : dict
        Baseline assay conditions (Quantity values in base units).
    fit_config : FitConfig
        Per-fit configuration; each fit runs with ``per_replica=False``.
    sens_config : SensitivityConfig
        What to perturb and how.
    progress : callable, optional
        Called ``progress(done, total)`` after every fit (baseline included).
    should_cancel : callable, optional
        Polled in the perturbation loop; on ``True`` the run stops early and
        returns a partial result.

    Returns
    -------
    SensitivityResult
        Populated per ``sens_config.mode``.

    Raises
    ------
    ValueError
        For an invalid config (``n_samples < 1``, no positive ``delta_pct``,
        negative percent, unknown key, HEATMAP with ``x_key == y_key`` or a
        non-perturbable axis key), an assay with no Ka keys (DYE_ALONE), or a
        baseline fit that does not converge.
    """
    y_avg = ms.average_signal(active_only=True)
    x_base = np.asarray(ms.concentrations)

    # Derive the assay type / Ka keys from a baseline assay instance (no fit).
    baseline_assay = _build_assay(assay_cls, x_base, y_avg, base_conditions, {})
    assay_type: AssayType = baseline_assay.assay_type
    ka_keys = ASSAY_REGISTRY[assay_type].log_scale_keys
    if not ka_keys:
        raise ValueError('This assay has no association constant to analyse, so sensitivity analysis does not apply.')

    valid_keys = {k for k, _ in perturbable_inputs(assay_type, base_conditions)}

    # --- Validate the config before doing any real work ------------------
    if sens_config.n_samples < 1:
        raise ValueError('Number of samples must be at least 1.')
    for key, pct in sens_config.delta_pct.items():
        if key not in valid_keys:
            raise ValueError(f'Unknown sensitivity input key: {key!r}.')
        if pct < 0:
            raise ValueError(f'Percentage for {key!r} must not be negative.')
    if not any(pct > 0 for pct in sens_config.delta_pct.values()):
        raise ValueError('No input is set to vary — give at least one input a positive ± percentage.')

    if sens_config.mode is SensitivityMode.HEATMAP:
        for axis, axis_key in (('X', sens_config.x_key), ('Y', sens_config.y_key)):
            if axis_key not in valid_keys:
                raise ValueError(f'Heatmap {axis} axis input {axis_key!r} is not a perturbable input for this assay.')
        if sens_config.x_key == sens_config.y_key:
            raise ValueError('Heatmap X and Y axes must be two different inputs.')
        if sens_config.n_steps < 1:
            raise ValueError('Heatmap grid resolution must be at least 1.')

    # --- Baseline (unperturbed) fit --------------------------------------
    total = estimated_fit_count(sens_config) + 1  # +1 for the baseline
    done = 0
    baseline_result = fit_assay(baseline_assay, replace(fit_config, per_replica=False))
    done += 1
    if progress is not None:
        progress(done, total)
    if not baseline_result.success:
        raise ValueError(
            'The unperturbed (baseline) fit did not converge, so there is no reference Ka to compare against. '
            'Check the data, conditions, and bounds, and confirm a normal fit succeeds first.'
        )
    baseline_ka = {k: baseline_result.parameters[k].to('1/M').magnitude for k in ka_keys}

    rng = np.random.default_rng(sens_config.seed)
    n_success = 0
    n_total = 0

    def _cancelled() -> bool:
        return should_cancel is not None and should_cancel()

    # --- Dispatch on mode ------------------------------------------------
    if sens_config.mode in (SensitivityMode.JOINT, SensitivityMode.OAT):
        active_keys = [k for k, v in sens_config.delta_pct.items() if v > 0]
        histograms: dict[str, dict[str, np.ndarray]] = {}

        if sens_config.mode is SensitivityMode.JOINT:
            label_to_keys = {'joint': active_keys}
        else:  # OAT — one histogram per active key
            label_to_keys = {k: [k] for k in active_keys}

        for label, keys in label_to_keys.items():
            samples: dict[str, list[float]] = {k: [] for k in ka_keys}
            for _ in range(sens_config.n_samples):
                if _cancelled():
                    break
                offsets = {k: float(rng.uniform(-sens_config.delta_pct[k], sens_config.delta_pct[k])) for k in keys}
                ka = _fit_one(assay_cls, x_base, y_avg, base_conditions, offsets, fit_config, ka_keys)
                n_total += 1
                done += 1
                if progress is not None:
                    progress(done, total)
                if ka is None:
                    continue
                n_success += 1
                for k in ka_keys:
                    samples[k].append(ka[k])
            histograms[label] = {k: np.asarray(samples[k], dtype=float) for k in ka_keys}
            if _cancelled():
                break

        return SensitivityResult(
            mode=sens_config.mode,
            assay_type=assay_type.name,
            ka_keys=ka_keys,
            baseline_ka=baseline_ka,
            histograms=histograms,
            x_offsets_pct=None,
            y_offsets_pct=None,
            ka_grid=None,
            n_success=n_success,
            n_total=n_total,
            config=sens_config,
        )

    # --- HEATMAP ---------------------------------------------------------
    dx = sens_config.delta_pct.get(sens_config.x_key, 0.0)
    dy = sens_config.delta_pct.get(sens_config.y_key, 0.0)
    x_offsets = np.linspace(-dx, dx, sens_config.n_steps)
    y_offsets = np.linspace(-dy, dy, sens_config.n_steps)
    ka_grid = {k: np.full((sens_config.n_steps, sens_config.n_steps), np.nan) for k in ka_keys}

    for iy, oy in enumerate(y_offsets):
        for ix, ox in enumerate(x_offsets):
            if _cancelled():
                break
            offsets = {sens_config.x_key: float(ox), sens_config.y_key: float(oy)}
            ka = _fit_one(assay_cls, x_base, y_avg, base_conditions, offsets, fit_config, ka_keys)
            n_total += 1
            done += 1
            if progress is not None:
                progress(done, total)
            if ka is None:
                continue
            n_success += 1
            for k in ka_keys:
                ka_grid[k][iy, ix] = ka[k]
        if _cancelled():
            break

    return SensitivityResult(
        mode=sens_config.mode,
        assay_type=assay_type.name,
        ka_keys=ka_keys,
        baseline_ka=baseline_ka,
        histograms=None,
        x_offsets_pct=x_offsets,
        y_offsets_pct=y_offsets,
        ka_grid=ka_grid,
        n_success=n_success,
        n_total=n_total,
        config=sens_config,
    )
