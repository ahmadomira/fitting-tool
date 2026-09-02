"""Ka input-sensitivity analysis — scientific behaviour tests.

Verifies the compute layer (`core.pipeline.sensitivity`) as *behaviour*, not
implementation:

- perturbation is the only variance source (near-zero Δ → samples collapse onto
  the baseline Ka);
- larger Δ produces strictly larger Ka spread;
- the titrant and the non-fitted ``Ka_dye`` are both genuinely perturbable;
- the ``seed`` knob makes the perturbation draws reproducible;
- the heatmap centre cell (0 %, 0 %) reproduces the baseline;
- ``estimated_fit_count`` matches the driver's actual per-mode workload;
- invalid configs fail fast with ``ValueError``.

Fits run on clean synthetic IDA data with the tight recovery bounds from
``conftest`` and a small ``n_trials`` — enough to converge reliably while
keeping the many-fit loops fast. Where a test compares two runs, the *global*
RNG (which drives the multi-start optimiser) is re-seeded identically before
each run so that any difference is attributable to the sensitivity ``seed``
alone.
"""

import numpy as np
import pytest

from core.assays.dye_alone import DyeAloneAssay
from core.assays.ida import IDAAssay
from core.data_processing.measurement_set import MeasurementSet
from core.pipeline.fit_pipeline import FitConfig
from core.pipeline.sensitivity import (
    TITRANT,
    SensitivityConfig,
    SensitivityMode,
    estimated_fit_count,
    run_sensitivity,
)
from core.units import Q_
from tests.conftest import (
    GDA_IDA_RECOVERY_BOUNDS,
    IDA_TRUE,
    _make_dye_alone_data,
    _make_ida_data,
)

# Small but reliable: tight recovery bounds mean clean IDA data converges with
# few trials, so the sensitivity loops stay fast.
_N_TRIALS = 12
_N_POINTS = 20


@pytest.fixture
def ida_setup():
    """(ms, assay_cls, conditions, fit_config) for a clean IDA sensitivity run."""
    x, y = _make_ida_data(IDA_TRUE, n_points=_N_POINTS)
    ms = MeasurementSet(concentrations=x, signals=y.reshape(1, -1), replica_ids=('r0',))
    conditions = {
        'Ka_dye': Q_(IDA_TRUE['Ka_dye'], '1/M'),
        'h0': Q_(IDA_TRUE['h0'], 'M'),
        'd0': Q_(IDA_TRUE['d0'], 'M'),
    }
    fit_config = FitConfig(n_trials=_N_TRIALS, custom_bounds=GDA_IDA_RECOVERY_BOUNDS)
    return ms, IDAAssay, conditions, fit_config


def _run(ida_setup, sens_config, *, global_seed=0):
    """Run a sensitivity analysis after pinning the global (optimiser) RNG."""
    ms, assay_cls, conditions, fit_config = ida_setup
    np.random.seed(global_seed)
    return run_sensitivity(ms, assay_cls, conditions, fit_config, sens_config)


# ---------------------------------------------------------------------------
# Perturbation is the only variance source
# ---------------------------------------------------------------------------


def test_negligible_delta_collapses_onto_baseline(ida_setup):
    """A vanishingly small Δ leaves every Ka sample essentially at the baseline.

    Isolates that the *perturbation* is what moves Ka: with inputs unchanged the
    fit reproduces the unperturbed reference to well within 2 %.
    """
    cfg = SensitivityConfig(mode=SensitivityMode.JOINT, n_samples=6, delta_pct={TITRANT: 1e-3}, seed=1)
    result = _run(ida_setup, cfg)

    samples = result.histograms['joint']['Ka_guest']
    baseline = result.baseline_ka['Ka_guest']
    assert samples.size == 6
    # Every sample sits on top of the baseline; spread is negligible.
    assert np.allclose(samples, baseline, rtol=0.02)
    assert np.std(samples) / baseline < 0.02


def test_spread_grows_with_delta(ida_setup):
    """Sample std at ±10 % strictly exceeds ±2 %, which strictly exceeds 0.

    Same seed for both runs so the only difference is the perturbation width.
    """
    small = _run(ida_setup, SensitivityConfig(SensitivityMode.JOINT, 8, {TITRANT: 2.0}, seed=7))
    big = _run(ida_setup, SensitivityConfig(SensitivityMode.JOINT, 8, {TITRANT: 10.0}, seed=7))

    std_small = float(np.std(small.histograms['joint']['Ka_guest']))
    std_big = float(np.std(big.histograms['joint']['Ka_guest']))
    assert std_big > std_small > 0


# ---------------------------------------------------------------------------
# Both input paths (titrant vector and non-fitted Ka_dye) are perturbable
# ---------------------------------------------------------------------------


def test_titrant_only_oat_produces_spread(ida_setup):
    """OAT on the titrant alone yields a non-empty, non-degenerate Ka spread.

    Guards the direct ``x_data`` perturbation path (distinct from conditions).
    """
    cfg = SensitivityConfig(mode=SensitivityMode.OAT, n_samples=8, delta_pct={TITRANT: 8.0}, seed=3)
    result = _run(ida_setup, cfg)

    assert set(result.histograms) == {TITRANT}
    samples = result.histograms[TITRANT]['Ka_guest']
    assert samples.size > 0
    assert np.std(samples) > 0


def test_ka_dye_only_oat_shifts_ka(ida_setup):
    """Perturbing the non-fitted ``Ka_dye`` input alone moves the fitted Ka.

    A known binding-constant input carries its own uncertainty; the fit absorbs
    it into Ka_guest, so an OAT sweep on Ka_dye must produce real spread that
    departs from the baseline.
    """
    cfg = SensitivityConfig(mode=SensitivityMode.OAT, n_samples=8, delta_pct={'Ka_dye': 15.0}, seed=5)
    result = _run(ida_setup, cfg)

    samples = result.histograms['Ka_dye']['Ka_guest']
    baseline = result.baseline_ka['Ka_guest']
    assert samples.size > 0
    assert np.std(samples) > 0
    # At least one perturbed fit lands meaningfully away from the baseline.
    assert np.max(np.abs(samples - baseline)) / baseline > 0.01


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_seed_makes_draws_reproducible(ida_setup):
    """Identical sensitivity seed → identical samples; a different seed differs.

    The global optimiser RNG is pinned the same for all three runs, so only the
    sensitivity ``seed`` (which drives the perturbation draws) can change the
    result.
    """
    cfg_a = SensitivityConfig(SensitivityMode.JOINT, 6, {TITRANT: 5.0}, seed=123)
    cfg_b = SensitivityConfig(SensitivityMode.JOINT, 6, {TITRANT: 5.0}, seed=456)

    r1 = _run(ida_setup, cfg_a)
    r2 = _run(ida_setup, cfg_a)
    r3 = _run(ida_setup, cfg_b)

    s1 = r1.histograms['joint']['Ka_guest']
    s2 = r2.histograms['joint']['Ka_guest']
    s3 = r3.histograms['joint']['Ka_guest']

    np.testing.assert_array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def test_heatmap_center_matches_baseline_and_shape(ida_setup):
    """The (0 %, 0 %) centre cell reproduces the baseline; grid is n_steps².

    Uses a titrant × h0 axis pair to exercise one axis from each perturbation
    path. On clean data every cell converges, so no NaNs appear.
    """
    cfg = SensitivityConfig(
        mode=SensitivityMode.HEATMAP,
        n_samples=1,
        delta_pct={TITRANT: 5.0, 'h0': 5.0},
        seed=0,
        x_key=TITRANT,
        y_key='h0',
        n_steps=3,
    )
    result = _run(ida_setup, cfg)

    grid = result.ka_grid['Ka_guest']
    assert grid.shape == (3, 3)
    # linspace(-d, d, 3)[1] == 0 on both axes → centre is an unperturbed fit.
    center = grid[1, 1]
    assert center == pytest.approx(result.baseline_ka['Ka_guest'], rel=0.03)
    assert result.n_success == 9
    assert result.n_total == 9
    assert np.all(np.isfinite(grid))


# ---------------------------------------------------------------------------
# estimated_fit_count — matches the driver's per-mode workload
# ---------------------------------------------------------------------------


def test_estimated_fit_count_joint():
    cfg = SensitivityConfig(SensitivityMode.JOINT, 200, {TITRANT: 5.0})
    assert estimated_fit_count(cfg) == 200


def test_estimated_fit_count_oat_counts_only_active_keys():
    # Two keys vary (>0), one is held fixed (0) and must not be counted.
    cfg = SensitivityConfig(SensitivityMode.OAT, 50, {TITRANT: 5.0, 'h0': 5.0, 'd0': 0.0})
    assert estimated_fit_count(cfg) == 50 * 2


def test_estimated_fit_count_heatmap_is_grid_squared():
    cfg = SensitivityConfig(
        SensitivityMode.HEATMAP, 200, {TITRANT: 5.0, 'h0': 5.0}, x_key=TITRANT, y_key='h0', n_steps=7
    )
    assert estimated_fit_count(cfg) == 49


# ---------------------------------------------------------------------------
# Fail-fast contracts
# ---------------------------------------------------------------------------


def test_n_samples_below_one_raises(ida_setup):
    ms, assay_cls, conditions, fit_config = ida_setup
    cfg = SensitivityConfig(SensitivityMode.JOINT, 0, {TITRANT: 5.0})
    with pytest.raises(ValueError, match='at least 1'):
        run_sensitivity(ms, assay_cls, conditions, fit_config, cfg)


def test_negative_percent_raises(ida_setup):
    ms, assay_cls, conditions, fit_config = ida_setup
    cfg = SensitivityConfig(SensitivityMode.JOINT, 6, {TITRANT: -5.0})
    with pytest.raises(ValueError, match='negative'):
        run_sensitivity(ms, assay_cls, conditions, fit_config, cfg)


def test_all_zero_percent_raises(ida_setup):
    ms, assay_cls, conditions, fit_config = ida_setup
    cfg = SensitivityConfig(SensitivityMode.JOINT, 6, {TITRANT: 0.0})
    with pytest.raises(ValueError, match='vary'):
        run_sensitivity(ms, assay_cls, conditions, fit_config, cfg)


def test_heatmap_same_axis_twice_raises(ida_setup):
    ms, assay_cls, conditions, fit_config = ida_setup
    cfg = SensitivityConfig(SensitivityMode.HEATMAP, 6, {TITRANT: 5.0, 'h0': 5.0}, x_key=TITRANT, y_key=TITRANT)
    with pytest.raises(ValueError, match='different'):
        run_sensitivity(ms, assay_cls, conditions, fit_config, cfg)


def test_unknown_input_key_raises(ida_setup):
    ms, assay_cls, conditions, fit_config = ida_setup
    cfg = SensitivityConfig(SensitivityMode.JOINT, 6, {'not_an_input': 5.0})
    with pytest.raises(ValueError, match='Unknown'):
        run_sensitivity(ms, assay_cls, conditions, fit_config, cfg)


def test_dye_alone_has_no_ka_to_analyse():
    """DYE_ALONE has no association constant, so sensitivity does not apply."""
    x, y = _make_dye_alone_data({'slope': 5e7, 'intercept': 0.0})
    ms = MeasurementSet(concentrations=x, signals=y.reshape(1, -1), replica_ids=('r0',))
    cfg = SensitivityConfig(SensitivityMode.JOINT, 6, {TITRANT: 5.0})
    with pytest.raises(ValueError, match='association constant'):
        run_sensitivity(ms, DyeAloneAssay, {}, FitConfig(n_trials=5), cfg)
