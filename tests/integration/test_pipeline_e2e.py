"""P6: End-to-end integration tests for the fitting pipeline.

Exercises the full pipeline: assay construction → fit_assay() / fit_linear_assay()
→ FitResult validation.  Complements P1 (parameter recovery) by testing:

- FitResult structure, metadata, and serialization
- FitConfig customization (bounds overrides, log-scale overrides, n_trials)
- Chained workflow: dye-alone calibration → bounds derivation → downstream fit
- Failure modes (unknown params, wrong bounds)

Tolerance:
- Clean data:  10% for Ka, R² > 0.999
- Noisy data (5% Gaussian): 25% for Ka
"""

import numpy as np
import pytest

from core.assays.dba import DBAAssay
from core.assays.dye_alone import DyeAloneAssay
from core.assays.gda import GDAAssay
from core.assays.ida import IDAAssay
from core.data_processing.measurement_set import MeasurementSet
from core.models.equilibrium import dba_signal
from core.optimizer.filters import calculate_fit_metrics
from core.pipeline.fit_pipeline import (
    FitConfig,
    FitResult,
    bounds_from_dye_alone,
    fit_assay,
    fit_linear_assay,
    fit_measurement_set,
    summarize_parameters,
)
from core.units import Q_
from tests.conftest import (
    DBA_RECOVERY_BOUNDS,
    DBA_TRUE,
    DYE_ALONE_TRUE,
    GDA_IDA_RECOVERY_BOUNDS,
    _make_dba_data,
    assert_within_tolerance,
)

CLEAN_TOL = 0.10
NOISY_TOL = 0.25
N_TRIALS_CLEAN = 100
N_TRIALS_NOISY = 120

# Dye-alone ground truth CONSISTENT with binding assay signal coefficients.
# The dye-alone slope = I_dye_free and intercept = I0 from the binding model.
# (The conftest DYE_ALONE_TRUE uses different values, so we define our own.)
_CHAIN_DYE_ALONE_TRUE = {
    'slope': 5e7,  # matches DBA_TRUE['I_dye_free']
    'intercept': 0.0,  # matches DBA_TRUE['I0']
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dba_assay(fixture_data):
    x, y, true = fixture_data
    return DBAAssay(x_data=x, y_data=y, fixed_conc=Q_(true['fixed_conc'], 'M'), mode='DtoH'), true


def _gda_assay(fixture_data):
    x, y, true = fixture_data
    return GDAAssay(
        x_data=x, y_data=y, Ka_dye=Q_(true['Ka_dye'], '1/M'), h0=Q_(true['h0'], 'M'), g0=Q_(true['g0'], 'M')
    ), true


def _ida_assay(fixture_data):
    x, y, true = fixture_data
    return IDAAssay(
        x_data=x, y_data=y, Ka_dye=Q_(true['Ka_dye'], '1/M'), h0=Q_(true['h0'], 'M'), d0=Q_(true['d0'], 'M')
    ), true


def _dye_alone_assay(fixture_data):
    x, y, true = fixture_data
    return DyeAloneAssay(x_data=x, y_data=y), true


# ---------------------------------------------------------------------------
# DBA end-to-end
# ---------------------------------------------------------------------------


def _max_rel_err(assay, result, y):
    """Max point-wise relative error of the fitted model at the data points.

    Recomputes the model on the data grid, since ``x_fit``/``y_fit`` are now a
    denser display curve no longer aligned 1:1 with the measured points.
    """
    params = np.array([result.parameters[k].magnitude for k in assay.parameter_keys])
    y_at_data = assay.forward_model(params)
    return np.max(np.abs((y_at_data - y).magnitude) / np.abs(y.magnitude))


@pytest.fixture(scope='module')
def dba_clean_result():
    """One shared clean DBA fit, reused by the round-trip and structure tests."""
    x, y = _make_dba_data(DBA_TRUE)
    assay = DBAAssay(
        x_data=Q_(x, 'M'),
        y_data=Q_(y, 'au'),
        fixed_conc=Q_(DBA_TRUE['fixed_conc'], 'M'),
        mode='DtoH',
    )
    # Module fixtures run before the autouse function-scoped seeder, so seed
    # explicitly — and restore the global RNG state to avoid leaking into
    # other module-scoped fixtures.
    state = np.random.get_state()
    np.random.seed(42)
    try:
        result = fit_assay(assay, FitConfig(n_trials=N_TRIALS_CLEAN, custom_bounds=DBA_RECOVERY_BOUNDS))
    finally:
        np.random.set_state(state)
    return result, Q_(y, 'au'), assay


class TestDBAEndToEnd:
    def test_clean_round_trip(self, dba_clean_result):
        result, y, assay = dba_clean_result

        assert result.success
        assert result.r_squared > 0.999
        assert_within_tolerance(result.parameters['Ka_dye'], DBA_TRUE['Ka_dye'], CLEAN_TOL, 'Ka_dye')
        # Signal reconstruction: fitted model matches the data point-wise.
        max_rel_err = _max_rel_err(assay, result, y)
        assert max_rel_err < 0.03, f'Max point-wise relative error {max_rel_err:.2%} > 3%'

    def test_fit_result_structure(self, dba_clean_result):
        result, _, assay = dba_clean_result

        # Type and model
        assert result.assay_type == 'DBA_DtoH'
        assert result.model_name == 'equilibrium_4param'

        # Parameters populated, each with a pool the reported range comes from
        summaries = {s.key: s for s in summarize_parameters(result) if not s.is_log}
        for key in ('Ka_dye', 'I0', 'I_dye_free', 'I_dye_bound'):
            assert key in result.parameters
            assert summaries[key].stats is not None

        # Fit curve is a dense, sorted grid spanning the data range (smooth display curve)
        assert result.x_fit.shape == result.y_fit.shape
        assert len(result.x_fit) > len(assay.x_data)
        x_mag = result.x_fit.magnitude
        assert np.all(np.diff(x_mag) > 0)
        assert x_mag[0] == pytest.approx(assay.x_data.magnitude.min())
        assert x_mag[-1] == pytest.approx(assay.x_data.magnitude.max())

        # Counts
        assert result.n_passing > 0
        assert result.n_total >= result.n_passing

        # Conditions
        assert 'fixed_conc' in result.conditions

        # FitConfig snapshot
        assert 'n_trials' in result.fit_config

        # UUID and timestamp populated
        assert len(result.id) > 0
        assert len(result.timestamp) > 0


# ---------------------------------------------------------------------------
# GDA end-to-end
# ---------------------------------------------------------------------------


class TestGDAEndToEnd:
    def test_clean_round_trip(self, gda_clean):
        assay, true = _gda_assay(gda_clean)
        x, y, _ = gda_clean
        result = fit_assay(assay, FitConfig(n_trials=N_TRIALS_CLEAN, custom_bounds=GDA_IDA_RECOVERY_BOUNDS))

        assert result.success
        assert result.r_squared > 0.999
        assert_within_tolerance(result.parameters['Ka_guest'], true['Ka_guest'], CLEAN_TOL, 'Ka_guest')
        # Reconstruction tolerance is looser for GDA (weaker identifiability).
        max_rel_err = _max_rel_err(assay, result, y)
        assert max_rel_err < 0.10, f'Max point-wise relative error {max_rel_err:.2%} > 10%'


# ---------------------------------------------------------------------------
# IDA end-to-end
# ---------------------------------------------------------------------------


class TestIDAEndToEnd:
    def test_clean_round_trip(self, ida_clean):
        assay, true = _ida_assay(ida_clean)
        x, y, _ = ida_clean
        result = fit_assay(assay, FitConfig(n_trials=N_TRIALS_CLEAN, custom_bounds=GDA_IDA_RECOVERY_BOUNDS))

        assert result.success
        assert result.r_squared > 0.999
        assert_within_tolerance(result.parameters['Ka_guest'], true['Ka_guest'], CLEAN_TOL, 'Ka_guest')
        max_rel_err = _max_rel_err(assay, result, y)
        assert max_rel_err < 0.01, f'Max point-wise relative error {max_rel_err:.2%} > 1%'

    def test_noisy_round_trip(self, ida_noisy):
        assay, true = _ida_assay(ida_noisy)
        result = fit_assay(assay, FitConfig(n_trials=N_TRIALS_NOISY, custom_bounds=GDA_IDA_RECOVERY_BOUNDS))

        assert result.success
        assert_within_tolerance(result.parameters['Ka_guest'], true['Ka_guest'], NOISY_TOL, 'Ka_guest')


# ---------------------------------------------------------------------------
# DyeAlone end-to-end
# ---------------------------------------------------------------------------


class TestDyeAloneEndToEnd:
    def test_linear_fit_round_trip(self, dye_alone_clean):
        assay, true = _dye_alone_assay(dye_alone_clean)
        result = fit_linear_assay(assay)

        assert result.success
        assert result.r_squared > 0.999
        assert_within_tolerance(result.parameters['slope'], true['slope'], CLEAN_TOL, 'slope')
        assert_within_tolerance(result.parameters['intercept'], true['intercept'], CLEAN_TOL, 'intercept')

    def test_metadata(self, dye_alone_clean):
        assay, _ = _dye_alone_assay(dye_alone_clean)
        result = fit_linear_assay(assay)

        assert result.assay_type == 'DYE_ALONE'
        assert result.model_name == 'linear'
        assert result.metadata.get('method') == 'linear_regression'
        assert result.n_passing == 1
        assert result.n_total == 1


# ---------------------------------------------------------------------------
# FitConfig customization
# ---------------------------------------------------------------------------


class TestFitConfigCustomization:
    def test_relaxed_min_r_squared_passes_more(self, dba_clean):
        """Relaxing min_r_squared allows more fits to pass filtering."""
        assay, _ = _dba_assay(dba_clean)
        config_strict = FitConfig(n_trials=20, custom_bounds=DBA_RECOVERY_BOUNDS, min_r_squared=0.999)
        config_relaxed = FitConfig(n_trials=20, custom_bounds=DBA_RECOVERY_BOUNDS, min_r_squared=0.0)

        # Re-seed before each fit so both runs see identical multistart state;
        # otherwise the RNG drifts between calls and the comparison is meaningless.
        np.random.seed(42)
        result_strict = fit_assay(assay, config_strict)
        np.random.seed(42)
        result_relaxed = fit_assay(assay, config_relaxed)

        assert result_relaxed.n_passing >= result_strict.n_passing


# ---------------------------------------------------------------------------
# Chained workflow: dye-alone → downstream fit
# ---------------------------------------------------------------------------


class TestChainedWorkflow:
    """Test the calibration chain: dye-alone → derive signal bounds → downstream fit.

    Uses dye-alone ground truth consistent with binding assay I_dye_free/I0,
    since the conftest DYE_ALONE_TRUE uses different (unrelated) values.
    """

    def _consistent_dye_alone_assay(self):
        """Build a dye-alone assay with signal params matching binding assays."""
        from core.models.linear import linear_signal

        x = np.linspace(0, 20e-6, 15)
        y = linear_signal(_CHAIN_DYE_ALONE_TRUE['slope'], _CHAIN_DYE_ALONE_TRUE['intercept'], x)
        return DyeAloneAssay(x_data=Q_(x, 'M'), y_data=Q_(y, 'au'))

    def _dye_alone_bounds(self):
        """Fit consistent dye-alone and derive signal bounds."""
        assay = self._consistent_dye_alone_assay()
        dye_result = fit_linear_assay(assay)
        assert dye_result.success
        return bounds_from_dye_alone(dye_result, margin=0.2)

    def test_dye_alone_to_dba(self, dba_clean):
        signal_bounds = self._dye_alone_bounds()
        assay, true = _dba_assay(dba_clean)

        # Dye-alone constrains I_dye_free and I0; I_dye_bound still needs
        # separate bounds (not observable in dye-alone experiment).
        config = FitConfig(
            n_trials=N_TRIALS_CLEAN,
            custom_bounds={
                **DBA_RECOVERY_BOUNDS,  # I_dye_bound + Ka_dye defaults
                **signal_bounds,  # override I_dye_free and I0
            },
        )
        result = fit_assay(assay, config)

        assert result.success
        assert_within_tolerance(result.parameters['Ka_dye'], true['Ka_dye'], CLEAN_TOL, 'Ka_dye')

    def test_bounds_from_dye_alone_values(self, dye_alone_clean):
        """Verify derived bounds bracket the known truth."""
        assay, true = _dye_alone_assay(dye_alone_clean)
        dye_result = fit_linear_assay(assay)
        bounds = bounds_from_dye_alone(dye_result, margin=0.2)

        # I_dye_free bounds should bracket the true slope (≈ I_dye_free)
        lo, hi = bounds['I_dye_free']
        assert lo.magnitude > 0
        assert lo.magnitude < true['slope'] < hi.magnitude

        # I0 bounds should bracket the true intercept
        lo, hi = bounds['I0']
        assert lo.magnitude <= true['intercept'] <= hi.magnitude


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_unknown_custom_bounds_key_raises(self, dba_clean):
        assay, _ = _dba_assay(dba_clean)
        config = FitConfig(custom_bounds={'bogus_param': (Q_(0, 'au'), Q_(1, 'au'))})
        with pytest.raises(ValueError, match='Unknown parameter'):
            fit_assay(assay, config)

    def test_unknown_log_scale_param_raises(self, dba_clean):
        assay, _ = _dba_assay(dba_clean)
        config = FitConfig(
            custom_bounds=DBA_RECOVERY_BOUNDS,
            log_scale_params=['nonexistent'],
        )
        with pytest.raises(ValueError, match='Unknown log-scale parameter'):
            fit_assay(assay, config)

    def test_very_wrong_bounds_no_crash(self, dba_clean):
        """Extremely wrong Ka bounds produce a result (no exception)."""
        assay, true = _dba_assay(dba_clean)
        wrong_bounds = {
            **DBA_RECOVERY_BOUNDS,
            'Ka_dye': (Q_(1e-15, '1/M'), Q_(1e-14, '1/M')),  # far from truth
        }
        result = fit_assay(assay, FitConfig(n_trials=20, custom_bounds=wrong_bounds))

        # Should return a FitResult (no crash), but Ka will be wrong
        assert isinstance(result, FitResult)

    def test_underdetermined_data_returns_result(self):
        """n_points < n_params (2 points, 4-param model) must complete and
        return a FitResult — failure is reported via success/n_passing, not
        a crash deep inside the optimizer."""
        assay = IDAAssay(
            x_data=Q_(np.array([1e-6, 2e-6]), 'M'),
            y_data=Q_(np.array([100.0, 80.0]), 'au'),
            Ka_dye=Q_(5e5, '1/M'),
            h0=Q_(10e-6, 'M'),
            d0=Q_(5e-6, 'M'),
        )
        result = fit_assay(assay, FitConfig(n_trials=5))
        assert isinstance(result, FitResult)

    def test_bounds_from_dye_alone_rejects_nonlinear(self):
        """bounds_from_dye_alone raises on non-linear FitResult."""
        result = FitResult(
            parameters={'Ka_dye': Q_(5e5, '1/M'), 'I0': Q_(0.0, 'au')},
            rmse=0.01,
            r_squared=0.999,
            n_passing=5,
            n_total=10,
            x_fit=Q_(np.array([1e-6]), 'M'),
            y_fit=Q_(np.array([100.0]), 'au'),
            assay_type='DBA_DtoH',
            model_name='equilibrium_4param',
        )
        with pytest.raises(ValueError, match='linear'):
            bounds_from_dye_alone(result)


# ---------------------------------------------------------------------------
# GAP-1: DBA HtoD mode recovery
# ---------------------------------------------------------------------------


class TestDBAHtoD:
    def test_htod_mode_clean_round_trip(self):
        """Ka_dye recovered in HtoD mode (host titrated, dye fixed)."""
        x = np.linspace(1e-7, 50e-6, 30)
        y = dba_signal(
            I0=DBA_TRUE['I0'],
            Ka_dye=DBA_TRUE['Ka_dye'],
            I_dye_free=DBA_TRUE['I_dye_free'],
            I_dye_bound=DBA_TRUE['I_dye_bound'],
            x_titrant=x,
            y_fixed=DBA_TRUE['fixed_conc'],
            mode='HtoD',
        )
        assay = DBAAssay(
            x_data=Q_(x, 'M'),
            y_data=Q_(y, 'au'),
            fixed_conc=Q_(DBA_TRUE['fixed_conc'], 'M'),
            mode='HtoD',
        )
        result = fit_assay(assay, FitConfig(n_trials=N_TRIALS_CLEAN, custom_bounds=DBA_RECOVERY_BOUNDS))

        assert result.success
        assert result.assay_type == 'DBA_HtoD'
        assert_within_tolerance(result.parameters['Ka_dye'], DBA_TRUE['Ka_dye'], CLEAN_TOL, 'Ka_dye')


# ---------------------------------------------------------------------------
# GAP-10: fit_measurement_set convenience function
# ---------------------------------------------------------------------------


class TestFitMeasurementSet:
    def test_fit_measurement_set_dye_alone(self, dye_alone_clean):
        """fit_measurement_set dispatches to fit_linear_assay for DyeAlone."""
        x, y, true = dye_alone_clean
        signals = y.magnitude.reshape(1, -1)
        ms = MeasurementSet(
            concentrations=x.magnitude,
            signals=signals,
            replica_ids=('r0',),
        )
        result = fit_measurement_set(ms, DyeAloneAssay, {})

        assert result.success
        assert result.model_name == 'linear'
        assert result.measurement_set_id == ms.id

    def test_fit_measurement_set_gda(self, gda_clean):
        """fit_measurement_set dispatches to fit_assay for GDA."""
        x, y, true = gda_clean
        signals = y.magnitude.reshape(1, -1)
        ms = MeasurementSet(
            concentrations=x.magnitude,
            signals=signals,
            replica_ids=('r0',),
        )
        conditions = {
            'Ka_dye': Q_(true['Ka_dye'], '1/M'),
            'h0': Q_(true['h0'], 'M'),
            'g0': Q_(true['g0'], 'M'),
        }
        # Dispatch test only — 30 trials with tight bounds is plenty for success.
        result = fit_measurement_set(
            ms,
            GDAAssay,
            conditions,
            config=FitConfig(n_trials=30, custom_bounds=GDA_IDA_RECOVERY_BOUNDS),
        )

        assert result.success
        assert result.measurement_set_id == ms.id


# ---------------------------------------------------------------------------
# GAP-14: DyeAlone noisy recovery
# ---------------------------------------------------------------------------


class TestDyeAloneNoisy:
    def test_noisy_linear_fit(self):
        from tests.conftest import _make_dye_alone_data

        x, y = _make_dye_alone_data(DYE_ALONE_TRUE, noise_frac=0.05)
        assay = DyeAloneAssay(x_data=Q_(x, 'M'), y_data=Q_(y, 'au'))
        result = fit_linear_assay(assay)

        assert result.success
        assert_within_tolerance(result.parameters['slope'], DYE_ALONE_TRUE['slope'], NOISY_TOL, 'slope')


# ---------------------------------------------------------------------------
# GAP-18: bounds_from_dye_alone margin edge cases
# ---------------------------------------------------------------------------


class TestBoundsMarginEdgeCases:
    def test_margin_zero_collapses_to_point(self, dye_alone_clean):
        assay, true = _dye_alone_assay(dye_alone_clean)
        result = fit_linear_assay(assay)
        bounds = bounds_from_dye_alone(result, margin=0.0)

        lo, hi = bounds['I_dye_free']
        assert lo.magnitude == pytest.approx(hi.magnitude)

    def test_failed_fit_raises(self):
        """bounds_from_dye_alone rejects failed fits."""
        r = FitResult(
            parameters={'slope': Q_(1.0, 'au/M'), 'intercept': Q_(0.0, 'au')},
            rmse=np.inf,
            r_squared=0.0,
            n_passing=0,
            n_total=10,
            x_fit=Q_(np.array([]), 'M'),
            y_fit=Q_(np.array([]), 'au'),
            assay_type='DYE_ALONE',
            model_name='linear',
        )
        with pytest.raises(ValueError, match='failed'):
            bounds_from_dye_alone(r)


# ---------------------------------------------------------------------------
# #31: representative fit is a real fit, never a synthetic per-parameter median
# ---------------------------------------------------------------------------


class TestRepresentativeFitTrustworthy:
    """Regression for #31. On the degenerate IDA signal model every valid fit
    sits on a different point of the offset/contrast manifold, so an
    independent per-parameter median lands *off* the manifold and reconstructs
    worse than every real fit. The reported result must instead be an actual
    fit (the best by R²), so its RMSE/R² are real and self-consistent."""

    def test_reported_rmse_is_the_pool_minimum(self, ida_clean):
        assay, _true = _ida_assay(ida_clean)
        result = fit_assay(assay, FitConfig(n_trials=N_TRIALS_CLEAN, custom_bounds=GDA_IDA_RECOVERY_BOUNDS))

        assert result.success
        rmse_pool = result.quality_samples['rmse']
        ridx = result.representative_index
        # Reported metrics belong to the representative trial, the pool's best.
        assert result.rmse == pytest.approx(float(rmse_pool[ridx]))
        assert result.rmse == pytest.approx(float(rmse_pool.min()))
        assert result.r_squared == pytest.approx(float(result.quality_samples['r_squared'].max()))

    def test_representative_is_no_worse_than_the_old_per_parameter_median(self, ida_clean):
        """The old aggregator reported the per-parameter median; on this
        degenerate model that median reconstructs worse than the representative
        real fit. The reported RMSE must be ≤ that median's RMSE."""
        assay, _true = _ida_assay(ida_clean)
        result = fit_assay(assay, FitConfig(n_trials=N_TRIALS_CLEAN, custom_bounds=GDA_IDA_RECOVERY_BOUNDS))

        pool = np.column_stack([result.parameter_samples[k] for k in assay.parameter_keys])
        median_vec = np.median(pool, axis=0)
        y_median = assay.forward_model(median_vec).magnitude
        rmse_median, _ = calculate_fit_metrics(assay.y_data.magnitude, y_median)

        assert result.rmse <= rmse_median + 1e-12
