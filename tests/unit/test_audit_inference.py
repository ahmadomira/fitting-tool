"""Inference checks using explicit line solutions, residuals, and unit conversions."""

import numpy as np
import pytest

from core.assays.dye_alone import DyeAloneAssay
from core.data_processing.measurement_set import MeasurementSet
from core.optimizer.linear_fit import linear_regression
from core.optimizer.multistart import multistart_minimize
from core.pipeline.fit_pipeline import FitConfig, _config_to_dict, fit_measurement_set
from core.units import Q_


def test_calibration_rejects_rank_deficient_concentration_design():
    # A constant concentration identifies only slope*x + intercept.
    with pytest.raises(ValueError, match='distinct'):
        linear_regression(np.array([1e-9, 1e-9, 1e-9]), np.array([3.0, 4.0, 5.0]))


@pytest.mark.parametrize('bad_y', [np.array([1.0, np.nan, 3.0]), np.array([1.0, np.inf, 3.0])])
def test_calibration_rejects_nonfinite_observations(bad_y):
    with pytest.raises(ValueError, match='finite'):
        linear_regression(np.array([1e-6, 2e-6, 3e-6]), bad_y)


def test_calibration_matches_independent_normal_equation_solution():
    # Centered sums give slope=2.5 au/uM, intercept=0, residuals=(.5,-1,.5).
    slope, intercept, r2, rmse = linear_regression(np.array([1e-6, 2e-6, 3e-6]), np.array([3.0, 4.0, 8.0]))
    assert slope == pytest.approx(2.5e6)
    assert intercept == pytest.approx(0.0, abs=1e-12)
    assert rmse == pytest.approx(np.sqrt(0.5))
    assert r2 == pytest.approx(25 / 28)


def test_optimizer_does_not_label_root_sse_as_rmse_without_sample_count():
    # Four unit residuals have SSE=4, residual norm=2, RMSE=1. A scalar
    # objective alone cannot disclose the observation count to the optimizer.
    attempt = multistart_minimize(lambda p: 4.0, [(0.0, 0.0)], initial_guesses=[np.array([0.0])])[0]
    assert attempt.cost == 4.0
    assert np.isnan(attempt.rmse)
    assert np.isnan(attempt.r_squared)


def test_bound_provenance_converts_both_endpoints_to_its_recorded_unit():
    config = FitConfig(custom_bounds={'Ka_dye': (Q_(1.0, '1/uM'), Q_(2e6, '1/M'))})
    lower, upper, unit = _config_to_dict(config)['custom_bounds']['Ka_dye']
    assert Q_(lower, unit).to('1/M').magnitude == pytest.approx(1e6)
    assert Q_(upper, unit).to('1/M').magnitude == pytest.approx(2e6)


def test_dye_calibration_can_fit_each_replica_and_pool_measured_lines():
    # Independent literal lines: 2 au/uM + 1 au, and 3 au/uM + 2 au.
    ms = MeasurementSet(
        concentrations=np.array([0.0, 1e-6, 2e-6]),
        signals=np.array([[1.0, 3.0, 5.0], [2.0, 5.0, 8.0]]),
        replica_ids=('a', 'b'),
    )
    result = fit_measurement_set(ms, DyeAloneAssay, {}, FitConfig(per_replica=True))
    assert result.success
    assert result.n_passing == 2
    np.testing.assert_allclose(result.parameter_samples['slope'], [2e6, 3e6])
    np.testing.assert_allclose(result.parameter_samples['intercept'], [1.0, 2.0])
    assert result.uncertainty_source == 'replicate'
