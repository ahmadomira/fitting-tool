"""P5: Optimizer boundary tests.

Unit tests for edge cases and boundary conditions in the multi-start
optimizer and result filtering/aggregation functions.

Covers:
- generate_initial_guesses: empty bounds, single/zero trials, log-scale
- multistart_minimize: convergence, sorting, failure handling
- filter_by_rmse / filter_by_r_squared: empty input, thresholds
- select_valid_fits: R² floor gate, optional RMSE trim, #31 regression
- calculate_fit_metrics: perfect fit, constant y, known values
"""

import numpy as np
import pytest

from core.optimizer.filters import (
    calculate_fit_metrics,
    filter_by_r_squared,
    filter_by_rmse,
    select_valid_fits,
)
from core.optimizer.multistart import FitAttempt, generate_initial_guesses, multistart_minimize


def _make_attempt(params, cost=1.0, rmse=0.1, r_squared=0.99, success=True):
    """Create a FitAttempt with sensible defaults."""
    return FitAttempt(
        params=np.array(params, dtype=float),
        cost=cost,
        rmse=rmse,
        r_squared=r_squared,
        success=success,
    )


# ---------------------------------------------------------------------------
# generate_initial_guesses
# ---------------------------------------------------------------------------


class TestGenerateInitialGuesses:
    def test_zero_trials_returns_empty(self):
        guesses = generate_initial_guesses(0, [(0, 1)])
        assert guesses == []

    def test_single_trial(self):
        guesses = generate_initial_guesses(1, [(0, 10)])
        assert len(guesses) == 1
        assert guesses[0].shape == (1,)

    def test_empty_bounds_returns_zero_length_arrays(self):
        guesses = generate_initial_guesses(5, [])
        assert len(guesses) == 5
        for g in guesses:
            assert g.shape == (0,)

    def test_guesses_within_bounds(self):
        bounds = [(1.0, 5.0), (10.0, 20.0), (-3.0, 3.0)]
        guesses = generate_initial_guesses(50, bounds)
        for g in guesses:
            for i, (lo, hi) in enumerate(bounds):
                assert lo <= g[i] <= hi, f'param {i}: {g[i]} not in [{lo}, {hi}]'

    def test_log_scale_positive_bounds(self):
        bounds = [(1e-8, 1e12)]
        guesses = generate_initial_guesses(100, bounds, log_scale_params=[0])
        values = np.array([g[0] for g in guesses])
        assert np.all(values >= 1e-8)
        assert np.all(values <= 1e12)
        # Log-scale should span many orders of magnitude
        log_range = np.log10(values.max()) - np.log10(values.min())
        assert log_range > 5, f'Log range {log_range} too narrow for log-scale sampling'

    def test_log_scale_non_positive_lower_falls_back_to_linear(self):
        """When lower bound is 0, log-scale falls back to linear (no log10(0) crash)."""
        bounds = [(0.0, 100.0)]
        guesses = generate_initial_guesses(50, bounds, log_scale_params=[0])
        for g in guesses:
            assert 0.0 <= g[0] <= 100.0

    def test_equal_bounds_gives_fixed_value(self):
        bounds = [(5.0, 5.0), (3.0, 3.0)]
        guesses = generate_initial_guesses(10, bounds)
        for g in guesses:
            assert g[0] == 5.0
            assert g[1] == 3.0


# ---------------------------------------------------------------------------
# multistart_minimize
# ---------------------------------------------------------------------------


class TestMultistartMinimize:
    def test_simple_quadratic(self):
        """Minimizes f(x) = (x-3)^2 with bounds [0, 10]."""
        results = multistart_minimize(
            objective=lambda x: (x[0] - 3.0) ** 2,
            bounds=[(0.0, 10.0)],
            n_trials=10,
        )
        assert len(results) > 0
        assert results[0].params[0] == pytest.approx(3.0, abs=1e-4)
        assert results[0].cost == pytest.approx(0.0, abs=1e-8)

    def test_results_sorted_by_cost(self):
        results = multistart_minimize(
            objective=lambda x: (x[0] - 5.0) ** 2,
            bounds=[(0.0, 10.0)],
            n_trials=20,
        )
        costs = [r.cost for r in results]
        assert costs == sorted(costs)

    def test_custom_initial_guesses(self):
        """Provided initial_guesses are used instead of random generation."""
        guesses = [np.array([2.0]), np.array([4.0])]
        results = multistart_minimize(
            objective=lambda x: (x[0] - 3.0) ** 2,
            bounds=[(0.0, 10.0)],
            initial_guesses=guesses,
        )
        assert len(results) == 2

    def test_compute_metrics_callback(self):
        """compute_metrics populates rmse and r_squared fields."""
        results = multistart_minimize(
            objective=lambda x: (x[0] - 3.0) ** 2,
            bounds=[(0.0, 10.0)],
            n_trials=5,
            compute_metrics=lambda params: (0.042, 0.998),
        )
        for r in results:
            assert r.rmse == 0.042
            assert r.r_squared == 0.998

    def test_all_attempts_fail_returns_empty(self):
        """Objective that always raises returns empty results."""

        def bad_objective(x):
            raise RuntimeError('always fails')

        results = multistart_minimize(
            objective=bad_objective,
            bounds=[(0.0, 10.0)],
            n_trials=5,
        )
        assert results == []


# ---------------------------------------------------------------------------
# filter_by_rmse
# ---------------------------------------------------------------------------


class TestFilterByRmse:
    def test_empty_input(self):
        assert filter_by_rmse([]) == []

    def test_all_identical_rmse_all_pass(self):
        attempts = [_make_attempt([i], rmse=0.1) for i in range(5)]
        result = filter_by_rmse(attempts, threshold_factor=1.0)
        assert len(result) == 5

    def test_threshold_factor_1_keeps_only_best(self):
        attempts = [
            _make_attempt([1], rmse=0.1),
            _make_attempt([2], rmse=0.2),
            _make_attempt([3], rmse=0.5),
        ]
        result = filter_by_rmse(attempts, threshold_factor=1.0)
        assert len(result) == 1
        assert result[0].rmse == 0.1

    def test_explicit_reference_rmse(self):
        attempts = [
            _make_attempt([1], rmse=0.1),
            _make_attempt([2], rmse=0.3),
            _make_attempt([3], rmse=0.5),
        ]
        # reference_rmse=0.2, threshold_factor=2.0 → threshold=0.4
        result = filter_by_rmse(attempts, threshold_factor=2.0, reference_rmse=0.2)
        assert len(result) == 2  # rmse 0.1 and 0.3 pass; 0.5 doesn't


# ---------------------------------------------------------------------------
# filter_by_r_squared
# ---------------------------------------------------------------------------


class TestFilterByRSquared:
    def test_empty_input(self):
        assert filter_by_r_squared([]) == []

    def test_min_zero_passes_all(self):
        attempts = [
            _make_attempt([1], r_squared=0.5),
            _make_attempt([2], r_squared=-0.1),
            _make_attempt([3], r_squared=0.99),
        ]
        result = filter_by_r_squared(attempts, min_r_squared=0.0)
        assert len(result) == 2  # -0.1 < 0.0 fails

    def test_min_one_strict(self):
        attempts = [
            _make_attempt([1], r_squared=0.999),
            _make_attempt([2], r_squared=1.0),
        ]
        result = filter_by_r_squared(attempts, min_r_squared=1.0)
        assert len(result) == 1
        assert result[0].r_squared == 1.0

    def test_nan_r_squared_filtered_out(self):
        attempts = [
            _make_attempt([1], r_squared=np.nan),
            _make_attempt([2], r_squared=0.95),
        ]
        result = filter_by_r_squared(attempts, min_r_squared=0.9)
        assert len(result) == 1
        assert result[0].r_squared == 0.95


# ---------------------------------------------------------------------------
# select_valid_fits
# ---------------------------------------------------------------------------


class TestSelectValidFits:
    def test_empty_input(self):
        assert select_valid_fits([], min_r_squared=0.9) == []

    def test_r_squared_floor_is_the_gate(self):
        """The absolute R² floor decides membership; below-floor fits drop."""
        attempts = [
            _make_attempt([1.0], rmse=0.01, r_squared=0.99),
            _make_attempt([2.0], rmse=0.02, r_squared=0.5),
        ]
        valid = select_valid_fits(attempts, min_r_squared=0.9)
        assert len(valid) == 1
        assert valid[0].r_squared == 0.99

    def test_absolute_floor_keeps_a_good_fit_a_relative_rmse_trim_would_cut(self):
        """Regression for #31: a genuinely good fit just outside 1.5×best RMSE
        must NOT be discarded. Membership is decided by the absolute R² floor;
        a relative-RMSE trim on the same pool would cut it (asserted below)."""
        attempts = [
            _make_attempt([1.0], rmse=0.01, r_squared=0.999),  # best
            _make_attempt([1.1], rmse=0.02, r_squared=0.995),  # good, but > 1.5×0.01
        ]
        valid = select_valid_fits(attempts, min_r_squared=0.99)
        assert len(valid) == 2  # both survive on the absolute floor
        # The old relative-RMSE trim (1.5×min) would have dropped the second.
        assert len(filter_by_rmse(attempts, 1.5)) == 1

    def test_optional_rmse_trim_applies_when_set(self):
        """When rmse_threshold_factor is given it further trims the R²-passing
        pool relative to the best valid fit."""
        attempts = [
            _make_attempt([1.0], rmse=0.01, r_squared=0.999),
            _make_attempt([1.1], rmse=0.02, r_squared=0.995),
        ]
        valid = select_valid_fits(attempts, min_r_squared=0.99, rmse_threshold_factor=1.5)
        assert len(valid) == 1
        assert valid[0].rmse == 0.01


# ---------------------------------------------------------------------------
# calculate_fit_metrics
# ---------------------------------------------------------------------------


class TestCalculateFitMetrics:
    def test_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        rmse, r2 = calculate_fit_metrics(y, y)
        assert rmse == pytest.approx(0.0, abs=1e-15)
        assert r2 == pytest.approx(1.0)

    def test_constant_y_observed(self):
        """When all y_observed are identical, ss_tot=0 → R²=0."""
        y_obs = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        rmse, r2 = calculate_fit_metrics(y_obs, y_pred)
        assert r2 == 0.0
        assert rmse > 0

    def test_known_values(self):
        """Hand-computed RMSE and R² for simple data.

        residuals = [-0.1, 0, +0.1] → ss_res = 0.02, RMSE = sqrt(0.02/3);
        mean(y_obs) = 2 → ss_tot = 2, so R² = 1 − 0.02/2 = 0.99.
        """
        rmse, r2 = calculate_fit_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.0, 2.9]))
        assert rmse == pytest.approx(0.081649658092772, rel=1e-9)
        assert r2 == pytest.approx(0.99, rel=1e-12)

    def test_nan_prediction_propagates_nan(self):
        """A NaN anywhere in y_predicted (failed model eval) must yield NaN
        metrics, never a finite number that could pass QC filtering."""
        y_obs = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, np.nan, 3.0])
        rmse, r2 = calculate_fit_metrics(y_obs, y_pred)
        assert np.isnan(rmse)
        assert np.isnan(r2)


# ---------------------------------------------------------------------------
# GAP-9: linear_regression edge cases
# ---------------------------------------------------------------------------


class TestLinearRegression:
    """Edge cases for the linear regression helper."""

    def test_fewer_than_2_points_raises(self):
        from core.optimizer.linear_fit import linear_regression

        with pytest.raises(ValueError):
            linear_regression(np.array([1.0]), np.array([2.0]))
        with pytest.raises(ValueError):
            linear_regression(np.array([]), np.array([]))

    def test_perfect_fit(self):
        from core.optimizer.linear_fit import linear_regression

        x = np.array([0.0, 1.0, 2.0])
        y = np.array([3.0, 5.0, 7.0])  # slope=2, intercept=3
        slope, intercept, r2, rmse = linear_regression(x, y)

        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(3.0)
        assert r2 == pytest.approx(1.0)
        assert rmse == pytest.approx(0.0, abs=1e-12)
