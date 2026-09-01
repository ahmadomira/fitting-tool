"""End-to-end tests for per-replica fitting.

Verifies:
- parameter recovery on a multi-replica synthetic IDA dataset;
- outlier-replica robustness (cross-replica median absorbs one bad trace);
- rescaling invariance (per-replica results are indifferent to the
  rescale_parameters flag, since the scaler is an exact bijection);
- degenerate-replica handling (flat signal → ParamScaler.from_assay raises
  ValueError; that replica is skipped with a failure record, others survive);
- all-failure propagation via PerReplicaFitError;
- pool aggregation semantics (median/MAD over the flat pool of passing trials).

Per-replica fits are expensive (5 replicas × 60 trials), so the clean and
noisy reference fits are module-scoped fixtures shared by every test that
only *reads* the result.  Tests that need a different MeasurementSet
(outliers, degenerate replicas, rescale comparison) run their own fits.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from core.assays.ida import IDAAssay
from core.data_processing.measurement_set import MeasurementSet
from core.pipeline.fit_pipeline import (
    FitConfig,
    FitResult,
    PerReplicaFitError,
    fit_measurement_set,
    fit_measurement_set_per_replica,
)
from core.units import Q_
from tests.conftest import GDA_IDA_RECOVERY_BOUNDS, IDA_TRUE, _make_ida_data, assert_within_tolerance

N_REPLICAS = 5
N_TRIALS = 60  # Smaller than default; 5 replicas * 60 still fast enough for CI.
CLEAN_TOL = 0.10
NOISY_TOL = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ida_conditions() -> Dict[str, Any]:
    return {
        'Ka_dye': Q_(IDA_TRUE['Ka_dye'], '1/M'),
        'h0': Q_(IDA_TRUE['h0'], 'M'),
        'd0': Q_(IDA_TRUE['d0'], 'M'),
    }


def _ida_ms(
    n_replicas: int = N_REPLICAS,
    noise_frac: float = 0.05,
    seed: int = 2026,
) -> MeasurementSet:
    """Build a MeasurementSet with N independent noisy IDA replicas."""
    rng = np.random.default_rng(seed)
    x_ref, _ = _make_ida_data(IDA_TRUE, noise_frac=0.0)
    signals = np.stack(
        [
            _make_ida_data(IDA_TRUE, noise_frac=noise_frac, rng=np.random.default_rng(rng.integers(1 << 31)))[1]
            for _ in range(n_replicas)
        ]
    )
    return MeasurementSet(
        concentrations=x_ref,
        signals=signals,
        replica_ids=tuple(f'r{i}' for i in range(n_replicas)),
        metadata={'source_file': 'synthetic_per_replica_test'},
    )


def _per_replica_config() -> FitConfig:
    return FitConfig(
        n_trials=N_TRIALS,
        custom_bounds=GDA_IDA_RECOVERY_BOUNDS,
        per_replica=True,
    )


# ---------------------------------------------------------------------------
# Shared reference fits (module-scoped: each runs exactly once)
# ---------------------------------------------------------------------------


def _seeded_per_replica_fit(noise_frac: float, seed: int) -> FitResult:
    """Run a per-replica fit under a deterministic global RNG seed, restoring
    the prior RNG state afterwards (module fixtures run before the autouse
    per-test seeder, so leaked state could affect other module fixtures)."""
    ms = _ida_ms(noise_frac=noise_frac, seed=seed)
    state = np.random.get_state()
    np.random.seed(42)
    try:
        return fit_measurement_set_per_replica(ms, IDAAssay, _ida_conditions(), _per_replica_config())
    finally:
        np.random.set_state(state)


@pytest.fixture(scope='module')
def clean_pr_result() -> FitResult:
    """Per-replica fit of 5 noise-free replicas."""
    return _seeded_per_replica_fit(noise_frac=0.0, seed=1)


@pytest.fixture(scope='module')
def noisy_pr_result() -> FitResult:
    """Per-replica fit of 5 replicas with 5% Gaussian noise."""
    return _seeded_per_replica_fit(noise_frac=0.05, seed=7)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


class TestPerReplicateRecovery:
    def test_ka_recovered_from_clean_replicas(self, clean_pr_result):
        result = clean_pr_result

        assert result.success
        assert result.uncertainty_source == 'replicate'
        assert result.replica_fits is not None
        assert len(result.replica_fits) == N_REPLICAS
        assert result.metadata['n_replicas_fit'] == N_REPLICAS
        assert result.metadata['n_replicas_total'] == N_REPLICAS
        assert_within_tolerance(result.parameters['Ka_guest'], IDA_TRUE['Ka_guest'], CLEAN_TOL, 'Ka_guest')

    def test_ka_recovered_from_noisy_replicas(self, noisy_pr_result):
        result = noisy_pr_result

        assert result.success
        assert len(result.replica_fits) == N_REPLICAS
        assert_within_tolerance(result.parameters['Ka_guest'], IDA_TRUE['Ka_guest'], NOISY_TOL, 'Ka_guest')

    def test_replica_spread_is_nonzero_on_noisy_data(self, noisy_pr_result):
        """Cross-replica spread should capture real noise — not collapse to ~0."""
        from core.pipeline.fit_pipeline import summarize_parameters

        result = noisy_pr_result
        ka_mag = float(result.parameters['Ka_guest'].magnitude)
        stats = next(s.stats for s in summarize_parameters(result) if s.key == 'Ka_guest' and not s.is_log)

        assert stats['mad'] > 0.0
        assert stats['max'] > stats['min']
        # The spread should be at most comparable to Ka itself; anything vastly
        # larger indicates aggregation broke.
        assert stats['mad'] / ka_mag < 2.0

    def test_per_replica_fits_are_in_physical_units(self, clean_pr_result):
        """Each replica fit must already be in physical units (M^-1 etc)."""
        for rr in clean_pr_result.replica_fits:
            assert str(rr.parameters['Ka_guest'].units) == '1 / molar'
            assert rr.uncertainty_source == 'optimizer'  # each replica is a single-signal fit

    def test_dispatch_via_fit_measurement_set(self):
        """fit_measurement_set honors config.per_replica and dispatches."""
        ms = _ida_ms(noise_frac=0.0, seed=3)
        # Dispatch-only check: 20 trials is plenty to produce a result.
        config = FitConfig(n_trials=20, custom_bounds=GDA_IDA_RECOVERY_BOUNDS, per_replica=True)
        result = fit_measurement_set(ms, IDAAssay, _ida_conditions(), config)

        assert result.uncertainty_source == 'replicate'
        assert result.replica_fits is not None


# ---------------------------------------------------------------------------
# Outlier-replica robustness
# ---------------------------------------------------------------------------


class TestOutlierReplicateRobustness:
    def test_one_bad_replica_does_not_wreck_median(self):
        """Replace one replica trace with pure noise. The cross-replica
        median of Ka should still recover ground truth within noisy tolerance,
        because the bad replica's Ka is an outlier absorbed by the median."""
        ms_good = _ida_ms(noise_frac=0.02, seed=101)

        # Inject one garbage replica into a fresh MeasurementSet at idx 2
        signals = np.array(ms_good.signals, dtype=np.float64)
        rng = np.random.default_rng(999)
        peak = float(np.max(np.abs(signals)))
        signals[2] = rng.normal(0.0, peak * 0.5, size=signals.shape[1])
        ms = MeasurementSet(
            concentrations=ms_good.concentrations,
            signals=signals,
            replica_ids=ms_good.replica_ids,
            metadata=dict(ms_good.metadata),
        )

        result = fit_measurement_set_per_replica(ms, IDAAssay, _ida_conditions(), _per_replica_config())

        assert result.success
        # Most replicas should survive (the bad one may or may not converge)
        assert len(result.replica_fits) >= N_REPLICAS - 2
        assert_within_tolerance(
            result.parameters['Ka_guest'],
            IDA_TRUE['Ka_guest'],
            NOISY_TOL,
            'Ka_guest (outlier-injected)',
        )


# ---------------------------------------------------------------------------
# Rescaling invariance
# ---------------------------------------------------------------------------


class TestRescalingInvariance:
    """Parameter rescaling is an exact affine bijection
    [core/optimizer/scaling.py:20-23]. Per-replica fits must produce the
    same physical-unit parameters whether rescaling is on or off, up to a
    loose tolerance to absorb basin-selection jitter on noisy data."""

    def test_per_replica_result_matches_with_and_without_rescale(self):
        ms = _ida_ms(noise_frac=0.0, seed=17)

        cfg_on = FitConfig(
            n_trials=N_TRIALS,
            custom_bounds=GDA_IDA_RECOVERY_BOUNDS,
            per_replica=True,
            rescale_parameters=True,
        )
        cfg_off = FitConfig(
            n_trials=N_TRIALS,
            custom_bounds=GDA_IDA_RECOVERY_BOUNDS,
            per_replica=True,
            rescale_parameters=False,
        )

        r_on = fit_measurement_set_per_replica(ms, IDAAssay, _ida_conditions(), cfg_on)
        r_off = fit_measurement_set_per_replica(ms, IDAAssay, _ida_conditions(), cfg_off)

        ka_on = float(r_on.parameters['Ka_guest'].magnitude)
        ka_off = float(r_off.parameters['Ka_guest'].magnitude)
        # Both must recover ground truth (clean data, 10% tolerance);
        # that already pins them to the same physical value much tighter
        # than the bijection's first-order equivalence alone.
        assert_within_tolerance(ka_on, IDA_TRUE['Ka_guest'], CLEAN_TOL, 'Ka_guest (rescale on)')
        assert_within_tolerance(ka_off, IDA_TRUE['Ka_guest'], CLEAN_TOL, 'Ka_guest (rescale off)')
        # And they must agree with each other within the same tolerance.
        assert abs(ka_on - ka_off) / max(abs(ka_on), abs(ka_off)) <= CLEAN_TOL


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_degenerate_replica_is_skipped(self):
        """A flat (zero-variance) replica makes ParamScaler.from_assay raise
        ValueError; that replica must be skipped with a failure record,
        surviving replicas fit normally, and the aggregate result is
        produced."""
        ms_good = _ida_ms(noise_frac=0.02, seed=37)
        signals = np.array(ms_good.signals, dtype=np.float64)
        signals[1] = 0.0  # flat signal, violates s_max > 0
        ms = MeasurementSet(
            concentrations=ms_good.concentrations,
            signals=signals,
            replica_ids=ms_good.replica_ids,
            metadata=dict(ms_good.metadata),
        )

        result = fit_measurement_set_per_replica(ms, IDAAssay, _ida_conditions(), _per_replica_config())

        assert result.success
        assert 'replica_failures' in result.metadata
        assert 'r1' in result.metadata['replica_failures']
        assert result.metadata['n_replicas_fit'] == N_REPLICAS - 1

    def test_all_failures_raise_per_replica_fit_error(self):
        """Every replica is degenerate → PerReplicaFitError with full failure map."""
        ms_good = _ida_ms(n_replicas=3, noise_frac=0.0, seed=51)
        signals = np.zeros_like(ms_good.signals)
        ms = MeasurementSet(
            concentrations=ms_good.concentrations,
            signals=signals,
            replica_ids=ms_good.replica_ids,
            metadata=dict(ms_good.metadata),
        )

        with pytest.raises(PerReplicaFitError) as exc_info:
            fit_measurement_set_per_replica(ms, IDAAssay, _ida_conditions(), _per_replica_config())

        err = exc_info.value
        assert len(err.failures) == 3
        assert all(rid in err.failures for rid in ms.replica_ids)


# ---------------------------------------------------------------------------
# Pool aggregation semantics (approach (b))
# ---------------------------------------------------------------------------


class TestPoolAggregation:
    """The per-replica aggregate must carry the flat pool of every valid
    trial from every replica in `parameter_samples`. The reported value is
    the representative real trial (a row of the pool, not a per-parameter
    aggregate), and the default-mode uncertainty is the pool's MAD."""

    def test_aggregate_has_parameter_samples_pool(self, noisy_pr_result):
        result = noisy_pr_result

        assert result.parameter_samples is not None
        # One entry per parameter key, same length across keys.
        lengths = {k: len(arr) for k, arr in result.parameter_samples.items()}
        assert len(set(lengths.values())) == 1
        pool_size = next(iter(lengths.values()))
        # Pool size equals the sum of per-replica n_passing
        expected = sum(rf.n_passing for rf in result.replica_fits)
        assert pool_size == expected
        # And equals n_passing on the aggregate, and metadata pool_size
        assert result.n_passing == pool_size
        assert result.metadata['pool_size'] == pool_size
        # Pool is strictly larger than the number of replicas (many trials each)
        assert pool_size > len(result.replica_fits)

    def test_per_replica_fits_carry_parameter_samples(self, noisy_pr_result):
        for rf in noisy_pr_result.replica_fits:
            assert rf.parameter_samples is not None
            for k in rf.parameters.keys():
                assert len(rf.parameter_samples[k]) == rf.n_passing

    def test_reported_value_is_representative_and_range_is_pool_extent(self, noisy_pr_result):
        from core.pipeline.fit_pipeline import summarize_parameters

        result = noisy_pr_result
        ridx = result.representative_index
        summaries = {s.key: s for s in summarize_parameters(result) if not s.is_log}
        for k, q in result.parameters.items():
            pool = result.parameter_samples[k]
            # Reported value = the representative real trial (a row of the pool),
            # NOT a per-parameter median that could land off the manifold.
            assert float(q.magnitude) == pytest.approx(float(pool[ridx]), rel=1e-12)
            # The quoted range is the full extent of the accepted pool, so the
            # representative must lie inside it.
            stats = summaries[k].stats
            assert stats['min'] == pytest.approx(float(np.min(pool)), rel=1e-12)
            assert stats['max'] == pytest.approx(float(np.max(pool)), rel=1e-12)
            assert stats['min'] <= float(q.magnitude) <= stats['max']

    def test_serialization_round_trip_of_parameter_samples(self, noisy_pr_result):
        result = noisy_pr_result
        d = result.to_dict()
        restored = FitResult.from_dict(d)

        assert restored.parameter_samples is not None
        for k, arr in result.parameter_samples.items():
            np.testing.assert_array_equal(restored.parameter_samples[k], arr)
        assert restored.n_passing == result.n_passing

    def test_average_mode_also_populates_parameter_samples(self):
        """`fit_assay` in average mode must also retain the passing-trial
        pool so box-whisker plots work in both modes."""
        from core.pipeline.fit_pipeline import fit_assay

        ms = _ida_ms(n_replicas=3, noise_frac=0.02, seed=206)
        assay = ms.to_assay(IDAAssay, conditions=_ida_conditions(), use_average=True)
        result = fit_assay(
            assay,
            FitConfig(n_trials=N_TRIALS, custom_bounds=GDA_IDA_RECOVERY_BOUNDS),
        )

        assert result.parameter_samples is not None
        for k, arr in result.parameter_samples.items():
            assert len(arr) == result.n_passing
        # Reported value = the representative real trial, not a per-param median.
        ridx = result.representative_index
        for k, q in result.parameters.items():
            assert float(q.magnitude) == pytest.approx(float(result.parameter_samples[k][ridx]), rel=1e-12)
