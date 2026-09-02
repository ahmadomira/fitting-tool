"""Tests for prepare_plot_data() — the MeasurementSet → plot dict bridge."""

import numpy as np

from core.data_processing.measurement_set import MeasurementSet
from core.data_processing.plotting import prepare_plot_data
from core.pipeline.fit_pipeline import FitResult
from core.units import Q_


def _simple_ms(n_replicas=3, n_points=5):
    conc = np.array([1.0, 2.0, 3.0, 4.0, 5.0][:n_points])
    signals = np.stack([conc * (i + 1) for i in range(n_replicas)])
    return MeasurementSet(
        concentrations=conc,
        signals=signals,
        replica_ids=tuple(f'r{i}' for i in range(n_replicas)),
    )


def _simple_fit_result():
    x = np.linspace(0, 5, 20)
    return FitResult(
        parameters={'slope': Q_(1.0, 'au/M')},
        rmse=0.01,
        r_squared=0.99,
        n_passing=1,
        n_total=1,
        x_fit=Q_(x, 'M'),
        y_fit=Q_(x * 1.05, 'au'),
        assay_type='DYE_ALONE',
        model_name='linear',
    )


class TestPrepPlotDataReplicas:
    def test_active_replicas_only(self):
        ms = _simple_ms()
        ms.set_active('r1', False)
        data = prepare_plot_data(ms)

        ids = [rid for rid, _ in data['active_replicas']]
        assert len(ids) == 2
        assert 'r0' in ids and 'r2' in ids
        assert 'r1' not in ids
        assert data['dropped_replicas'] == []

    def test_show_dropped_includes_inactive(self):
        ms = _simple_ms()
        ms.set_active('r1', False)
        data = prepare_plot_data(ms, show_dropped=True)

        assert len(data['dropped_replicas']) == 1
        assert data['dropped_replicas'][0][0] == 'r1'

    def test_average_matches_active(self):
        ms = _simple_ms()
        ms.set_active('r2', False)
        data = prepare_plot_data(ms)

        expected_avg = ms.average_signal(active_only=True)
        np.testing.assert_allclose(data['average'], expected_avg)

    def test_concentrations_match_ms(self):
        ms = _simple_ms()
        data = prepare_plot_data(ms)
        np.testing.assert_array_equal(data['concentrations'], ms.concentrations)


class TestPrepPlotDataFits:
    def test_fit_results_included(self):
        ms = _simple_ms()
        fr = _simple_fit_result()
        data = prepare_plot_data(ms, fit_results=[fr])

        assert len(data['fits']) == 1
        assert data['fits'][0]['label'] == 'Best Fit'
        np.testing.assert_array_equal(data['fits'][0]['x'], fr.x_fit.magnitude)
        assert data['fits'][0]['id'] == fr.id

    def test_no_fit_results_none(self):
        ms = _simple_ms()
        data = prepare_plot_data(ms, fit_results=None)
        assert data['fits'] == []
