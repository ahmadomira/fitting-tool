"""Widget tests for DistributionWidget — selection, highlight, toggles.

Requires a QApplication (the ``qapp`` fixture).
"""

import numpy as np
import pytest

pytest.importorskip('PyQt6')


@pytest.fixture
def pooled_result():
    """A FitResult carrying parameter + quality pools (average mode)."""
    from core.pipeline.fit_pipeline import FitResult
    from core.units import Q_

    n = 8
    ps = {
        'Ka_guest': np.linspace(1e6, 3e6, n),
        'I0': np.linspace(-1.0, 1.0, n),
        'I_dye_free': np.linspace(4.9e7, 5.1e7, n),
        'I_dye_bound': np.linspace(2.9e8, 3.1e8, n),
    }
    qs = {'rmse': np.linspace(0.01, 0.05, n), 'r_squared': np.linspace(0.999, 0.99, n)}
    rep = 2
    return FitResult(
        parameters={k: Q_(float(v[rep]), 'dimensionless') for k, v in ps.items()},
        rmse=0.02,
        r_squared=0.996,
        n_passing=n,
        n_total=n,
        x_fit=Q_(np.zeros(3), 'M'),
        y_fit=Q_(np.zeros(3), 'au'),
        assay_type='IDA',
        model_name='equilibrium_4param',
        parameter_samples=ps,
        quality_samples=qs,
        representative_index=rep,
    )


class _Spot:
    """Stand-in for a pyqtgraph SpotItem carrying a pool index."""

    def __init__(self, idx):
        self._idx = idx

    def data(self):
        return self._idx


def test_multi_hit_click_reports_front_most(qapp, pooled_result):
    """Regression: sigClicked hands an ndarray of ≥2 SpotItems; the handler
    must not raise (the old ``if not points`` did) and reports the front one."""
    from gui.plotting.distribution_widget import DistributionWidget

    w = DistributionWidget()
    w.update_result(pooled_result)
    captured = []
    w.representative_selected.connect(captured.append)

    w._on_point_clicked(None, np.array([_Spot(5), _Spot(3)], dtype=object))
    assert captured == [5]  # front-most (index 0 of the array)


def test_empty_click_does_not_raise(qapp, pooled_result):
    from gui.plotting.distribution_widget import DistributionWidget

    w = DistributionWidget()
    w.update_result(pooled_result)
    w._on_point_clicked(None, np.array([], dtype=object))  # must not raise


def test_strip_coords_map_the_drawn_points(qapp):
    """_draw_strip returns pool-index → (x, y) matching the values it drew."""
    import pyqtgraph as pg

    from gui.plotting.distribution_widget import DistributionWidget

    w = DistributionWidget()
    plot = pg.PlotItem()
    values = np.array([1.0, 2.0, 3.0, 4.0])
    coords = w._draw_strip(plot, values, None, 'Ka_guest', 0, [], False, [0], interactive=False)
    assert set(coords.keys()) == set(range(4))
    for i, v in enumerate(values):
        assert coords[i][1] == pytest.approx(v)  # y is the pool value (raw)


def test_selection_ring_sits_on_representative_log_value(qapp, pooled_result):
    """Issue 2: the ring lands on the representative's *displayed* value —
    log₁₀(Ka) on a log-scaled Ka subplot, not the raw Ka (the old bug)."""
    import pyqtgraph as pg

    from gui.plotting.distribution_widget import DistributionWidget

    w = DistributionWidget()
    w.update_result(pooled_result)

    ka_plot = w._plot_by_key['Ka_guest'].getPlotItem()
    rings = [it for it in ka_plot.items if isinstance(it, pg.ScatterPlotItem) and it.zValue() == 1000]
    assert len(rings) == 1
    _, ys = rings[0].getData()
    rep = pooled_result.representative_index
    expected = np.log10(pooled_result.parameter_samples['Ka_guest'][rep])
    assert float(ys[0]) == pytest.approx(expected)


def test_toggle_hides_and_shows_subplots(qapp, pooled_result):
    from gui.plotting.distribution_widget import DistributionWidget

    w = DistributionWidget()
    w.update_result(pooled_result)

    # isHidden() reflects the explicit visibility flag (isVisible() is also
    # False here because the top-level window is never shown in a headless test).
    w._on_toggle('Ka_guest', False)
    assert w._plot_by_key['Ka_guest'].isHidden()
    assert 'Ka_guest' in w._hidden_keys
    assert w._stack.currentWidget() is w._plot_container  # others still shown

    for k in list(w._all_keys):
        w._on_toggle(k, False)
    assert w._stack.currentWidget() is w._all_hidden  # explicit all-hidden page

    w._on_toggle('Ka_guest', True)
    assert not w._plot_by_key['Ka_guest'].isHidden()
    assert 'Ka_guest' not in w._hidden_keys
    assert w._stack.currentWidget() is w._plot_container
