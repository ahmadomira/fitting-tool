"""Style roundtrip test: PlotStyleWidget → PlotWidget.apply_style."""

import pytest

pytest.importorskip('PyQt6')
pytest.importorskip('pyqtgraph')


def test_style_signal_updates_plot_style(qapp, minimal_plot_data):
    from gui.plotting.plot_style import PlotStyleWidget
    from gui.plotting.plot_widget import PlotWidget

    pw = PlotWidget()
    pw.update_plot(minimal_plot_data)

    style_widget = PlotStyleWidget()
    style_widget.style_changed.connect(pw.apply_style)

    initial_style = style_widget.current_style()
    initial_size = initial_style['data_points']['size']

    # Change a parameter directly via the Parameter tree
    params = style_widget._params
    new_size = initial_size + 2
    params['Replicas', 'Size'] = new_size

    # apply_style should have been called via signal; check _style updated
    assert pw._style['data_points']['size'] == new_size


def test_apply_style_with_no_items_does_not_raise(qapp):
    """apply_style on empty plot should not raise."""
    from gui.plotting.plot_style import PlotStyleWidget
    from gui.plotting.plot_widget import PlotWidget

    pw = PlotWidget()
    style_widget = PlotStyleWidget()
    pw.apply_style(style_widget.current_style())  # must not raise
