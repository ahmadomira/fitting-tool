"""Tests for the consolidated image-export pipeline.

Verifies:
  * single-plot PNG export honours the requested width
  * single-plot SVG export writes a parseable vector file
  * composite distributions PNG hits exact ``round(width_in * dpi) x
    round(height_in * dpi)`` pixel dimensions (the core promise of the
    new native-export path)
  * composite distributions SVG writes a vector file
  * ``build_composite_layout`` produces one PlotItem per requested key
  * unsupported export formats raise ``ValueError``
"""

import numpy as np
import pytest

pytest.importorskip('PyQt6')
pytest.importorskip('pyqtgraph')


@pytest.fixture
def simple_plot_widget(qapp):
    from gui.plotting.plot_widget import PlotWidget

    x = np.linspace(0, 1e-4, 20)
    pw = PlotWidget()
    pw.update_plot(
        {
            'concentrations': x,
            'active_replicas': [('r1', x + 0.01), ('r2', x * 0.95 + 0.015)],
            'dropped_replicas': [],
            'average': x + 0.0125,
            'fits': [{'x': x, 'y': x + 0.02, 'label': 'fit', 'id': 'abc'}],
        }
    )
    return pw


@pytest.fixture
def fitted_dist_widget(qapp):
    """A DistributionWidget loaded with a FitResult that has parameter_samples."""
    from core.pipeline.fit_pipeline import FitResult
    from gui.plotting.distribution_widget import DistributionWidget

    rng = np.random.default_rng(0)
    samples = {
        'Ka_guest': rng.lognormal(15, 0.2, size=200),
        'I0': rng.normal(100, 5, size=200),
        'I_dye_free': rng.normal(5e4, 2e3, size=200),
        'I_dye_bound': rng.normal(8e4, 3e3, size=200),
    }
    x = np.linspace(0, 1e-4, 20)
    result = FitResult(
        parameters={k: float(np.median(v)) for k, v in samples.items()},
        rmse=0.005,
        r_squared=0.998,
        n_passing=200,
        n_total=200,
        x_fit=x,
        y_fit=x,
        assay_type='GDA',
        model_name='equilibrium_4param',
        parameter_samples=samples,
    )
    widget = DistributionWidget()
    widget.update_result(result)
    return widget


# ----------------------------------------------------------------------
# Single-plot export
# ----------------------------------------------------------------------


def test_plot_widget_png_honours_requested_width(simple_plot_widget, tmp_path):
    from PyQt6.QtGui import QImage

    path = tmp_path / 'plot.png'
    simple_plot_widget.export_image(str(path), width_px=1200)

    assert path.exists()
    img = QImage(str(path))
    assert img.width() == 1200
    assert img.height() > 0


def test_plot_widget_svg_writes_vector_file(simple_plot_widget, tmp_path):
    path = tmp_path / 'plot.svg'
    simple_plot_widget.export_image(str(path))

    assert path.exists()
    assert path.stat().st_size > 0
    content = path.read_text(encoding='utf-8')
    assert '<svg' in content


def test_plot_widget_rejects_unknown_extension(simple_plot_widget, tmp_path):
    with pytest.raises(ValueError, match='Unsupported export format'):
        simple_plot_widget.export_image(str(tmp_path / 'x.bmp'))


# ----------------------------------------------------------------------
# Composite distributions export — the core re-architecture promise
# ----------------------------------------------------------------------


def test_distribution_png_width_is_exact_height_derived(fitted_dist_widget, tmp_path):
    """The composite PNG's width matches the request exactly; height comes
    from the live cell aspect × the chosen layout (which preserves the
    per-cell font:cell ratio of the GUI)."""
    from PyQt6.QtGui import QImage

    path = tmp_path / 'dist.png'
    fitted_dist_widget.save_plot(
        keys=['Ka_guest', 'I0'],
        rows=1,
        cols=2,
        width_in=4.0,
        dpi=200,
        path=str(path),
        format='png',
    )

    assert path.exists()
    img = QImage(str(path))
    assert img.width() == 800  # 4.0 in × 200 DPI, exact
    # Height = output_w × (rows × cell_h) / (cols × cell_w). With the
    # headless fallback cell of 320 × 380:
    #   img.height = 800 × (1 × 380) / (2 × 320) = 475
    cell_w, cell_h = fitted_dist_widget.live_per_cell_size()
    expected = round(800 * (1 * cell_h) / (2 * cell_w))
    assert abs(img.height() - expected) <= 1


def test_distribution_png_grid_layout(fitted_dist_widget, tmp_path):
    """A 2x2 layout with 3 selected keys exports as a valid composite PNG."""
    from PyQt6.QtGui import QImage

    path = tmp_path / 'dist_grid.png'
    fitted_dist_widget.save_plot(
        keys=['Ka_guest', 'I0', 'I_dye_free'],
        rows=2,
        cols=2,
        width_in=6.0,
        dpi=150,
        path=str(path),
        format='png',
    )

    assert path.exists()
    img = QImage(str(path))
    assert img.width() == 900  # 6.0 in × 150 DPI
    # Height: 900 × (2 × cell_h) / (2 × cell_w) — depends on live cell aspect
    cell_w, cell_h = fitted_dist_widget.live_per_cell_size()
    expected = round(900 * (2 * cell_h) / (2 * cell_w))
    assert abs(img.height() - expected) <= 1


def test_distribution_svg_writes_vector_file(fitted_dist_widget, tmp_path):
    path = tmp_path / 'dist.svg'
    fitted_dist_widget.save_plot(
        keys=['Ka_guest'],
        rows=1,
        cols=1,
        width_in=4.0,
        dpi=200,
        path=str(path),
        format='svg',
    )

    assert path.exists()
    assert path.stat().st_size > 0
    assert '<svg' in path.read_text(encoding='utf-8')


def test_save_plot_rejects_unknown_format(fitted_dist_widget, tmp_path):
    with pytest.raises(ValueError, match='Unsupported format'):
        fitted_dist_widget.save_plot(
            keys=['Ka_guest'],
            rows=1,
            cols=1,
            width_in=4.0,
            dpi=200,
            path=str(tmp_path / 'x.bmp'),
            format='bmp',
        )


def test_derive_height_in_matches_live_cell_aspect(fitted_dist_widget):
    """``derive_height_in`` returns a height that preserves the live cell aspect."""
    cell_w, cell_h = fitted_dist_widget.live_per_cell_size()
    # 2 rows × 3 cols, width 9 in → expected aspect = (3*cell_w)/(2*cell_h)
    h = fitted_dist_widget.derive_height_in(width_in=9.0, rows=2, cols=3)
    expected = 9.0 * (2 * cell_h) / (3 * cell_w)
    assert abs(h - expected) < 1e-9


# ----------------------------------------------------------------------
# Composite layout building
# ----------------------------------------------------------------------


def test_build_composite_layout_rejects_oversubscribed_grid(fitted_dist_widget):
    with pytest.raises(ValueError, match='cannot fit'):
        fitted_dist_widget.build_composite_layout(
            keys=['Ka_guest', 'I0', 'I_dye_free', 'I_dye_bound'],
            rows=1,
            cols=2,
        )


def test_build_composite_layout_rejects_empty_selection(fitted_dist_widget):
    with pytest.raises(ValueError, match='No matching'):
        fitted_dist_widget.build_composite_layout(
            keys=['not_a_real_key'],
            rows=1,
            cols=1,
        )


# ----------------------------------------------------------------------
# Annotation reparenting — regression tests for the export-time
# TextItem fix. Without reparenting, the annotation gets multiplied by
# the ViewBox's data-coords-to-scene-pixels scaling (~1e7 for typical
# molar X axes) and drifts off-canvas.
# ----------------------------------------------------------------------


@pytest.fixture
def annotated_plot_widget(qapp):
    """A PlotWidget with a fit result + annotation visible.

    The widget is shown off-screen and events are processed so the
    ViewBox's view rect resolves to real data bounds — without this,
    the annotation positions at the default (0, 0…1, 1) rect and lands
    on top of the plot's data instead of a corner.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from core.pipeline.fit_pipeline import FitResult
    from gui.plotting.plot_widget import PlotWidget

    x = np.linspace(0, 1e-4, 20)
    pw = PlotWidget()
    pw.update_plot(
        {
            'concentrations': x,
            'active_replicas': [('r1', x * 1.1 + 0.01)],
            'dropped_replicas': [],
            'average': x * 1.05 + 0.012,
            'fits': [{'x': x, 'y': x * 1.05 + 0.012, 'label': 'GDA fit', 'id': 'abc'}],
        }
    )
    result = FitResult(
        parameters={'Ka_guest': 1e6, 'I0': 100.0, 'I_dye_free': 5e4, 'I_dye_bound': 8e4},
        rmse=0.005,
        r_squared=0.998,
        n_passing=87,
        n_total=100,
        x_fit=x,
        y_fit=x * 1.05 + 0.012,
        assay_type='GDA',
        model_name='equilibrium_4param',
    )
    pw.set_fit_results([result])
    pw.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    pw.resize(800, 600)
    pw.show()
    QApplication.processEvents()
    # Rebuild the annotation now that the view rect is real.
    pw.set_fit_results([result])
    return pw


def test_annotation_state_restored_after_export(annotated_plot_widget, tmp_path):
    """The annotation TextItem's parent and position survive a round-trip export."""
    annotation = annotated_plot_widget._annotation_item
    assert annotation is not None, 'annotation should be visible'

    saved_parent = annotation.parentItem()
    saved_pos = annotation.pos()

    annotated_plot_widget.export_image(str(tmp_path / 'annot.png'), width_px=1200)

    assert annotation.parentItem() is saved_parent
    assert annotation.pos() == saved_pos


def test_live_per_cell_size_uses_live_widget_when_shown(qapp):
    """When the live widget has been laid out, ``live_per_cell_size`` reflects it."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from core.pipeline.fit_pipeline import FitResult
    from gui.plotting.distribution_widget import DistributionWidget

    rng = np.random.default_rng(0)
    samples = {
        'Ka_guest': rng.lognormal(15, 0.2, size=50),
        'I0': rng.normal(100, 5, size=50),
    }
    x = np.linspace(0, 1e-4, 10)
    result = FitResult(
        parameters={k: float(np.median(v)) for k, v in samples.items()},
        rmse=0.005,
        r_squared=0.998,
        n_passing=50,
        n_total=50,
        x_fit=x,
        y_fit=x,
        assay_type='GDA',
        model_name='equilibrium_4param',
        parameter_samples=samples,
    )
    dw = DistributionWidget()
    dw.update_result(result)
    dw.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    dw.resize(600, 400)
    dw.show()
    QApplication.processEvents()

    w, h = dw.live_per_cell_size()
    # 2 plots in a HBox in a 600-wide widget → each ~300 px wide
    assert 200 < w < 400, f'expected ~300 px per cell, got {w}'
    assert 200 < h < 500, f'expected ~400 px per cell, got {h}'


def test_live_per_cell_size_falls_back_when_widget_not_shown(qapp):
    """When the widget hasn't been shown, fallback dimensions are used.

    The fallback is what keeps headless / hidden-tab exports working
    without crashing — the saved file has predictable proportions
    even if it doesn't track a non-existent live widget.
    """
    from gui.plotting.distribution_widget import _FALLBACK_CELL_H, _FALLBACK_CELL_W, DistributionWidget

    dw = DistributionWidget()
    w, h = dw.live_per_cell_size()
    assert (w, h) == (_FALLBACK_CELL_W, _FALLBACK_CELL_H)


def test_annotated_export_has_non_trivial_pixels_in_corner(annotated_plot_widget, tmp_path):
    """The annotation occupies a corner; that corner must not be all background.

    Catches the regression where the TextItem ends up at ~1e9 px and
    leaves the canvas — a blank corner is the visible symptom.
    """
    from PyQt6.QtGui import QImage

    path = tmp_path / 'annot.png'
    annotated_plot_widget.export_image(str(path), width_px=1200)
    img = QImage(str(path))

    w, h = img.width(), img.height()
    patch_w, patch_h = 240, 120

    def _dark_count(x0, y0):
        n = 0
        for yi in range(y0, y0 + patch_h, 4):
            for xi in range(x0, x0 + patch_w, 4):
                c = img.pixelColor(xi, yi)
                if c.red() < 200 and c.green() < 200 and c.blue() < 200:
                    n += 1
        return n

    corner_counts = {
        'top-left': _dark_count(5, 5),
        'top-right': _dark_count(w - patch_w - 5, 5),
        'bottom-left': _dark_count(5, h - patch_h - 5),
        'bottom-right': _dark_count(w - patch_w - 5, h - patch_h - 5),
    }
    # The annotation lives in one of the four corners. If it rendered,
    # at least one corner has many dark pixels (text + border). If it
    # drifted off-canvas (regression), every corner is near-blank.
    best = max(corner_counts.values())
    assert best > 80, (
        f'no corner contains the annotation (corner_counts={corner_counts}) — annotation likely off-canvas'
    )
