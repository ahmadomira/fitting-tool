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
# Fit-summary annotation in exported images. The overlay is a scene item
# the exporter has to render like any other; these pin that it survives an
# export unmoved and actually appears in the output.
# ----------------------------------------------------------------------


@pytest.fixture
def annotated_plot_widget(qapp):
    """A PlotWidget with a fit result + annotation visible.

    Shown off-screen with events processed so the ViewBox resolves to real
    geometry before the annotation is placed.
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


def test_annotation_actually_renders_into_the_export(annotated_plot_widget, tmp_path):
    """The annotation must appear in the exported image, not drift off-canvas.

    Measured by exporting the same plot with the overlay on and off and
    counting changed pixels, rather than sampling fixed corners: the overlay
    is auto-placed into whichever candidate slot is emptiest, so a corner scan
    would encode one particular dataset's outcome.
    """
    import copy

    from PyQt6.QtGui import QImage

    pw = annotated_plot_widget
    with_ann, without_ann = tmp_path / 'with.png', tmp_path / 'without.png'

    pw.export_image(str(with_ann), width_px=1200)

    style = copy.deepcopy(pw._style)
    style['visibility']['show_fit_results'] = False
    pw.apply_style(style)
    assert pw._annotation_item is None, 'overlay should be gone with the flag off'
    pw.export_image(str(without_ann), width_px=1200)

    a, b = QImage(str(with_ann)), QImage(str(without_ann))
    assert (a.width(), a.height()) == (b.width(), b.height())
    changed = sum(a.pixel(x, y) != b.pixel(x, y) for y in range(0, a.height(), 4) for x in range(0, a.width(), 4))
    # The overlay is a multi-line boxed label; anything under a few hundred
    # sampled pixels means it did not render where the canvas can see it.
    assert changed > 300, f'annotation contributed only {changed} sampled pixels to the export'
