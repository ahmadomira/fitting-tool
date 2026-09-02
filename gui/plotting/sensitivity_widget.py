"""Ka input-sensitivity visualisation (pyqtgraph).

Renders the output of :func:`core.pipeline.sensitivity.run_sensitivity`:

- **JOINT / OAT** — a grid of Ka histograms (rows = perturbed input, columns =
  Ka key), each annotated with the median Ka, the coefficient of variation, and
  a marker line at the unperturbed baseline Ka.
- **HEATMAP** — a viridis image per Ka key over the two swept input offsets,
  with a colour bar and a hover crosshair reading out Ka; failed-fit cells are
  transparent.

The widget mirrors :class:`gui.plotting.distribution_widget.DistributionWidget`:
a :class:`QStackedLayout` with a placeholder page, a progress page, and a
results page; ``apply_style`` stores the style dict and re-renders (honouring
``style['distribution']['ka_scale']`` for log₁₀-vs-linear Ka axes).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGridLayout,
    QLabel,
    QProgressBar,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from core.assays.registry import ASSAY_REGISTRY, AssayType
from core.pipeline.sensitivity import TITRANT, SensitivityMode, SensitivityResult
from gui.plotting.colors import BACKGROUND_COLOR, FOREGROUND_COLOR
from gui.plotting.labels import fmt_param, fmt_unit_html
from gui.plotting.plot_style import DEFAULT_STYLE
from gui.plotting.plot_widget import ScientificAxisItem, _format_exponent_unicode
from gui.widgets.assay_conditions import ASSAY_CONDITIONS

# Fill/edge for histogram bars.
_BAR_RGB = (31, 119, 180)


def _wire_x_exponent(plot_item: pg.PlotItem, axis: ScientificAxisItem, base_label: str) -> None:
    """Re-apply *plot_item*'s x-label with a ``×10ⁿ`` suffix when *axis* factors one out.

    ``ScientificAxisItem`` fires ``on_exponent_changed`` on every deferred
    ``tickStrings`` pass; funnelling the label update through that callback is
    the only reliable way to append the exponent (reading ``axis.exponent``
    synchronously after a range change is a documented no-no).
    """
    plot_item.setLabel('bottom', base_label)

    def _on_exp(exp: Optional[int]) -> None:
        if exp is not None:
            plot_item.setLabel('bottom', f'{base_label}  (×10{_format_exponent_unicode(exp)})')
        else:
            plot_item.setLabel('bottom', base_label)

    axis.on_exponent_changed = _on_exp


class SensitivityWidget(QWidget):
    """Visualise a :class:`SensitivityResult` (histograms or a Ka heatmap).

    Public API mirrors :class:`DistributionWidget`: :meth:`update_result`
    dispatches on the result mode, :meth:`apply_style` restyles in place,
    :meth:`clear` resets to the placeholder, and :meth:`show_running` /
    :meth:`set_progress` drive the progress page during a run.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(
            '<qt>Spread of the fitted association constant when the input '
            'concentrations (and any known K<sub>a</sub>) are perturbed within '
            'the ±% you set.<br>Histograms rank each input; the heatmap maps Ka '
            'over two swept inputs.</qt>'
        )
        self._style: dict = dict(DEFAULT_STYLE)
        self._result: SensitivityResult | None = None
        # References kept alive for the heatmap hover handler.
        self._heatmap_glw: pg.GraphicsLayoutWidget | None = None
        self._hover_targets: list[dict] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._stack_host = QWidget()
        self._stack = QStackedLayout(self._stack_host)

        self._placeholder = QLabel('Configure ±Δ and run a sensitivity analysis.')
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet('color: rgba(0,0,0,0.35); font-size: 14px;')
        self._stack.addWidget(self._placeholder)

        self._progress_page = QWidget()
        prog_layout = QVBoxLayout(self._progress_page)
        prog_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prog_layout.setSpacing(10)
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimumWidth(240)
        self._progress_bar.setMaximumWidth(360)
        self._progress_label = QLabel('Running… cancel from the Sensitivity panel.')
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setStyleSheet('color: rgba(0,0,0,0.55); font-size: 13px;')
        prog_layout.addWidget(self._progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)
        prog_layout.addWidget(self._progress_label)
        self._stack.addWidget(self._progress_page)

        self._results_host = QWidget()
        self._results_layout = QVBoxLayout(self._results_host)
        self._results_layout.setContentsMargins(0, 0, 0, 0)
        self._results_layout.setSpacing(4)
        self._stack.addWidget(self._results_host)

        self._stack.setCurrentWidget(self._placeholder)
        outer.addWidget(self._stack_host, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_result(self, result: SensitivityResult) -> None:
        """Store *result* and render it (dispatch on ``result.mode``)."""
        self._result = result
        self._render()

    def apply_style(self, style: dict) -> None:
        """Adopt a new style dict and re-render the current result if any."""
        self._style = style
        if self._result is not None:
            self._render()

    def clear(self) -> None:
        """Reset to the empty placeholder page."""
        self._result = None
        self._clear_results()
        self._stack.setCurrentWidget(self._placeholder)

    def show_running(self) -> None:
        """Switch to the progress page (indeterminate until the first update)."""
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setValue(0)
        self._stack.setCurrentWidget(self._progress_page)

    def set_progress(self, done: int, total: int) -> None:
        """Advance the progress bar to *done* / *total* fits."""
        if total > 0:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(done)
        else:
            self._progress_bar.setRange(0, 0)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_results(self) -> None:
        """Tear down the results page (plots + heatmap hover wiring)."""
        if self._heatmap_glw is not None:
            scene = self._heatmap_glw.scene()
            if scene is not None:
                try:
                    scene.sigMouseMoved.disconnect()
                except (TypeError, RuntimeError):
                    pass
        self._heatmap_glw = None
        self._hover_targets = []
        while self._results_layout.count():
            item = self._results_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _render(self) -> None:
        """Rebuild the results page from ``self._result``."""
        self._clear_results()
        result = self._result
        if result is None:
            self._stack.setCurrentWidget(self._placeholder)
            return
        if result.mode is SensitivityMode.HEATMAP:
            self._render_heatmap(result)
        else:
            self._render_histograms(result)
        self._stack.setCurrentWidget(self._results_host)

    def _assay_type(self, result: SensitivityResult) -> Optional[AssayType]:
        try:
            return AssayType[result.assay_type]
        except KeyError:
            return None

    def _ka_unit_html(self, assay_type: Optional[AssayType], ka_key: str) -> str:
        """HTML unit for a Ka key (e.g. ``M⁻¹``), empty when unknown."""
        if assay_type is None:
            return ''
        return fmt_unit_html(ASSAY_REGISTRY[assay_type].units.get(ka_key, ''))

    def _input_label(self, assay_type: Optional[AssayType], key: str) -> str:
        """HTML display label for a perturbable input key.

        ``TITRANT`` → the assay's titrant species (registry ``x_label``); a
        condition key → its :class:`ConditionField` label; otherwise a
        parameter label (``fmt_param`` falls back to the raw key).
        """
        if key == TITRANT:
            return ASSAY_REGISTRY[assay_type].x_label if assay_type is not None else 'Titrant'
        if assay_type is not None:
            for field in ASSAY_CONDITIONS[assay_type][1]:
                if field.key == key:
                    return field.label
        return fmt_param(key)

    # ---- JOINT / OAT histograms --------------------------------------

    def _render_histograms(self, result: SensitivityResult) -> None:
        assay_type = self._assay_type(result)
        use_log = self._style.get('distribution', {}).get('ka_scale', 'log₁₀') == 'log₁₀'
        histograms = result.histograms or {}
        labels = list(histograms.keys())
        ka_keys = result.ka_keys

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        for r, label in enumerate(labels):
            row_label = 'Joint' if label == 'joint' else self._input_label(assay_type, label)
            for c, ka_key in enumerate(ka_keys):
                samples = histograms[label].get(ka_key)
                pw = self._make_histogram(
                    samples=samples,
                    baseline=result.baseline_ka.get(ka_key),
                    ka_key=ka_key,
                    row_label=row_label,
                    use_log=use_log,
                    unit_html=self._ka_unit_html(assay_type, ka_key),
                )
                grid.addWidget(pw, r, c)
                grid.setColumnStretch(c, 1)
            grid.setRowStretch(r, 1)

        self._results_layout.addWidget(container)

    def _make_histogram(
        self,
        *,
        samples: Optional[np.ndarray],
        baseline: Optional[float],
        ka_key: str,
        row_label: str,
        use_log: bool,
        unit_html: str,
    ) -> pg.PlotWidget:
        bottom_axis = ScientificAxisItem(orientation='bottom', use_exponent=not use_log)
        bottom_axis.enableAutoSIPrefix(False)
        pw = pg.PlotWidget(background=BACKGROUND_COLOR, axisItems={'bottom': bottom_axis})
        pw.setMinimumSize(220, 200)
        plot = pw.getPlotItem()
        plot.getViewBox().setDefaultPadding(0.05)

        ka_label = fmt_param(ka_key)
        if use_log:
            x_base_label = f'log₁₀({ka_label})'
        else:
            x_base_label = f'{ka_label} [{unit_html}]' if unit_html else ka_label
        plot.setLabel('left', 'Count')
        _wire_x_exponent(plot, bottom_axis, x_base_label)

        arr = None if samples is None else np.asarray(samples, dtype=float)
        if arr is None or arr.size == 0:
            plot.setTitle(f'{row_label} — no successful fits', size=f'{self._title_size()}pt', bold=True)
            self._apply_axis_style(plot)
            return pw

        median = float(np.median(arr))
        cv = 100.0 * float(np.std(arr)) / median if median else float('nan')

        positive = arr[arr > 0] if use_log else arr
        disp = np.log10(positive) if use_log else positive
        if disp.size:
            nbins = int(np.clip(round(np.sqrt(disp.size)), 10, 40))
            counts, edges = np.histogram(disp, bins=nbins)
            bars = pg.BarGraphItem(
                x0=edges[:-1],
                x1=edges[1:],
                y0=0,
                y1=counts,
                brush=pg.mkBrush(*_BAR_RGB, 160),
                pen=pg.mkPen(*_BAR_RGB, 220),
            )
            plot.addItem(bars)

        if baseline is not None and (not use_log or baseline > 0):
            base_x = float(np.log10(baseline)) if use_log else float(baseline)
            median_c = self._style.get('distribution', {}).get('median_line_color', (220, 0, 0, 255))
            median_w = self._style.get('distribution', {}).get('median_line_width', 2.5)
            line = pg.InfiniteLine(
                pos=base_x,
                angle=90,
                pen=pg.mkPen(median_c, width=median_w, style=Qt.PenStyle.DashLine),
                label='baseline',
                labelOpts={'position': 0.92, 'color': median_c},
            )
            plot.addItem(line)

        val_html = f'{median:.3g} {unit_html}'.strip()
        plot.setTitle(
            f'{row_label} — median {val_html} · CV {cv:.1f}%',
            size=f'{self._title_size()}pt',
            bold=True,
        )
        self._apply_axis_style(plot)
        plot.enableAutoRange()
        return pw

    # ---- HEATMAP ------------------------------------------------------

    def _render_heatmap(self, result: SensitivityResult) -> None:
        assay_type = self._assay_type(result)
        cfg = result.config
        x_offsets = np.asarray(result.x_offsets_pct, dtype=float)
        y_offsets = np.asarray(result.y_offsets_pct, dtype=float)
        cmap = pg.colormap.get('viridis')

        glw = pg.GraphicsLayoutWidget()
        glw.setBackground(BACKGROUND_COLOR)
        glw.setMinimumHeight(320)
        self._heatmap_glw = glw
        self._hover_targets = []

        x0, x_w, step_x = self._axis_extent(x_offsets)
        y0, y_h, step_y = self._axis_extent(y_offsets)
        x_label = f'Δ {self._input_label(assay_type, cfg.x_key)} [%]'
        y_label = f'Δ {self._input_label(assay_type, cfg.y_key)} [%]'

        for i, ka_key in enumerate(result.ka_keys):
            grid = np.asarray(result.ka_grid[ka_key], dtype=float)
            finite = np.isfinite(grid)
            if finite.any():
                lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
            else:
                lo, hi = 0.0, 1.0
            span = (hi - lo) or 1.0
            norm = np.where(finite, (grid - lo) / span, 0.0)
            rgba = cmap.map(norm, mode='byte')
            rgba[~finite] = (0, 0, 0, 0)

            plot = glw.addPlot(row=0, col=2 * i)
            plot.setLabel('bottom', x_label)
            plot.setLabel('left', y_label)
            plot.setTitle(fmt_param(ka_key), size=f'{self._title_size()}pt', bold=True)
            plot.getViewBox().setDefaultPadding(0.02)

            img = pg.ImageItem()
            img.setImage(rgba, axisOrder='row-major')
            img.setRect(QRectF(x0, y0, x_w, y_h))
            plot.addItem(img)
            self._apply_axis_style(plot)

            cbar = pg.ColorBarItem(
                values=(lo, hi),
                colorMap=cmap,
                label=self._ka_unit_html(assay_type, ka_key),
                interactive=False,
            )
            glw.addItem(cbar, row=0, col=2 * i + 1)

            pen = pg.mkPen(FOREGROUND_COLOR, width=1, style=Qt.PenStyle.DashLine)
            vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
            vline.setVisible(False)
            hline.setVisible(False)
            plot.addItem(vline, ignoreBounds=True)
            plot.addItem(hline, ignoreBounds=True)
            readout = pg.TextItem(anchor=(0, 1), color=FOREGROUND_COLOR, fill=pg.mkBrush(255, 255, 255, 200))
            readout.setVisible(False)
            plot.addItem(readout, ignoreBounds=True)

            self._hover_targets.append(
                {
                    'plot': plot,
                    'vb': plot.getViewBox(),
                    'grid': grid,
                    'x_offsets': x_offsets,
                    'y_offsets': y_offsets,
                    'step_x': step_x,
                    'step_y': step_y,
                    'vline': vline,
                    'hline': hline,
                    'readout': readout,
                    'ka_key': ka_key,
                    'unit_html': self._ka_unit_html(assay_type, ka_key),
                }
            )

        glw.scene().sigMouseMoved.connect(self._on_heatmap_hover)
        self._results_layout.addWidget(glw)

    @staticmethod
    def _axis_extent(offsets: np.ndarray) -> tuple[float, float, float]:
        """Return ``(origin, length, step)`` mapping cell centres to *offsets*.

        ``setRect`` maps the pixel grid ``[0..n]`` onto ``[origin, origin+length]``;
        extending half a step past each end centres each cell on its offset.
        """
        if offsets.size >= 2:
            step = float(offsets[1] - offsets[0])
        else:
            step = 1.0
        origin = float(offsets[0]) - step / 2 if offsets.size else -0.5
        length = float(offsets[-1] - offsets[0]) + step if offsets.size else 1.0
        return origin, length, step

    def _on_heatmap_hover(self, scene_pos) -> None:
        """Update the crosshair + Ka readout for the heatmap under the cursor."""
        for target in self._hover_targets:
            plot = target['plot']
            if not plot.sceneBoundingRect().contains(scene_pos):
                target['vline'].setVisible(False)
                target['hline'].setVisible(False)
                target['readout'].setVisible(False)
                continue
            view_pt = target['vb'].mapSceneToView(scene_pos)
            x_offsets = target['x_offsets']
            y_offsets = target['y_offsets']
            step_x = target['step_x'] or 1.0
            step_y = target['step_y'] or 1.0
            ix = int(np.clip(round((view_pt.x() - x_offsets[0]) / step_x), 0, x_offsets.size - 1))
            iy = int(np.clip(round((view_pt.y() - y_offsets[0]) / step_y), 0, y_offsets.size - 1))
            ka = target['grid'][iy, ix]
            cx, cy = float(x_offsets[ix]), float(y_offsets[iy])
            target['vline'].setPos(cx)
            target['hline'].setPos(cy)
            target['vline'].setVisible(True)
            target['hline'].setVisible(True)
            if np.isfinite(ka):
                ka_label = fmt_param(target['ka_key'])
                target['readout'].setHtml(f'{ka_label} = {ka:.3g} {target["unit_html"]}'.strip())
            else:
                target['readout'].setHtml('fit failed')
            target['readout'].setPos(cx, cy)
            target['readout'].setVisible(True)

    # ------------------------------------------------------------------
    # Styling helpers
    # ------------------------------------------------------------------

    def _title_size(self) -> int:
        return self._style.get('distribution', {}).get('title_font_size', 16)

    def _apply_axis_style(self, plot_item: pg.PlotItem) -> None:
        """Apply distribution fonts/pens to a subplot's axes."""
        dist = self._style.get('distribution', {})
        label_font = QFont()
        label_font.setPointSize(dist.get('label_font_size', 14))
        tick_font = QFont()
        tick_font.setPointSize(dist.get('tick_font_size', 12))
        for axis_name in ('left', 'bottom'):
            axis = plot_item.getAxis(axis_name)
            axis.label.setFont(label_font)
            axis.setTickFont(tick_font)
            axis.setPen(pg.mkPen(FOREGROUND_COLOR, width=1))
            axis.setTextPen(pg.mkPen(FOREGROUND_COLOR))
