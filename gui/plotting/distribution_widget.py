"""Box-whisker distribution plots for fitted parameters (pyqtgraph)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QStackedLayout, QVBoxLayout, QWidget

from core.assays.registry import ASSAY_REGISTRY, AssayType
from core.pipeline.fit_pipeline import FitResult
from gui.plotting.colors import BACKGROUND_COLOR, FOREGROUND_COLOR, PALETTES, REPLICA_PALETTE, rgba
from gui.plotting.labels import fmt_param, fmt_param_plain, fmt_unit_html
from gui.plotting.plot_style import DEFAULT_STYLE
from gui.plotting.plot_widget import ScientificAxisItem, _format_exponent_unicode
from gui.widgets.replica_panel import _display_label

_BOX_HALF = 0.3
_CAP_HALF = 0.2
_JITTER_HALF = 0.25

# Fit-quality metrics shown as extra distribution subplots.
# key -> (title, uses_signal_unit): RMSE carries the assay's signal (y) unit,
# sourced from the registry at render time; R² is dimensionless.
_QUALITY_METRICS = {'rmse': ('RMSE', True), 'r_squared': ('R²', False)}

# Fallback per-cell logical pixel size when the live widget hasn't
# been shown yet (e.g. headless tests, dialog opened before the
# distributions tab was visible). The export then renders at this
# default; the user can resize the live widget and re-export to get
# different proportions.
_FALLBACK_CELL_W = 320
_FALLBACK_CELL_H = 380


def _log10_positive(values: np.ndarray) -> np.ndarray:
    """``log10`` of a strictly-positive pool, matching ``ensemble.describe_log10``.

    A log-scale parameter (Ka) is bounded > 0; a non-positive sample is a real
    defect, so raise rather than emit a silent ``-inf``/``nan`` point that would
    show as a gap in the distribution plot (the summary table's ``describe_log10``
    already raises on the same condition — keep the two consistent).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size and np.min(arr) <= 0:
        raise ValueError('Cannot log10-transform a non-positive value for the log-scale distribution.')
    return np.log10(arr)


def _box_stats(data: np.ndarray) -> Dict[str, float]:
    """Compute box-whisker statistics for a 1-D array."""
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    whisker_lo = max(data.min(), q1 - 1.5 * iqr)
    whisker_hi = min(data.max(), q3 + 1.5 * iqr)
    return {
        'q1': float(q1),
        'median': float(median),
        'q3': float(q3),
        'whisker_lo': float(whisker_lo),
        'whisker_hi': float(whisker_hi),
    }


def _refresh_axis_label_with_exponent(plot_item: pg.PlotItem, exp: int | None) -> None:
    """Re-apply the y-axis label, appending ×10ⁿ when an exponent is set.

    The base label (without the exponent suffix) is stored on the
    PlotItem as ``_y_base_label`` by :meth:`DistributionWidget._populate_subplot`.
    ``ScientificAxisItem`` fires its ``on_exponent_changed`` callback on
    every tickStrings() pass; that callback funnels through this
    helper, so live and export PlotItems share one reactive update
    path.
    """
    base = getattr(plot_item, '_y_base_label', None)
    if not base:
        return
    if exp is not None:
        exp_str = _format_exponent_unicode(exp)
        plot_item.setLabel('left', f'{base}  (×10{exp_str})')
    else:
        plot_item.setLabel('left', base)


def _wire_exponent_callback(plot_item: pg.PlotItem, axis: ScientificAxisItem) -> None:
    """Wire *axis* to update *plot_item*'s y-label when its exponent changes."""

    def _on_exp(exp: int | None) -> None:
        _refresh_axis_label_with_exponent(plot_item, exp)

    axis.on_exponent_changed = _on_exp


class DistributionWidget(QWidget):
    """Side-by-side box-whisker plots of fitted parameter distributions.

    One pyqtgraph subplot per parameter, arranged horizontally.

    **Per-replica mode**: x-axis = replica ID (A, B, C, …).  Each replica
    gets its own box-whisker at its x position.  A red pooled-median line
    spans all positions.

    **Average mode**: x-axis labeled "Pool of Fits" with no tick marks.
    One box-whisker at x=0 for the single optimizer pool.

    Two extra subplots show the **RMSE** and **R²** distributions across the
    valid fits. A gold ring marks the current representative in every subplot;
    clicking any point (parameter or quality) reports that fit as the
    representative (emits :attr:`representative_selected`). Each subplot can be
    shown/hidden via the checkbox row beneath the plots.

    Signals
    -------
    representative_selected(int)
        Emitted with a pool index when the user clicks a fit-quality point.
    """

    representative_selected = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(
            '<qt>Distribution of each parameter and of fit quality (RMSE, R²) '
            'across all valid fits.<br>Click any point to report that fit as the '
            'representative; the gold ring marks the current one.<br>Show or hide '
            'each plot with the checkboxes below.</qt>'
        )
        self._style: dict = dict(DEFAULT_STYLE)
        self._plots: list[pg.PlotWidget] = []
        self._result: FitResult | None = None
        self._param_keys: list[str] = []
        # Visibility toggles (issue 7): ordered keys = param keys then quality keys.
        self._all_keys: list[str] = []
        # RMSE starts hidden: it is monotone in R², which is already shown, so it
        # adds a panel without adding information. Re-enable via its checkbox.
        self._hidden_keys: set[str] = {'rmse'}
        self._plot_by_key: dict[str, pg.PlotWidget] = {}
        self._checkboxes: dict[str, QCheckBox] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._stack_host = QWidget()
        self._stack = QStackedLayout(self._stack_host)

        self._placeholder = QLabel('Run a fit to see parameter and fit-quality distributions.')
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet('color: rgba(0,0,0,0.35); font-size: 14px;')
        self._stack.addWidget(self._placeholder)

        self._plot_container = QWidget()
        self._plot_layout = QHBoxLayout(self._plot_container)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_layout.setSpacing(4)
        # Trailing stretch absorbs leftover width so a few plots left-align at
        # their max width instead of over-stretching across the whole pane.
        self._plot_layout.addStretch(1)
        self._stack.addWidget(self._plot_container)

        self._all_hidden = QLabel('All distributions hidden — re-enable a checkbox below.')
        self._all_hidden.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._all_hidden.setStyleSheet('color: rgba(0,0,0,0.35); font-size: 14px;')
        self._stack.addWidget(self._all_hidden)

        self._stack.setCurrentWidget(self._placeholder)
        outer.addWidget(self._stack_host, stretch=1)

        # Row of show/hide checkboxes beneath the plots (issue 7).
        self._toggle_bar_host = QWidget()
        self._toggle_bar = QHBoxLayout(self._toggle_bar_host)
        self._toggle_bar.setContentsMargins(4, 0, 4, 0)
        self._toggle_bar.setSpacing(10)
        self._toggle_bar_host.setVisible(False)
        outer.addWidget(self._toggle_bar_host)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_result(self, result: FitResult) -> None:
        """Redraw all subplots from a FitResult's parameter_samples.

        One subplot per parameter, then RMSE and R² fit-quality subplots
        (when ``quality_samples`` is present). Only the parameter subplots are
        reported by :meth:`param_keys` / exported; the quality subplots are a
        live, clickable diagnostic for picking a representative fit.
        """
        self._result = result

        if result.parameter_samples is None or not result.parameters:
            self._stack.setCurrentWidget(self._placeholder)
            self._toggle_bar_host.setVisible(False)
            return

        param_keys = list(result.parameter_samples.keys())
        self._param_keys = list(param_keys)

        quality_keys = []
        if result.quality_samples:
            quality_keys = [k for k in ('rmse', 'r_squared') if k in result.quality_samples]

        all_keys = param_keys + quality_keys
        self._all_keys = all_keys

        self._rebuild_subplots(len(all_keys))
        self._plot_by_key = {}

        for i, key in enumerate(param_keys):
            pw = self._plots[i]
            plot_item = pw.getPlotItem()
            self._clear_subplot(plot_item)
            self._populate_subplot(plot_item, key=key, param_idx=i, add_legend=(i == 0))
            self._plot_by_key[key] = pw

        for j, metric_key in enumerate(quality_keys):
            pw = self._plots[len(param_keys) + j]
            plot_item = pw.getPlotItem()
            self._clear_subplot(plot_item)
            self._populate_quality_subplot(plot_item, metric_key=metric_key)
            self._plot_by_key[metric_key] = pw

        # Drop stale hidden keys (e.g. assay switch) so a new key-set is all-visible,
        # then apply visibility and rebuild the toggle bar.
        self._hidden_keys &= set(all_keys)
        for key, pw in self._plot_by_key.items():
            pw.setVisible(key not in self._hidden_keys)
        self._sync_toggle_bar(all_keys)
        self._update_stack_page()

    # ------------------------------------------------------------------
    # Visibility toggles (issue 7)
    # ------------------------------------------------------------------

    def _checkbox_label(self, key: str) -> str:
        """Plain-text label for a subplot's toggle checkbox."""
        if key in _QUALITY_METRICS:
            return _QUALITY_METRICS[key][0]
        return fmt_param_plain(key)

    def _sync_toggle_bar(self, keys: list[str]) -> None:
        """Rebuild the checkbox row when the key-set changes; else reflect state."""
        if list(self._checkboxes.keys()) != list(keys):
            while self._toggle_bar.count():
                item = self._toggle_bar.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
            self._checkboxes = {}
            self._toggle_bar.addWidget(QLabel('Show:'))
            for key in keys:
                cb = QCheckBox(self._checkbox_label(key))
                cb.setChecked(key not in self._hidden_keys)
                cb.toggled.connect(lambda on, k=key: self._on_toggle(k, on))
                self._toggle_bar.addWidget(cb)
                self._checkboxes[key] = cb
            self._toggle_bar.addStretch(1)
        else:
            for key, cb in self._checkboxes.items():
                cb.blockSignals(True)
                cb.setChecked(key not in self._hidden_keys)
                cb.blockSignals(False)
        self._toggle_bar_host.setVisible(True)

    def _on_toggle(self, key: str, checked: bool) -> None:
        """Show/hide one subplot without re-rendering (no jitter/stat recompute)."""
        if checked:
            self._hidden_keys.discard(key)
        else:
            self._hidden_keys.add(key)
        pw = self._plot_by_key.get(key)
        if pw is not None:
            pw.setVisible(checked)
        self._update_stack_page()

    def _update_stack_page(self) -> None:
        """Pick the stacked page: placeholder, all-hidden notice, or the plots."""
        if self._result is None or not self._all_keys:
            self._stack.setCurrentWidget(self._placeholder)
        elif all(k in self._hidden_keys for k in self._all_keys):
            self._stack.setCurrentWidget(self._all_hidden)
        else:
            self._stack.setCurrentWidget(self._plot_container)

    def apply_style(self, style: dict) -> None:
        """Update visual style (fonts, palette) from PlotStyleWidget."""
        self._style = style
        if self._result is not None:
            self.update_result(self._result)

    def clear(self) -> None:
        """Reset to empty placeholder."""
        self._result = None
        self._param_keys = []
        self._all_keys = []
        for pw in self._plots:
            self._clear_subplot(pw.getPlotItem())
        self._toggle_bar_host.setVisible(False)
        self._stack.setCurrentWidget(self._placeholder)

    def param_keys(self) -> list[str]:
        """Parameter keys of the currently displayed subplots, in order."""
        return list(self._param_keys)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def live_per_cell_size(self) -> tuple[int, int]:
        """Pixel size of a single live distribution subplot.

        This is the source of truth for export rendering: the transient
        export layout uses cells of exactly this size so the
        font:cell, line-width:cell, and marker:cell ratios all match
        the GUI by construction. Falls back to a sensible default when
        the live widget hasn't been laid out yet (headless tests,
        hidden tab).
        """
        if self._plots:
            w = self._plots[0].width()
            h = self._plots[0].height()
            if w >= 50 and h >= 50:
                return w, h
        return _FALLBACK_CELL_W, _FALLBACK_CELL_H

    def derive_height_in(self, *, width_in: float, rows: int, cols: int) -> float:
        """Height in inches that preserves live cell aspect for the chosen layout.

        With the export figure rendered at ``cell_w × cols`` wide and
        ``cell_h × rows`` tall (where ``cell_*`` come from
        :meth:`live_per_cell_size`), the figure aspect is fixed. Given
        a user-chosen ``width_in``, the matching ``height_in`` is the
        only choice that keeps each export cell at the same aspect as
        the live GUI cell.
        """
        cw, ch = self.live_per_cell_size()
        return width_in * (rows * ch) / max(cols * cw, 1)

    def build_composite_layout(
        self,
        keys: list[str],
        rows: int,
        cols: int,
    ) -> pg.GraphicsLayoutWidget:
        """Build a transient ``GraphicsLayoutWidget`` whose cells match the live GUI.

        Each cell is sized to the live distribution subplot's pixel
        dimensions, so when the caller hands the resulting scene to
        :func:`gui.plotting.export.export_scene` the painter scales
        every element uniformly. Font:cell, line:cell, and marker:cell
        ratios are preserved by construction — no per-element scaling.

        Raises
        ------
        ValueError
            If no distributions are loaded, no keys match, or
            ``rows * cols`` is less than the number of selected keys.
        """
        if not self._param_keys:
            raise ValueError('No distributions to render. Run a fit first.')

        indices = [self._param_keys.index(k) for k in keys if k in self._param_keys]
        if not indices:
            raise ValueError('No matching distributions selected.')
        if rows * cols < len(indices):
            raise ValueError(f'Layout {rows}x{cols} cannot fit {len(indices)} subplots.')

        glw = pg.GraphicsLayoutWidget()
        glw.setBackground(BACKGROUND_COLOR)

        for i, key_idx in enumerate(indices):
            r, c = divmod(i, cols)
            left_axis = ScientificAxisItem(orientation='left')
            left_axis.enableAutoSIPrefix(False)
            plot_item = glw.addPlot(row=r, col=c, axisItems={'left': left_axis})
            _wire_exponent_callback(plot_item, left_axis)
            plot_item.getViewBox().setDefaultPadding(0.05)
            plot_item.getAxis('bottom').enableAutoSIPrefix(False)

            key = self._param_keys[key_idx]
            self._populate_subplot(plot_item, key=key, param_idx=key_idx, add_legend=(i == 0), interactive=False)

        return glw

    def save_plot(
        self,
        *,
        keys: list[str],
        rows: int,
        cols: int,
        width_in: float,
        dpi: int,
        path: str,
        format: str = 'png',
    ) -> None:
        """Save the distributions composite as PNG or SVG.

        The export figure's aspect is fixed by ``rows × cols`` of live
        cells, so the user picks only ``width_in``; the output height
        in pixels is the painter-scaled live cell height × rows.

        PyQtGraph's ``ImageExporter`` / ``SVGExporter`` render the
        transient scene at this fixed aspect into the requested output
        width — font:cell ratios match the GUI exactly.
        """
        from gui.plotting.export import (
            export_scene,
            prepare_widget_for_offscreen_render,
        )

        fmt = format.lower()
        if fmt not in ('png', 'svg'):
            raise ValueError(f"Unsupported format: '{format}'. Use 'png' or 'svg'.")

        cell_w, cell_h = self.live_per_cell_size()
        glw = self.build_composite_layout(keys, rows, cols)
        try:
            logical_w = cell_w * cols
            logical_h = cell_h * rows
            prepare_widget_for_offscreen_render(glw, logical_w, logical_h)
            if fmt == 'png':
                target_w = round(width_in * dpi)
                export_scene(glw.scene(), path, width_px=target_w)
            else:
                # SVG is vector; native scene size suffices.
                export_scene(glw.scene(), path)
        finally:
            glw.deleteLater()

    @staticmethod
    def auto_layout(n: int) -> tuple[int, int]:
        """Default rows×cols for *n* subplots (matches legacy Auto behaviour)."""
        import math

        if n <= 1:
            return 1, max(1, n)
        if n == 2:
            return 1, 2
        if n == 3:
            return 1, 3
        if n == 4:
            return 2, 2
        cols = 2
        return math.ceil(n / cols), cols

    # ------------------------------------------------------------------
    # Subplot construction
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_subplot(plot_item: pg.PlotItem) -> None:
        """Remove all data items AND the legend from a PlotItem."""
        plot_item.clear()
        legend = plot_item.legend
        if legend is not None:
            if legend.scene() is not None:
                legend.scene().removeItem(legend)
            plot_item.legend = None

    def _rebuild_subplots(self, n: int) -> None:
        """Ensure we have exactly *n* PlotWidget instances with ScientificAxisItem."""
        while len(self._plots) > n:
            pw = self._plots.pop()
            self._plot_layout.removeWidget(pw)
            pw.deleteLater()
        while len(self._plots) < n:
            left_axis = ScientificAxisItem(orientation='left')
            left_axis.enableAutoSIPrefix(False)

            pw = pg.PlotWidget(
                background=BACKGROUND_COLOR,
                axisItems={'left': left_axis},
            )
            # Share width equally (large stretch beats the trailing stretch when
            # there is room to share) but cap each so few plots don't over-stretch.
            pw.setMinimumWidth(170)
            pw.setMaximumWidth(380)
            plot_item = pw.getPlotItem()
            plot_item.getViewBox().setDefaultPadding(0.05)
            plot_item.getAxis('bottom').enableAutoSIPrefix(False)

            _wire_exponent_callback(plot_item, left_axis)

            self._plots.append(pw)
            self._plot_layout.insertWidget(self._plot_layout.count() - 1, pw, 1000)

    def _populate_subplot(
        self,
        plot_item: pg.PlotItem,
        *,
        key: str,
        param_idx: int,
        add_legend: bool = False,
        interactive: bool = True,
    ) -> None:
        """Populate one PlotItem with the box-whisker distribution for *key*.

        Single source of truth for distribution subplot rendering: used
        both by :meth:`update_result` on the live widget and by
        :meth:`build_composite_layout` on the export-time transient
        ``GraphicsLayoutWidget``. The PlotItem's y-axis base label is
        stored on the item itself (``_y_base_label``) so
        :func:`_refresh_axis_label_with_exponent` can reactively
        re-apply it when ``ScientificAxisItem`` factors out an
        exponent.
        """
        result = self._result
        if result is None or result.parameter_samples is None or not result.parameters:
            return

        assay_type = self._lookup_assay_type(result.assay_type)
        log_keys: set[str] = set()
        units: dict[str, str] = {}
        if assay_type is not None:
            meta = ASSAY_REGISTRY[assay_type]
            log_keys = set(meta.log_scale_keys)
            units = dict(meta.units)

        ka_scale = self._style.get('distribution', {}).get('ka_scale', 'log₁₀')
        palette_name = self._style['data_points'].get('palette', 'Default (Tab10)')
        palette = PALETTES.get(palette_name, REPLICA_PALETTE)

        replica_samples = self._extract_replica_samples(result)
        n_replicas = len(replica_samples) if replica_samples else 0

        if n_replicas > 0:
            x_positions = list(range(n_replicas))
            x_tick_labels = [_display_label(i) for i in range(n_replicas)]
            x_axis_label = 'Replica'
        else:
            x_positions = [0]
            x_tick_labels = []
            x_axis_label = 'Pool of Fits'

        pool = result.parameter_samples[key]
        use_log = (key in log_keys) and ka_scale == 'log₁₀'
        values = _log10_positive(pool) if use_log else pool
        pool_stats = _box_stats(values)

        # Boxes: one per replica in per-replica mode, one pooled in average mode
        if replica_samples:
            for r_idx, r_samp in enumerate(replica_samples):
                r_vals = r_samp.get(key)
                if r_vals is None:
                    continue
                r_display = _log10_positive(r_vals) if use_log else r_vals
                r_stats = _box_stats(r_display)
                color = palette[r_idx % len(palette)]
                self._draw_box(plot_item, r_stats, float(x_positions[r_idx]), color)
        else:
            self._draw_box(plot_item, pool_stats, 0.0)

        coords = self._draw_strip(
            plot_item, values, replica_samples, key, param_idx, palette, use_log, x_positions, interactive=interactive
        )
        self._draw_median_line(plot_item, pool_stats['median'], x_positions)

        if replica_samples:
            self._draw_replica_medians(plot_item, replica_samples, key, palette, use_log, x_positions)

        self._draw_selection_highlight(plot_item, coords)

        # Y-axis label — base form; ScientificAxisItem appends ×10ⁿ via callback.
        unit_str = units.get(key, '')
        label = fmt_param(key)
        if use_log:
            base_label = f'log₁₀({label})'
        else:
            base_label = f'{label} [{fmt_unit_html(unit_str)}]' if unit_str else label
        plot_item._y_base_label = base_label
        # Reset cached exponent so the next tickStrings() pass re-fires the callback.
        left_axis = plot_item.getAxis('left')
        if hasattr(left_axis, 'exponent'):
            left_axis.exponent = None
        plot_item.setLabel('left', base_label)

        # X-axis ticks and label
        bottom_ax = plot_item.getAxis('bottom')
        bottom_ax.setTicks([[(pos, lbl) for pos, lbl in zip(x_positions, x_tick_labels)]])
        plot_item.setLabel('bottom', x_axis_label)

        # Title — mirror the log-wrap so title + axis stay consistent
        title_size = self._style.get('distribution', {}).get('title_font_size', 16)
        title_text = f'log₁₀({label})' if use_log else label
        plot_item.setTitle(title_text, size=f'{title_size}pt', bold=True)

        self._apply_axis_style(plot_item)

        x_pad = 0.8
        plot_item.setXRange(min(x_positions) - x_pad, max(x_positions) + x_pad, padding=0)
        plot_item.enableAutoRange(axis='y')

        if add_legend:
            self._add_legend(plot_item, replica_samples is not None)

    def _populate_quality_subplot(self, plot_item: pg.PlotItem, *, metric_key: str) -> None:
        """Populate one subplot with the RMSE or R² distribution over the pool.

        The strip plot is clickable: clicking a point emits
        :attr:`representative_selected` with that fit's pool index.
        """
        result = self._result
        if result is None or result.quality_samples is None or metric_key not in result.quality_samples:
            return

        values = np.asarray(result.quality_samples[metric_key], dtype=float)
        if values.size == 0:
            return
        stats = _box_stats(values)

        self._draw_box(plot_item, stats, 0.0)
        coords = self._draw_quality_strip(plot_item, values)
        self._draw_median_line(plot_item, stats['median'], [0])
        self._draw_selection_highlight(plot_item, coords)

        # RMSE label unit = the assay's signal unit from the registry (the same
        # 'a.u.' string the main plot uses). Pass it plainly — Pint's HTML
        # formatter mis-parses the dotted 'a.u.' alias (renders 'u a').
        title, uses_signal_unit = _QUALITY_METRICS.get(metric_key, (metric_key, False))
        unit = ''
        if uses_signal_unit:
            assay_type = self._lookup_assay_type(result.assay_type)
            if assay_type is not None:
                unit = ASSAY_REGISTRY[assay_type].y_unit
        base_label = f'{title} [{unit}]' if unit else title
        plot_item._y_base_label = base_label
        left_axis = plot_item.getAxis('left')
        if hasattr(left_axis, 'exponent'):
            left_axis.exponent = None
        plot_item.setLabel('left', base_label)

        plot_item.getAxis('bottom').setTicks([[]])
        plot_item.setLabel('bottom', 'Pool of Fits')

        title_size = self._style.get('distribution', {}).get('title_font_size', 16)
        plot_item.setTitle(title, size=f'{title_size}pt', bold=True)

        self._apply_axis_style(plot_item)
        plot_item.setXRange(-0.8, 0.8, padding=0)
        plot_item.enableAutoRange(axis='y')

    def _draw_quality_strip(self, plot_item: pg.PlotItem, values: np.ndarray) -> dict[int, tuple[float, float]]:
        """Jittered, clickable strip — clicking a point selects that fit.

        Returns the pool-index → ``(x, y)`` coordinate map so the selection
        ring is drawn on the exact dot it points to.
        """
        rng = np.random.default_rng(7)
        n = values.size
        jitter = rng.uniform(-_JITTER_HALF, _JITTER_HALF, size=n)
        scatter = pg.ScatterPlotItem(
            x=jitter,
            y=values,
            size=7,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 100, 130),
            data=list(range(n)),
        )
        scatter.setToolTip('Click a point to report that fit as the representative.')
        scatter.sigClicked.connect(self._on_point_clicked)
        plot_item.addItem(scatter)
        return {g: (float(jitter[g]), float(values[g])) for g in range(n)}

    def _on_point_clicked(self, _scatter, points, *_args) -> None:
        """Emit representative_selected with the front-most clicked fit's index.

        ``sigClicked`` hands us a NumPy array of ``SpotItem`` (front-most
        first). Use an explicit length check — ``if not points`` on an
        ndarray raises the ambiguous-truth ``ValueError``.
        """
        if len(points) == 0:
            return
        idx = points[0].data()
        if idx is None:
            return
        self.representative_selected.emit(int(idx))

    def _draw_selection_highlight(self, plot_item: pg.PlotItem, coords: dict[int, tuple[float, float]]) -> None:
        """Ring the current representative fit at its exact drawn coordinate.

        ``coords`` maps a pool index to the same ``(x, y)`` handed to that
        subplot's scatter, so the ring lands on the selected dot for free —
        correct across raw/log axes, per-replica columns, and jitter.
        """
        result = self._result
        if result is None or result.representative_index is None:
            return
        xy = coords.get(result.representative_index)
        if xy is None:
            return
        ring = pg.ScatterPlotItem(
            x=[xy[0]],
            y=[xy[1]],
            symbol='o',
            size=16,
            pen=pg.mkPen(255, 215, 0, 255, width=2.5),
            brush=None,
        )
        ring.setZValue(1000)  # above strips, boxes, and the median line
        plot_item.addItem(ring)

    # ------------------------------------------------------------------
    # Drawing primitives (PlotItem-based)
    # ------------------------------------------------------------------

    def _draw_box(
        self,
        plot_item: pg.PlotItem,
        stats: Dict[str, float],
        x_center: float,
        color: Tuple[int, int, int] | None = None,
    ) -> None:
        """Draw the box (Q1–Q3), whiskers, and caps at x_center."""
        q1, q3 = stats['q1'], stats['q3']
        wlo, whi = stats['whisker_lo'], stats['whisker_hi']

        dist = self._style.get('distribution', {})
        box_w = dist.get('box_border_width', 1.5)
        wh_w = dist.get('whisker_width', 1.2)

        if color is not None:
            box_pen = pg.mkPen(rgba(color, 200), width=box_w)
            box_brush = pg.mkBrush(rgba(color, 40))
            line_pen = pg.mkPen(rgba(color, 180), width=wh_w)
        else:
            box_pen = pg.mkPen(FOREGROUND_COLOR, width=box_w)
            box_brush = pg.mkBrush(200, 200, 200, 80)
            line_pen = pg.mkPen(FOREGROUND_COLOR, width=wh_w)

        box = pg.BarGraphItem(
            x0=[x_center - _BOX_HALF],
            y0=[q1],
            x1=[x_center + _BOX_HALF],
            y1=[q3],
            pen=box_pen,
            brush=box_brush,
        )
        plot_item.addItem(box)

        for wy in (wlo, whi):
            whisker = pg.PlotCurveItem(
                x=[x_center, x_center],
                y=[q1 if wy == wlo else q3, wy],
                pen=line_pen,
            )
            plot_item.addItem(whisker)
            cap = pg.PlotCurveItem(
                x=[x_center - _CAP_HALF, x_center + _CAP_HALF],
                y=[wy, wy],
                pen=line_pen,
            )
            plot_item.addItem(cap)

    def _draw_strip(
        self,
        plot_item: pg.PlotItem,
        values: np.ndarray,
        replica_samples: Optional[List[Dict[str, np.ndarray]]],
        key: str,
        param_idx: int,
        palette: list,
        use_log: bool,
        x_positions: list[int],
        *,
        interactive: bool = True,
    ) -> dict[int, tuple[float, float]]:
        """Jittered strip plot with each replica at its own x position.

        Returns the pool-index → ``(x, y)`` coordinate map (the exact numbers
        handed to the scatter) so the selection ring lands on the right dot,
        correct across raw/log axes, per-replica columns, and jitter. Points
        carry their global pool index as ``data``; when *interactive*,
        clicking one reports that fit as the representative.
        """
        rng = np.random.default_rng(42 + param_idx)
        coords: dict[int, tuple[float, float]] = {}

        if replica_samples:
            offset = 0
            for r_idx, r_samp in enumerate(replica_samples):
                # Unlike the box/median loops, the strip cannot tolerate a
                # missing key by skipping: that would desync the pooled index
                # from the drawn points and misplace the selection ring. Fail
                # loudly with a clear message instead of a bare KeyError.
                r_vals = r_samp.get(key)
                if r_vals is None:
                    raise ValueError(
                        f'Replica {r_idx} has no samples for parameter {key!r}; '
                        f'cannot align the selection strip with the pooled fits.'
                    )
                display = _log10_positive(r_vals) if use_log else r_vals
                n = len(display)
                x_base = x_positions[r_idx] if r_idx < len(x_positions) else r_idx
                jitter = rng.uniform(-_JITTER_HALF, _JITTER_HALF, size=n)
                xs = np.full(n, x_base) + jitter
                gidx = list(range(offset, offset + n))
                for g, x, y in zip(gidx, xs, display):
                    coords[g] = (float(x), float(y))
                scatter = pg.ScatterPlotItem(
                    x=xs,
                    y=display,
                    size=5,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(rgba(palette[r_idx % len(palette)], 120)),
                    data=gidx,
                )
                if interactive:
                    scatter.sigClicked.connect(self._on_point_clicked)
                plot_item.addItem(scatter)
                offset += n
            if offset != len(values):
                raise ValueError(f'strip point count {offset} != pool size {len(values)} for {key!r}')
        else:
            n = len(values)
            jitter = rng.uniform(-_JITTER_HALF, _JITTER_HALF, size=n)
            xs = np.zeros(n) + jitter
            for g in range(n):
                coords[g] = (float(xs[g]), float(values[g]))
            scatter = pg.ScatterPlotItem(
                x=xs,
                y=values,
                size=5,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(100, 100, 100, 100),
                data=list(range(n)),
            )
            if interactive:
                scatter.sigClicked.connect(self._on_point_clicked)
            plot_item.addItem(scatter)
        return coords

    def _draw_median_line(self, plot_item: pg.PlotItem, median: float, x_positions: list[int]) -> None:
        """Solid horizontal line at the pool median, spanning all positions."""
        x_lo = min(x_positions) - 0.5
        x_hi = max(x_positions) + 0.5
        dist = self._style.get('distribution', {})
        width = dist.get('median_line_width', 2.5)
        color = dist.get('median_line_color', (220, 0, 0, 255))
        line = pg.PlotCurveItem(
            x=[x_lo, x_hi],
            y=[median, median],
            pen=pg.mkPen(color, width=width),
        )
        plot_item.addItem(line)

    def _draw_replica_medians(
        self,
        plot_item: pg.PlotItem,
        replica_samples: List[Dict[str, np.ndarray]],
        key: str,
        palette: list,
        use_log: bool,
        x_positions: list[int],
    ) -> None:
        """Diamond markers at each replica's median value, at that replica's x."""
        marker_size = self._style.get('distribution', {}).get('replica_median_size', 10)
        for r_idx, r_samp in enumerate(replica_samples):
            r_vals = r_samp.get(key)
            if r_vals is None or len(r_vals) == 0:
                continue
            med = float(np.median(_log10_positive(r_vals) if use_log else r_vals))
            x_pos = x_positions[r_idx] if r_idx < len(x_positions) else r_idx
            color = palette[r_idx % len(palette)]
            marker = pg.ScatterPlotItem(
                x=[float(x_pos)],
                y=[med],
                symbol='d',
                size=marker_size,
                pen=pg.mkPen(FOREGROUND_COLOR, width=1),
                brush=pg.mkBrush(rgba(color, 220)),
            )
            plot_item.addItem(marker)

    def _add_legend(self, plot_item: pg.PlotItem, has_replicas: bool) -> None:
        """Add a legend to *plot_item* explaining visual elements."""
        dist = self._style.get('distribution', {})
        label_size = dist.get('label_font_size', 14)
        median_w = dist.get('median_line_width', 2.5)
        median_c = dist.get('median_line_color', (220, 0, 0, 255))
        marker_size = dist.get('replica_median_size', 10)

        legend = plot_item.addLegend(offset=(10, 10))
        legend.setLabelTextSize(f'{label_size}pt')

        median_item = pg.PlotCurveItem(pen=pg.mkPen(median_c, width=median_w))
        legend.addItem(median_item, 'Pool median')

        if has_replicas:
            diamond_item = pg.ScatterPlotItem(
                symbol='d',
                size=marker_size,
                pen=pg.mkPen(FOREGROUND_COLOR, width=1),
                brush=pg.mkBrush(150, 150, 150, 220),
            )
            legend.addItem(diamond_item, 'Replica median')

    def _apply_axis_style(self, plot_item: pg.PlotItem) -> None:
        """Apply distribution fonts to axes. Title is handled at setTitle()."""
        dist = self._style.get('distribution', {})
        label_size = dist.get('label_font_size', 14)
        tick_size = dist.get('tick_font_size', 12)

        label_font = QFont()
        label_font.setPointSize(label_size)
        tick_font = QFont()
        tick_font.setPointSize(tick_size)

        for axis_name in ('left', 'bottom'):
            axis = plot_item.getAxis(axis_name)
            axis.label.setFont(label_font)
            axis.setTickFont(tick_font)
            axis.setPen(pg.mkPen(FOREGROUND_COLOR, width=1))
            axis.setTextPen(pg.mkPen(FOREGROUND_COLOR))

    def _extract_replica_samples(
        self,
        result: FitResult,
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """Return per-replica parameter_samples if available."""
        if result.replica_fits is None:
            return None
        out = []
        for rf in result.replica_fits:
            if rf.parameter_samples is not None:
                out.append(rf.parameter_samples)
        return out or None

    @staticmethod
    def _lookup_assay_type(name: str) -> Optional[AssayType]:
        try:
            return AssayType[name]
        except KeyError:
            return None
