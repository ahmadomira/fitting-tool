"""Read-only display widget for FitResult statistics."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from core.pipeline.fit_pipeline import FitResult, summarize_parameters
from core.units import Q_
from gui.plotting.labels import fmt_param, fmt_unit_html
from gui.widgets.info_button import InfoGroupBox

_PARAMS_HELP_HTML = """
<h3>Fitted Parameters &mdash; columns</h3>

<p><b>Estimate</b> &mdash; the representative fit: one actual fit from the
valid pool (highest R&sup2;), the one drawn on the plot. It is a point
estimate among a distribution, not a proven optimum.</p>

<p><b>Median &plusmn; MAD</b> and <b>Mean &plusmn; SD</b> &mdash; the centre and
spread of each parameter across <i>all</i> valid fits. <b>MAD</b> is the median
absolute deviation, median(|x<sub>i</sub> &minus; median|): a robust spread that
ignores outlier fits (reported unscaled). <b>SD</b> is the sample standard
deviation (N&minus;1) &mdash; the ordinary spread, pulled by outliers and
assuming a roughly symmetric distribution. (The <i>median</i> absolute deviation
is not the <i>mean</i> absolute deviation mean(|x<sub>i</sub> &minus; mean|); we
use the median form because it pairs with the robust median.) None of these
change the Estimate or the curve.</p>

<p><b>68% Range</b> &mdash; the interval [P16, P84] holding the central 68% of
the fits. For a bell-shaped distribution this equals mean&nbsp;&plusmn;&nbsp;1
SD, but it is read straight from the fits, so it stays correct &mdash; and
reveals any skew &mdash; even when they are not bell-shaped. <b>Min-Max Range</b>
is the full extent of the fitted values &mdash; this is the range quoted beside
the Estimate in the plot annotation and the exported reports.</p>

<p><b>log<sub>10</sub>(K<sub>a</sub>) row</b> &mdash; for each association
constant, a second row reports log<sub>10</sub>(K<sub>a</sub>). Its statistics
come from the per-fit log<sub>10</sub> values directly &mdash; never from the
log of a K<sub>a</sub> spread, which would be meaningless.</p>

<p><b>What the spread means (and doesn&rsquo;t).</b> These numbers describe how
<i>reproducibly the optimiser converges</i> across its many starting points on
the same data &mdash; not the experimental uncertainty of your measurement. A
tight spread means the fit reliably lands on the same value; it does <b>not</b>
by itself mean the parameter is known to that precision. Real error bars come
from repeating the experiment (replicates) or resampling it (bootstrap)
&mdash; that is what <i>per-replica</i> fitting gives you. The wide spread on the
signal coefficients (I<sub>0</sub>, I<sub>D</sub>, I<sub>HD</sub>) is the
clearest case: it is not uncertainty, it is the model telling you those
coefficients are not individually determined &mdash; only K<sub>a</sub> is.</p>
"""


_FIT_QUALITY_HELP_HTML = """
<h3>Fit Quality</h3>

<p><b>RMSE</b> (root mean squared error) &mdash; average distance between
the fitted curve and the data points, in signal units. Lower is better.
Compare it to the noise level in your data: if RMSE is similar to the
noise floor, the fit is explaining the data about as well as possible.</p>

<p><b>R&sup2;</b> (coefficient of determination) &mdash; fraction of the
data variance explained by the fit, from 0 to 1. Values above 0.99 are
typical for clean fluorescence titrations; below 0.90 suggests the model
does not capture the data shape well.</p>

<p><b>Fits passing</b> &mdash; how many of the multi-start trials cleared
the minimum&nbsp;R&sup2; floor (plus the optional RMSE trim, if enabled) and
form the valid pool. A low ratio (e.g. 5/100) means most starting points led
the optimiser to poor solutions &mdash; consider widening bounds or
increasing the trial budget.</p>
"""


# Merged stat columns: (header, (central_key, spread_key)) from ensemble.describe().
# Rendered as a single "central ± spread" cell; the active pair is bold-emphasised.
_STAT_COLUMNS = (('Median ± MAD', ('median', 'mad')), ('Mean ± SD', ('mean', 'std')))
_P68_HEADER = '68% Range'  # central 68% interval [P16, P84]
_RANGE_HEADER = 'Min-Max Range'
_COLUMN_HEADERS = ('Parameter', 'Estimate') + tuple(h for h, _ in _STAT_COLUMNS) + (_P68_HEADER, _RANGE_HEADER, 'Units')
# Columns: 0 Parameter | 1 Estimate | 2 Median±MAD | 3 Mean±SD | 4 68% Range | 5 Min-Max Range | 6 Units.
_STAT_COL0 = 2
_P68_COL = _STAT_COL0 + len(_STAT_COLUMNS)
_RANGE_COL = _P68_COL + 1
_UNITS_COL = _RANGE_COL + 1


class FitSummaryWidget(QWidget):
    """Read-only display of ``FitResult`` statistics.

    Layout
    ------
    - ``QGroupBox("Fitted Parameters")`` with a Representative selector and
      columns: Parameter | Estimate | Median \u00b1 MAD | Mean \u00b1 SD | 68% Range |
      Min-Max Range | Units, plus a log\u2081\u2080 row for each association constant.
      ``Estimate`` is the representative fit; the remaining columns summarise
      the spread across all valid fits. Rows come from
      :func:`~core.pipeline.fit_pipeline.summarize_parameters`, shared with the
      plot annotation and the exports.
    - ``QGroupBox("Fit Quality")`` with RMSE, R-squared, Fits passing.
    """

    representative_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Sticky "Selected" representative: the last fit picked from a
        # distribution-plot click, kept across Best/Median/Worst switches so the
        # user can compare and switch back. Reset when a new fit is displayed.
        self._plot_selected_index: int | None = None
        self._last_result_id: str | None = None

        self._params_group = InfoGroupBox(
            'Fitted Parameters',
            'Fitted Parameters \u2014 columns',
            _PARAMS_HELP_HTML,
        )

        # Representative selector: which actual fit is reported (drives the curve).
        self._rep_combo = QComboBox()
        # Size to the widest item so no label is ever truncated.
        self._rep_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._rep_combo.setMinimumContentsLength(14)
        self._rep_combo.setToolTip(
            '<qt>Which valid fit is reported and drawn. Defaults to the best '
            '(highest R\u00b2).<br>Pick another, or click a point in the Distributions '
            'tab.</qt>'
        )
        self._rep_combo.currentIndexChanged.connect(self._on_rep_combo_changed)

        stats_row = QHBoxLayout()
        stats_row.addWidget(QLabel('Representative:'))
        stats_row.addWidget(self._rep_combo)
        stats_row.addStretch(1)

        self._table = QTableWidget(0, len(_COLUMN_HEADERS))
        self._table.setHorizontalHeaderLabels(list(_COLUMN_HEADERS))
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        params_layout = QVBoxLayout(self._params_group)
        params_layout.addLayout(stats_row)
        params_layout.addWidget(self._table)

        self._quality_group = InfoGroupBox('Fit Quality', 'Fit Quality', _FIT_QUALITY_HELP_HTML)
        quality_layout = QFormLayout(self._quality_group)
        self._rmse_label = QLabel('\u2014')
        self._r2_label = QLabel('\u2014')
        self._passing_label = QLabel('\u2014')
        quality_layout.addRow('RMSE:', self._rmse_label)
        quality_layout.addRow('R\u00b2:', self._r2_label)
        quality_layout.addRow('Fits passing:', self._passing_label)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self._params_group, stretch=3)
        main_layout.addWidget(self._quality_group, stretch=1)

    def update_result(self, result: FitResult) -> None:
        """Populate the widget from a ``FitResult``.

        ``Estimate`` is the representative fit; the Median ± MAD, Mean ± SD, and
        Range columns summarise the valid-fit pool. Rows (including the log₁₀
        twin of each association constant) come from
        :func:`~core.pipeline.fit_pipeline.summarize_parameters`, so the table,
        the plot annotation and the exports cannot disagree.
        """
        # A new fit clears the sticky plot-selection; refreshes of the same
        # result (e.g. a re-selection) keep it.
        if result.id != self._last_result_id:
            self._plot_selected_index = None
            self._last_result_id = result.id

        self._populate_rep_combo(result)

        rows = summarize_parameters(result)
        pool_missing = not result.parameter_samples

        self._table.setRowCount(len(rows))
        for row, spec in enumerate(rows):
            # Log rows are dimensionless; their source unit only labels them.
            unit_html = fmt_unit_html(spec.unit) if spec.unit else ''
            if spec.is_log:
                label = f'log₁₀({fmt_param(spec.key)} / {unit_html})' if unit_html else f'log₁₀({fmt_param(spec.key)})'
            else:
                label = fmt_param(spec.key)
            value_unit = '' if spec.is_log else spec.unit

            self._set_cell(row, 0, label)
            self._set_cell(row, 1, self._fmt(spec.estimate, value_unit))
            d = spec.stats
            if d is None:
                for col in range(_STAT_COL0, _RANGE_COL + 1):
                    self._set_cell(row, col, '—', tooltip='No fit pool stored for this result (e.g. a legacy import).')
            else:
                for offset, (_, (ck, sk)) in enumerate(_STAT_COLUMNS):
                    self._set_cell(row, _STAT_COL0 + offset, self._fmt_pair(d[ck], d[sk], value_unit))
                self._set_cell(row, _P68_COL, self._fmt_range(d['p16'], d['p84'], value_unit))
                self._set_cell(row, _RANGE_COL, self._fmt_range(d['min'], d['max'], value_unit))
            self._set_cell(row, _UNITS_COL, unit_html if value_unit else '—')

        rmse_html = f'{Q_(result.rmse, "au"):.3g~H}'
        self._rmse_label.setTextFormat(Qt.TextFormat.RichText)
        self._rmse_label.setText(rmse_html)
        self._r2_label.setText(f'{result.r_squared:.4f}')
        passing = f'{result.n_passing} / {result.n_total}'
        if pool_missing:
            passing += '  (pool unavailable)'
        self._passing_label.setText(passing)
        self._autosize_columns()

    # ------------------------------------------------------------------
    # Cell / selector helpers
    # ------------------------------------------------------------------

    def _fmt(self, magnitude: float, unit_str: str) -> str:
        """Pint HTML value with the trailing unit stripped (units have a column)."""
        if unit_str:
            return f'{Q_(magnitude, unit_str):.3g~H}'.rsplit(' ', 1)[0]
        return f'{magnitude:.3g}'

    def _fmt_pair(self, central: float, spread: float, unit_str: str) -> str:
        """Merged 'central ± spread' cell (units stripped; shown in the Units column)."""
        return f'{self._fmt(central, unit_str)} ± {self._fmt(spread, unit_str)}'

    def _fmt_range(self, lo: float, hi: float, unit_str: str) -> str:
        """Merged '[min, max]' range cell."""
        return f'[{self._fmt(lo, unit_str)}, {self._fmt(hi, unit_str)}]'

    def _set_cell(self, row: int, col: int, html: str, *, tooltip: str | None = None) -> None:
        lbl = QLabel(html)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        if tooltip:
            lbl.setToolTip(tooltip)
        self._table.setCellWidget(row, col, lbl)

    def _populate_rep_combo(self, result: FitResult) -> None:
        """Offer Best, Median, Worst, and the sticky Selected representative.

        Best/Worst are the highest/lowest R² fits; Median is the fit nearest the
        pool's median R². Selected is the last fit clicked in the distribution
        plots — kept across Best/Median/Worst switches so the user can compare
        and switch back to it. Ranking uses R².
        """
        self._rep_combo.blockSignals(True)
        self._rep_combo.clear()
        quality = result.quality_samples
        if quality and 'r_squared' in quality:
            r2 = np.asarray(quality['r_squared'], dtype=float)
            items = [
                ('Best', int(np.argmax(r2))),
                ('Median', int(np.argmin(np.abs(r2 - np.median(r2))))),
                ('Worst', int(np.argmin(r2))),
            ]
            sel = self._plot_selected_index
            if sel is not None and 0 <= sel < r2.size:
                items.append(('Selected', int(sel)))
            for label, idx in items:
                self._rep_combo.addItem(f'{label} · R²={r2[idx]:.4f}', idx)
            # Show whichever item matches the reported representative (first match).
            cur = result.representative_index
            if cur is not None:
                pos = next((i for i, (_, idx) in enumerate(items) if idx == cur), 0)
                self._rep_combo.setCurrentIndex(pos)
            self._rep_combo.setEnabled(True)
        else:
            self._rep_combo.addItem('—', -1)
            self._rep_combo.setEnabled(False)
        self._rep_combo.blockSignals(False)

    def note_plot_selection(self, index: int) -> None:
        """Remember *index* as the sticky 'Selected' representative.

        Called when the user clicks a fit in the distribution plots, so the
        choice survives Best/Median/Worst switches until the next fit.
        """
        self._plot_selected_index = int(index)

    def _on_rep_combo_changed(self, _index: int) -> None:
        idx = self._rep_combo.currentData()
        if idx is not None and idx >= 0:
            self.representative_selected.emit(int(idx))

    def _autosize_columns(self) -> None:
        # resizeColumnsToContents() ignores widgets set via setCellWidget(),
        # so we measure each QLabel's sizeHint() directly, then spread any
        # leftover viewport width evenly across columns.
        n_cols = self._table.columnCount()
        n_rows = self._table.rowCount()
        fm = self._table.horizontalHeader().fontMetrics()
        widths = []
        for col in range(n_cols):
            item = self._table.horizontalHeaderItem(col)
            header_text = item.text() if item is not None else ''
            w = fm.horizontalAdvance(header_text) + 24
            for row in range(n_rows):
                widget = self._table.cellWidget(row, col)
                if widget is not None:
                    w = max(w, widget.sizeHint().width() + 16)
            widths.append(w)

        extra = max(0, (self._table.viewport().width() - sum(widths)) // n_cols)
        for col, w in enumerate(widths):
            self._table.setColumnWidth(col, w + extra)

    def clear(self) -> None:
        """Reset all fields to their empty state."""
        self._table.setRowCount(0)
        self._table.setHorizontalHeaderLabels(list(_COLUMN_HEADERS))
        self._rmse_label.setText('\u2014')
        self._r2_label.setText('\u2014')
        self._passing_label.setText('\u2014')
