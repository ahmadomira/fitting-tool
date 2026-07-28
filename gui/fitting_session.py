"""FittingSession — one complete fitting workspace (one tab)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.assays.registry import ASSAY_REGISTRY, AssayType
from core.data_processing.measurement_set import MeasurementSet
from core.data_processing.plotting import prepare_plot_data
from core.io.formats.bmg_reader import BMG_PLACEHOLDER_KEY
from core.pipeline.fit_pipeline import FitConfig, FitResult, apply_statistics_mode, select_representative
from core.pipeline.sensitivity import estimated_fit_count
from gui.app_state import SessionState
from gui.plotting.distribution_widget import DistributionWidget
from gui.plotting.fit_summary_widget import FitSummaryWidget
from gui.plotting.plot_style import PlotStyleWidget
from gui.plotting.plot_widget import PlotWidget
from gui.plotting.sensitivity_widget import SensitivityWidget
from gui.widgets.assay_config_panel import AssayConfigPanel
from gui.widgets.bounds_panel import BoundsPanel
from gui.widgets.data_panel import DataPanel
from gui.widgets.fit_config_panel import FitConfigPanel
from gui.widgets.flat_tabs import FlatTabWidget
from gui.widgets.info_button import InfoGroupBox
from gui.widgets.preprocessing_panel import PreprocessingPanel
from gui.widgets.replica_panel import ReplicaPanel
from gui.widgets.sensitivity_panel import SensitivityPanel
from gui.workers import FitWorker, SensitivityWorker


class _SidebarScrollArea(QScrollArea):
    """Vertically-scrolling sidebar whose width follows its content.

    A plain ``QScrollArea`` reports a near-zero width hint regardless of its
    content, so a ``QSplitter`` will shrink it until the content clips. The
    sidebar scrolls vertically only, so its width must track the content: we
    advertise the content's *minimum* width (plus the frame and the vertical
    scrollbar) as both our minimum and preferred width. The splitter then honors
    it as the sidebar's floor and shrinks the *other* pane instead — content is
    never clipped — while the sidebar opens compact (the content minimum) rather
    than at the widest panel's much larger preferred width.
    """

    def _content_width(self) -> int:
        # Width needed to show the content un-clipped: the content's own minimum
        # width + the frame + the vertical scrollbar. Read from Qt at runtime so
        # it adapts to platform/style and to the panels' current contents.
        content = self.widget()
        if content is None:
            return 0
        return content.minimumSizeHint().width() + 2 * self.frameWidth() + self.verticalScrollBar().sizeHint().width()

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        if self.widget() is not None:
            hint.setWidth(self._content_width())
        return hint

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        if self.widget() is not None:
            hint.setWidth(self._content_width())
        return hint


_PLOT_STYLE_HELP_HTML = """
<h3>Plot Style</h3>

<p><b>What This Section Is For</b></p>
<p>Controls for the interactive plot on the right: which series are
visible (replicas, average, error bars, fit curves), axis/tick/legend
font sizes, marker shapes and palette, line widths and colors, and the
x-axis display unit. Changes apply live &mdash; no need to re-run the
fit.</p>

<p><b>What Affects What</b></p>
<ul>
  <li><b>Visibility</b> &mdash; toggles whole series on or off. Hidden
      series are not drawn and do not appear in the legend.</li>
  <li><b>Axes</b> &mdash; font sizes for labels/ticks and the x-axis
      unit (nM / &micro;M / mM / M). Data is stored in M internally and
      rescaled at render time.</li>
  <li><b>Replicas, Dropped Replicas, Average Line, Fit Curves, Error
      Bars</b> &mdash; per-series style groups.</li>
  <li><b>Legend, Annotations</b> &mdash; toggle the legend and the
      draggable fit-results overlay.</li>
</ul>

<p><b>Save and Reuse: Style Templates</b></p>
<p>You can save the entire Plot Style configuration and reload it
later, so you don&rsquo;t have to reconfigure fonts, colors, and
visibility every session.</p>
<ul>
  <li><b>Import &rarr; Load Style Template&hellip;</b> &mdash; load a
      <code>.json</code> template (all controls update at once).</li>
  <li><b>Export &rarr; Save Style Template</b> &mdash; write the
      current style to a <code>.json</code> file you can share or
      commit to your project.</li>
</ul>
<p>Style templates round-trip exactly, so they&rsquo;re a good way to
enforce a consistent look across figures in a paper or lab report.</p>

<p><b>Tip</b></p>
<p>If the fit-results annotation or the legend ends up in an awkward
spot, drag them with the mouse. The app remembers the drag position
until you toggle the corresponding overlay off and back on.</p>
"""

_DEMO_IDA_PATH = Path(__file__).parent.parent / 'data' / 'IDA_system.txt'


class FittingSession(QWidget):
    """One complete fitting workflow: data → preprocess → configure → fit → visualize.

    Each tab in the main window is one FittingSession.  All state is stored in
    :class:`SessionState`; widgets communicate via Qt signals coordinated here.

    Signals
    -------
    title_changed(str)
        Emitted when the tab title should change — the loaded dataset's filename,
        a user-set custom name, or "Untitled".
    status_message(str)
        Emitted to update the main window's status bar.
    """

    title_changed = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = SessionState()
        self._fit_worker: FitWorker | None = None
        self._sensitivity_worker: SensitivityWorker | None = None
        self._custom_tab_name: str | None = None
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Public API (called by FittingMainWindow)
    # ------------------------------------------------------------------

    def set_custom_tab_name(self, name: str) -> None:
        """Set a manual tab name (double-click rename); empty reverts to auto."""
        self._custom_tab_name = name.strip() or None
        self._emit_tab_title()

    def _tab_title(self) -> str:
        """Resolve the tab title: custom name, else dataset stem, else 'Untitled'."""
        if self._custom_tab_name:
            return self._custom_tab_name
        src = self._state.source_file
        return Path(src).stem if src else 'Untitled'

    def _emit_tab_title(self) -> None:
        self.title_changed.emit(self._tab_title())

    def run_fit(self) -> None:
        """Start a background fitting run."""
        ms = self._state.measurement_set
        if ms is None:
            QMessageBox.warning(self, 'No Data', 'Load a measurement file first.')
            return

        # Placeholder guard — fitting with column indices 1..N as
        # concentrations would silently produce meaningless Ka values.
        # Source-agnostic: any plate-reader import that lacks concentrations
        # sets this flag, regardless of instrument.
        if ms.metadata.get(BMG_PLACEHOLDER_KEY):
            QMessageBox.warning(
                self,
                'Concentrations Required',
                'This dataset has placeholder concentrations. Enter the real '
                'concentration vector in the Data panel before running the fit.',
            )
            self._data_panel.focus_concentration_table()
            return

        assay_cls = self._assay_panel.get_assay_class()
        conditions = self._assay_panel.current_conditions()
        config = self._fit_panel.current_config()
        custom_bounds = self._bounds_panel.current_bounds()
        if custom_bounds:
            config = replace(config, custom_bounds=custom_bounds)

        if config.per_replica and ms.n_active < 3:
            self.status_message.emit(
                f'Per-replica fit on {ms.n_active} replica(s) — uncertainty estimate may not be meaningful (<3 active).'
            )

        self._fit_worker = FitWorker(
            ms,
            assay_cls,
            conditions,
            config,
            source_file=self._state.source_file,
            parent=self,
        )
        self._fit_worker.finished.connect(self._on_fit_complete)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.start()
        self.status_message.emit('Fitting…')

    def load_demo_ida(self) -> None:
        """Load the bundled IDA demo dataset and run a fit with default settings.

        Resets all panels to defaults first so every click produces an
        identical, reproducible result regardless of prior session state.
        """
        if not _DEMO_IDA_PATH.exists():
            QMessageBox.warning(self, 'Demo Data Missing', f'Could not find:\n{_DEMO_IDA_PATH}')
            return

        # 1. Reset all panels to IDA defaults
        self._assay_panel.set_assay_type(AssayType.IDA)
        self._bounds_panel.reset_to_defaults()
        self._fit_panel.set_config(FitConfig(per_replica=False))

        # 2. Load data (triggers _on_data_loaded → resets replicas, clears results)
        self._data_panel.load_file(str(_DEMO_IDA_PATH))

        # 3. Apply z-score outlier removal with default settings
        self._preprocess_panel.apply()

        # 4. Run fit
        self.run_fit()

    def _default_save_name(self, suffix: str, tag: str = '') -> str:
        """Build a default save filename based on the loaded dataset's stem.

        Falls back to the ``source_file`` on the most recent fit result when
        no dataset is currently loaded (e.g. after importing results whose
        source file is missing on disk).
        """
        src = self._state.source_file
        if not src and self._state.fit_results:
            src = self._state.fit_results[-1].source_file
        if not src:
            return f'{tag or "output"}{suffix}'
        stem = Path(src).stem
        return f'{stem}{("_" + tag) if tag else ""}{suffix}'

    def _default_filename_base(self) -> str:
        """Stem-only version of :meth:`_default_save_name` (no tag, no suffix)."""
        src = self._state.source_file
        if not src and self._state.fit_results:
            src = self._state.fit_results[-1].source_file
        return Path(src).stem if src else 'output'

    def _distributions_export_config(self):
        """Build a DistributionsExportConfig from persisted QSettings.

        Used by the multi-artefact export when there's no opportunity to
        show the layout picker inline. Defaults match the standalone
        dialog: Auto layout, per-panel dimensions, 300 DPI, all subplots
        selected.
        """
        from gui.dialogs.save_distributions_dialog import (
            _PER_PANEL_IN,
            DistributionsExportConfig,
            SaveDistributionsPlotDialog,
        )
        from gui.plotting.distribution_widget import DistributionWidget
        from gui.preferences import _settings

        keys = self._distribution_widget.param_keys()
        s = _settings()
        s.beginGroup(SaveDistributionsPlotDialog.SETTINGS_GROUP)
        try:
            mode = s.value('layout_mode', 'auto', type=str)
            custom_rows = int(s.value('custom_rows', 2))
            custom_cols = int(s.value('custom_cols', 2))
            saved_w = float(s.value('width_in', 0.0))
            saved_h = float(s.value('height_in', 0.0))
            dpi = int(s.value('dpi', 300))
            sel = s.value('selected_keys', None)
        finally:
            s.endGroup()

        selected = [k for k in keys if k in set(sel)] if isinstance(sel, list) and sel else list(keys)
        n = max(1, len(selected))
        if mode == 'row':
            rows, cols = 1, n
        elif mode == 'col':
            rows, cols = n, 1
        elif mode == 'grid':
            rows, cols = (2, 2) if n <= 4 else DistributionWidget.auto_layout(n)
        elif mode == 'custom':
            rows, cols = custom_rows, custom_cols
            if rows * cols < n:
                rows, cols = DistributionWidget.auto_layout(n)
        else:
            rows, cols = DistributionWidget.auto_layout(n)

        if saved_w > 0 and saved_h > 0:
            width_in, height_in = saved_w, saved_h
        else:
            width_in, height_in = cols * _PER_PANEL_IN, rows * _PER_PANEL_IN

        return DistributionsExportConfig(
            keys=selected,
            rows=rows,
            cols=cols,
            width_in=width_in,
            height_in=height_in,
            dpi=dpi,
        )

    def export_results(self) -> None:
        """Export current fit results to JSON."""
        if not self._state.fit_results:
            QMessageBox.information(self, 'No Results', 'Run a fit first.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Fit Results', self._default_save_name('.json', 'results'), 'JSON (*.json)'
        )
        if not path:
            return
        from gui.session import export_results

        export_results(self._state.fit_results, path)
        self.status_message.emit(f'Results exported to {path}')

    def export_results_txt(self) -> None:
        """Export current fit results as a human-readable text report."""
        if not self._state.fit_results:
            QMessageBox.information(self, 'No Results', 'Run a fit first.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Results (TXT)', self._default_save_name('.txt', 'results'), 'Text files (*.txt)'
        )
        if not path:
            return
        from gui.session import export_results_txt

        export_results_txt(self._state.fit_results, path)
        self.status_message.emit(f'Results exported to {path}')

    def export_raw_data(self) -> None:
        """Export the currently loaded raw measurements to TXT or CSV.

        The output format is chosen by the file extension and round-trips
        through the existing :class:`TxtReader` and :class:`CsvReader`.
        """
        ms = self._state.measurement_set
        if ms is None:
            QMessageBox.information(self, 'No Data', 'Load a measurement file first.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Export Raw Data',
            self._default_save_name('.txt'),
            'Text file (*.txt);;CSV file (*.csv)',
        )
        if not path:
            return
        suffix = Path(path).suffix.lower()
        try:
            if suffix == '.csv':
                from core.io.formats.measurement_writer import write_measurements_csv

                write_measurements_csv(ms, path)
            else:
                # Default to TXT if the user typed a different extension.
                from core.io.formats.measurement_writer import write_measurements_txt

                if suffix != '.txt':
                    path = str(Path(path).with_suffix('.txt'))
                write_measurements_txt(ms, path)
        except Exception:
            QMessageBox.warning(
                self,
                'Export Error',
                'Could not save the raw data. Check that the destination folder exists and '
                'you have permission to write there.',
            )
            return
        self.status_message.emit(f'Raw data exported to {path}')

    def import_results(self) -> None:
        """Import fit results from JSON and replot."""
        path, _ = QFileDialog.getOpenFileName(self, 'Import Fit Results', '', 'JSON (*.json);;All files (*)')
        if not path:
            return
        try:
            from gui.session import import_results

            results = import_results(path)
            if self._state.measurement_set is not None:
                # Raw data already loaded — overlay imported fits immediately
                self._state.fit_results = results
                self._refresh_plot()
            elif results:
                # No raw data loaded — try to auto-load from source_file stored in result
                source_file = results[-1].source_file
                if source_file and Path(source_file).exists():
                    # load_file triggers _on_data_loaded which clears fit_results.
                    # Do NOT assign self._state.fit_results before load_file, or the
                    # in-place .clear() inside _on_data_loaded will wipe `results` too
                    # (same list object). Assign only after load_file returns.
                    self._data_panel.load_file(source_file)
                    self._state.fit_results = results
                    self._refresh_plot()
                else:
                    # Source file not available — show fit curves only
                    self._state.fit_results = results
                    meta = ASSAY_REGISTRY[self._state.assay_type]
                    last = results[-1]
                    plot_data = {
                        'concentrations': last.x_fit.magnitude,
                        'active_replicas': [],
                        'dropped_replicas': [],
                        'average': None,
                        'fits': [
                            {
                                'x': r.x_fit.magnitude,
                                'y': r.y_fit.magnitude,
                                'label': 'Best Fit',
                                'id': r.id,
                            }
                            for r in results
                        ],
                    }
                    self._plot_widget.update_plot(
                        plot_data,
                        x_label=meta.x_label,
                        y_label=meta.y_label,
                        y_unit=meta.y_unit,
                    )
                    self._plot_widget.set_fit_results(results)
            if results:
                self._summary_widget.update_result(results[-1])
            self.status_message.emit(f'Imported {len(results)} result(s) from {path}')
        except Exception:
            QMessageBox.warning(
                self,
                'Import Error',
                'Could not import these results. Make sure the file is a results file (.json) exported by SupraSimFit.',
            )

    def export_plot(self) -> None:
        """Save the current plot as PNG or SVG."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Plot',
            self._default_save_name('.png'),
            'PNG image (*.png);;SVG vector (*.svg)',
        )
        if not path:
            return
        try:
            self._plot_widget.export_image(path)
            self.status_message.emit(f'Plot saved to {path}')
        except Exception:
            QMessageBox.warning(
                self,
                'Export Error',
                'Could not save the plot. Check that the destination folder exists and you '
                'have permission to write there.',
            )

    def open_export_multiple_dialog(self, *, select_all_default: bool) -> None:
        """Show the consolidated multi-artefact export dialog."""
        from gui.dialogs.export_multiple_dialog import ExportMultipleDialog

        dlg = ExportMultipleDialog(self, select_all_default=select_all_default, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            outcomes = dlg.outcomes
            successes = sum(1 for _l, _p, e in outcomes if e is None)
            if successes:
                folder = outcomes[0][1].parent
                self.status_message.emit(f'Exported {successes}/{len(outcomes)} artefact(s) to {folder}')

    def save_distributions_plot(self) -> None:
        """Save the distributions plot as a composite PNG with a layout picker."""
        keys = self._distribution_widget.param_keys()
        if not keys:
            QMessageBox.warning(
                self,
                'Save Error',
                'No distributions to save. Run a fit first.',
            )
            return
        from gui.dialogs.save_distributions_dialog import SaveDistributionsPlotDialog

        dlg = SaveDistributionsPlotDialog(self._distribution_widget, self)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.config is None:
            return
        cfg = dlg.config
        ext = f'.{cfg.format}'
        filter_str = 'PNG image (*.png)' if cfg.format == 'png' else 'SVG vector (*.svg)'
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Distributions Plot',
            self._default_save_name(ext, 'distributions'),
            filter_str,
        )
        if not path:
            return
        try:
            self._distribution_widget.save_plot(
                keys=cfg.keys,
                rows=cfg.rows,
                cols=cfg.cols,
                width_in=cfg.width_in,
                dpi=cfg.dpi,
                path=path,
                format=cfg.format,
            )
            self.status_message.emit(f'Distributions plot saved to {path}')
        except Exception:
            QMessageBox.warning(
                self,
                'Save Error',
                'Could not save the distributions plot. Check that the destination folder '
                'exists and you have permission to write there.',
            )

    def save_style_template(self) -> None:
        """Save current plot style settings to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Style Template',
            'style_template.json',
            'JSON (*.json)',
        )
        if not path:
            return
        from gui.plotting.plot_style import save_style_json

        style = self._style_widget.widget.current_style()
        save_style_json(style, path)
        self.status_message.emit(f'Style template saved to {path}')

    def load_style_template(self) -> None:
        """Load plot style settings from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Load Style Template',
            '',
            'JSON (*.json);;All files (*)',
        )
        if not path:
            return
        try:
            from gui.plotting.plot_style import load_style_json

            style = load_style_json(path)
            self._style_widget.widget.load_style(style)
            # Reflect the loaded x_unit in the Data panel's Display Unit
            # combo so the two stay in sync after a style import.
            loaded_unit = style.get('axes', {}).get('x_unit')
            if loaded_unit:
                self._state.display_unit = loaded_unit
                self._data_panel.set_display_unit(loaded_unit)
            self.status_message.emit(f'Style template loaded from {path}')
        except Exception:
            QMessageBox.warning(
                self,
                'Load Error',
                'Could not load this style template. Make sure it is a style file (.json) exported by SupraSimFit.',
            )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter)

        # ---- Left panel (scrollable) --------------------------------
        # The sidebar scrolls vertically only; its width is driven by the
        # content's own size hints (see _SidebarScrollArea). The splitter honors
        # that as the sidebar's minimum and shrinks the plot pane instead, so the
        # content is never clipped. No max-width cap and childrenCollapsible(False)
        # keep it freely widenable yet never collapsible to zero.
        self._sidebar_scroll = _SidebarScrollArea()
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll = self._sidebar_scroll

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(6, 6, 6, 6)

        self._data_panel = DataPanel(initial_display_unit=self._state.display_unit)
        self._preprocess_panel = PreprocessingPanel()
        self._replica_panel = ReplicaPanel()
        self._assay_panel = AssayConfigPanel()
        self._fit_panel = FitConfigPanel()
        self._sensitivity_panel = SensitivityPanel()
        self._bounds_panel = BoundsPanel()
        self._style_widget = _Grouped(
            'Plot Style',
            PlotStyleWidget(),
            info_title='Plot Style',
            info_html=_PLOT_STYLE_HELP_HTML,
        )

        for widget in (
            self._data_panel,
            self._preprocess_panel,
            self._replica_panel,
            self._assay_panel,
            self._fit_panel,
            self._sensitivity_panel,
            self._bounds_panel,
            self._style_widget,
        ):
            left_layout.addWidget(widget)

        left_layout.addStretch()
        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        # ---- Right panel (tabbed plot area + summary, vertically resizable)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)

        # Tabbed plot area: Fit Curve | Distributions (flat/minimal, white pane).
        self._plot_widget = PlotWidget()
        self._distribution_widget = DistributionWidget()
        self._sensitivity_widget = SensitivityWidget()
        self._plot_tabs = FlatTabWidget(white_pane=True)
        self._plot_tabs.addTab(self._plot_widget, 'Fit Curve')
        self._plot_tabs.addTab(self._distribution_widget, 'Distributions')
        self._plot_tabs.addTab(self._sensitivity_widget, 'Ka Sensitivity')

        self._summary_widget = FitSummaryWidget()

        right_splitter.addWidget(self._plot_tabs)
        right_splitter.addWidget(self._summary_widget)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        splitter.addWidget(right_splitter)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # No explicit initial split: the sidebar opens at its content-driven
        # sizeHint width and the plot (stretch factor 1) takes the rest.

        # Initialise BoundsPanel and SensitivityPanel for default assay type
        self._bounds_panel.set_assay_type(self._state.assay_type)
        self._sensitivity_panel.set_assay_type(self._state.assay_type)

        # Push the initial display unit into PlotStyleWidget so the style
        # dict reflects the DataPanel's combo from the first emission.
        self._style_widget.widget.set_x_unit(self._state.display_unit)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._data_panel.data_loaded.connect(self._on_data_loaded)
        self._data_panel.data_cleared.connect(self._on_data_cleared)
        self._data_panel.display_unit_changed.connect(self._on_display_unit_changed)

        self._preprocess_panel.preprocessing_applied.connect(self._on_preprocessing_applied)
        self._preprocess_panel.preprocessing_reset.connect(self._on_preprocessing_reset)

        self._replica_panel.replicas_changed.connect(self._on_replicas_changed)

        self._assay_panel.assay_type_changed.connect(self._on_assay_type_changed)
        self._assay_panel.assay_type_changed.connect(self._sensitivity_panel.set_assay_type)
        self._assay_panel.conditions_changed.connect(self._on_conditions_changed)

        self._fit_panel.config_changed.connect(self._on_config_changed)

        self._bounds_panel.bounds_changed.connect(self._on_bounds_changed)

        self._sensitivity_panel.run_requested.connect(self.run_sensitivity)
        self._sensitivity_panel.cancel_requested.connect(self._cancel_sensitivity)

        # Direct widget-to-widget: style → plot / distributions / sensitivity
        self._style_widget.widget.style_changed.connect(self._plot_widget.apply_style)
        self._style_widget.widget.style_changed.connect(self._distribution_widget.apply_style)
        self._style_widget.widget.style_changed.connect(self._sensitivity_widget.apply_style)

        # Ensemble interaction: switch reported ± / pick a different representative.
        self._summary_widget.statistics_mode_changed.connect(self._on_statistics_mode_changed)
        self._summary_widget.representative_selected.connect(self._on_representative_selected)
        self._distribution_widget.representative_selected.connect(self._on_plot_fit_selected)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_data_loaded(self, ms: MeasurementSet) -> None:
        self._state.measurement_set = ms
        self._state.source_file = self._data_panel.current_path()
        self._emit_tab_title()
        self._state.fit_results.clear()
        self._preprocess_panel.set_measurement_set(ms)
        self._replica_panel.set_measurement_set(ms)
        self._summary_widget.clear()
        self._distribution_widget.clear()
        self._refresh_plot()
        active = ms.n_active
        total = ms.n_replicas
        self.status_message.emit(f'Loaded: {ms.n_points} pts × {total} replicas — {active}/{total} active')

    def _on_data_cleared(self) -> None:
        self._state.measurement_set = None
        self._state.source_file = None
        self._emit_tab_title()
        self._state.fit_results.clear()
        self._replica_panel.clear()
        self._summary_widget.clear()
        self._distribution_widget.clear()

    def _on_preprocessing_applied(self) -> None:
        self._replica_panel.refresh()
        self._refresh_plot()
        ms = self._state.measurement_set
        if ms:
            self.status_message.emit(f'Preprocessing applied — {ms.n_active}/{ms.n_replicas} replicas active')

    def _on_preprocessing_reset(self) -> None:
        self._replica_panel.refresh()
        self._refresh_plot()
        self.status_message.emit('Replicas reset — all active')

    def _on_replicas_changed(self) -> None:
        self._refresh_plot()

    def _on_assay_type_changed(self, assay_type: AssayType) -> None:
        self._state.assay_type = assay_type
        self._bounds_panel.set_assay_type(assay_type)
        meta = ASSAY_REGISTRY[assay_type]
        self._update_axis_labels(meta.x_label, meta.y_label, meta.y_unit)

    def _on_conditions_changed(self) -> None:
        self._state.conditions = self._assay_panel.current_conditions()

    def _on_config_changed(self) -> None:
        self._state.fit_config = self._fit_panel.current_config()

    def _on_bounds_changed(self) -> None:
        self._state.custom_bounds = self._bounds_panel.current_bounds()

    def _on_display_unit_changed(self, unit: str) -> None:
        """DataPanel announces a new plot-display unit — propagate to plot."""
        self._state.display_unit = unit
        self._style_widget.widget.set_x_unit(unit)

    def _on_fit_complete(self, result: FitResult) -> None:
        self._state.fit_results = [result]
        self._refresh_plot()
        self._summary_widget.update_result(result)
        self._plot_widget.set_fit_results(self._state.fit_results)
        self._distribution_widget.update_result(result)

        if result.uncertainty_source == 'replicate':  # JSON-compat magic value
            n_fit = result.metadata.get('n_replicas_fit', result.n_passing)
            n_total = result.metadata.get('n_replicas_total', result.n_total)
            self.status_message.emit(
                f'Per-replica fit complete — R²={result.r_squared:.4f}, '
                f'RMSE={result.rmse:.4e}, {n_fit}/{n_total} replicas fit'
            )
            failures = result.metadata.get('replica_failures') or {}
            if failures:
                details = '\n'.join(f'  • {rid}: {reason}' for rid, reason in failures.items())
                QMessageBox.warning(
                    self,
                    'Some replicas were skipped',
                    (
                        f'{len(failures)} of {n_total} replicas could not be fit '
                        f'and were excluded from the aggregate.\n\n'
                        f'Skipped replicas:\n{details}\n\n'
                        f'The aggregate below is based on the {n_fit} successful replica(s).'
                    ),
                )
        else:
            self.status_message.emit(
                f'Fit complete — R²={result.r_squared:.4f}, RMSE={result.rmse:.4e}, '
                f'{result.n_passing}/{result.n_total} trials passed'
            )

    def _on_fit_error(self, msg: str) -> None:
        QMessageBox.warning(self, 'Fit Error', msg)
        self.status_message.emit(f'Fit failed: {msg}')

    # ------------------------------------------------------------------
    # Ka sensitivity analysis
    # ------------------------------------------------------------------

    def run_sensitivity(self) -> None:
        """Start a background Ka input-sensitivity run."""
        ms = self._state.measurement_set
        if ms is None:
            QMessageBox.warning(self, 'No Data', 'Load a measurement file first.')
            return

        if self._state.assay_type is AssayType.DYE_ALONE:
            QMessageBox.warning(
                self,
                'Sensitivity Analysis',
                'Dye-only measurements have no association constant to analyse. '
                'Pick a binding assay (GDA, IDA, or DBA) to run a sensitivity analysis.',
            )
            return

        sens_config = self._sensitivity_panel.current_config()
        if not sens_config.delta_pct:
            QMessageBox.warning(
                self,
                'Sensitivity Analysis',
                'No input is set to vary. Set a ±% above zero for at least one input '
                '(the titrant or a concentration) so the analysis has something to perturb.',
            )
            return

        n_fits = estimated_fit_count(sens_config)
        if n_fits > 2000:
            reply = QMessageBox.question(
                self,
                'Large Sensitivity Run',
                f'This will run about {n_fits} fits and may take a while. Continue?',
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        assay_cls = self._assay_panel.get_assay_class()
        conditions = self._assay_panel.current_conditions()
        fit_config = self._fit_panel.current_config()

        self._sensitivity_worker = SensitivityWorker(
            ms,
            assay_cls,
            conditions,
            fit_config,
            sens_config,
            parent=self,
        )
        self._sensitivity_worker.progress.connect(self._sensitivity_widget.set_progress)
        self._sensitivity_worker.progress.connect(
            lambda done, total: self.status_message.emit(f'Sensitivity {done}/{total}…')
        )
        self._sensitivity_worker.finished.connect(self._on_sensitivity_complete)
        self._sensitivity_worker.error.connect(self._on_sensitivity_error)

        self._sensitivity_widget.show_running()
        self._sensitivity_panel.set_running(True)
        self._plot_tabs.setCurrentWidget(self._sensitivity_widget)
        self._sensitivity_worker.start()
        self.status_message.emit('Running sensitivity analysis…')

    def _on_sensitivity_complete(self, result) -> None:
        self._state.sensitivity_result = result
        self._sensitivity_widget.update_result(result)
        self._sensitivity_panel.set_running(False)
        self.status_message.emit(f'Sensitivity complete — {result.n_success}/{result.n_total} fits succeeded')

    def _on_sensitivity_error(self, msg: str) -> None:
        QMessageBox.warning(self, 'Sensitivity Analysis', msg)
        self._sensitivity_panel.set_running(False)
        self.status_message.emit(f'Sensitivity failed: {msg}')

    def _cancel_sensitivity(self) -> None:
        if self._sensitivity_worker is not None:
            self._sensitivity_worker.cancel()
            self.status_message.emit('Cancelling…')

    def _on_statistics_mode_changed(self, mode: str) -> None:
        """Switch the reported ± between median±MAD and mean±STDEV (no re-fit)."""
        results = self._state.fit_results
        if not results:
            return
        result = results[-1]
        if result.parameter_samples is None:
            return
        apply_statistics_mode(result, mode)
        self._summary_widget.update_result(result)
        # Refresh the plot annotation so its ± reflects the new mode.
        self._plot_widget.set_fit_results(self._state.fit_results)

    def _on_plot_fit_selected(self, index: int) -> None:
        """A fit picked by clicking a distribution point: remember it as the
        sticky 'Selected' choice, then report it as the representative."""
        self._summary_widget.note_plot_selection(index)
        self._on_representative_selected(index)

    def _on_representative_selected(self, index: int) -> None:
        """Report a different valid fit as the representative (from the
        Representative selector or a distribution-plot click)."""
        ms = self._state.measurement_set
        results = self._state.fit_results
        if ms is None or not results:
            return
        result = results[-1]
        if result.parameter_samples is None:
            return
        assay = ms.to_assay(
            self._assay_panel.get_assay_class(),
            conditions=self._assay_panel.current_conditions(),
            use_average=True,
        )
        select_representative(result, assay, index)
        self._refresh_plot()
        self._summary_widget.update_result(result)
        self._distribution_widget.update_result(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_plot(self) -> None:
        ms = self._state.measurement_set
        if ms is None:
            return
        meta = ASSAY_REGISTRY[self._state.assay_type]
        plot_data = prepare_plot_data(ms, self._state.fit_results, show_dropped=True)
        self._plot_widget.update_plot(
            plot_data,
            x_label=meta.x_label,
            y_label=meta.y_label,
            y_unit=meta.y_unit,
        )
        self._plot_widget.set_fit_results(self._state.fit_results)

    def _update_axis_labels(self, x: str, y: str, y_unit: str) -> None:
        self._plot_widget.set_axis_labels(x, y, y_unit)


class _GroupedPlain(QGroupBox):
    """Thin wrapper that places an existing widget inside a QGroupBox."""

    def __init__(self, title: str, widget: QWidget, parent=None):
        super().__init__(title, parent)
        self.widget = widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(widget)


class _GroupedWithInfo(InfoGroupBox):
    """Plain wrapper around :class:`InfoGroupBox` exposing ``self.widget``."""

    def __init__(
        self,
        title: str,
        widget: QWidget,
        info_title: str,
        info_html: str,
        parent=None,
    ):
        super().__init__(title, info_title, info_html, parent)
        self.widget = widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(widget)


def _Grouped(
    title: str,
    widget: QWidget,
    parent=None,
    info_title: str | None = None,
    info_html: str | None = None,
):
    """Factory: return a plain or info-bearing group-box wrapper."""
    if info_html is not None:
        return _GroupedWithInfo(title, widget, info_title or title, info_html, parent)
    return _GroupedPlain(title, widget, parent)
