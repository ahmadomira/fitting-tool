"""FittingMainWindow — thin shell managing tabbed fitting sessions."""

from __future__ import annotations

from PyQt6.QtCore import QCoreApplication, QEvent, QLocale, QObject, Qt
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QToolButton,
)

from _version import __version__
from gui.fitting_session import FittingSession
from gui.preferences import APP_NAME, ORG_NAME
from gui.update_check import UpdateCheckWorker, is_newer
from gui.update_dialog import UpdateAvailableDialog
from gui.widgets.flat_tabs import FlatTabWidget


class _SpinBoxWheelRedirect(QObject):
    """Block non-focused spinboxes from swallowing wheel events.

    Qt's event-filter mechanism cannot propagate a consumed wheel event
    to the target's parent, so we simply eat the event when a spinbox
    lacks focus. The user's primary complaint — accidentally changing
    values while scrolling the sidebar — is fixed; the sidebar scroll
    pauses while the cursor is over a pyqtgraph ParameterTree spinbox
    but no longer mutates its value. Our project-owned spinboxes go
    through :class:`gui.widgets.numeric_inputs.NoScrollSpinBox` /
    :class:`NoScrollDoubleSpinBox` instead, which ``event.ignore()``
    and so let Qt forward the wheel up to the scroll area cleanly.
    """

    def eventFilter(self, obj, event):  # type: ignore[override]
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QAbstractSpinBox):
            if not obj.hasFocus():
                return True
        return False


_APP_QSS = """
QToolBar {
    spacing: 6px;
    padding: 4px 10px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.15);
}
/* Scoped to the toolbar so it does not leak into the tab bar's scroll arrows. */
QToolBar QToolButton {
    padding: 5px 14px;
    min-width: 70px;
    border: 1px solid rgba(0, 0, 0, 0.22);
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(255,255,255,0.95), stop:1 rgba(225,225,225,0.95));
    font-size: 12px;
}
QToolBar QToolButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(255,255,255,1.0), stop:1 rgba(235,235,235,1.0));
    border-color: rgba(0, 0, 0, 0.32);
}
QToolBar QToolButton:pressed {
    background: rgba(195, 195, 195, 0.95);
}
QGroupBox {
    font-size: 14px;
    font-weight: bold;
    margin-top: 22px;
    border: 1px solid rgba(0, 0, 0, 0.2);
    border-radius: 5px;
    padding: 14px 6px 6px 6px;
    background-color: palette(window);
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 13px;
    padding: 2px 28px 2px 8px;
    color: rgba(20, 40, 90, 0.9);
    background-color: palette(window);
}
"""


class FittingMainWindow(QMainWindow):
    """Main application window: tab management + toolbar + menu routing.

    All fitting state lives in each :class:`~gui.fitting_session.FittingSession`
    tab.  The main window holds no fitting state of its own.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._update_worker: UpdateCheckWorker | None = None
        self._sim_window: QMainWindow | None = None
        self._set_title()
        self.resize(1280, 820)
        self._setup_tabs()
        self._setup_toolbar()
        self._setup_menus()
        self._setup_statusbar()
        # Open one empty session on startup
        self._new_session()
        # Silent background check for newer releases on GitHub.
        # Failures (offline, rate-limited, etc.) are intentionally swallowed.
        self._run_update_check(silent=True)

    def _set_title(self, suffix: str = '') -> None:
        """Set the window title to ``SupraSimFit <version> [suffix]``.

        Called once at construction with no suffix, and again with
        ``"(updates available)"`` when the background check finds a newer
        release.
        """
        base = f'SupraSimFit {__version__}'
        self.setWindowTitle(f'{base} {suffix}'.rstrip())

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def active_session(self) -> FittingSession | None:
        widget = self._tabs.currentWidget()
        return widget if isinstance(widget, FittingSession) else None

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_tabs(self) -> None:
        # Flat/minimal session tabs with an inline "+" glued after the last tab.
        # Closable + movable; FlatTabWidget hides the sole tab's close control so
        # the bar can never be emptied.
        self._tabs = FlatTabWidget(
            closable=True,
            movable=True,
            editable=True,
            add_callback=self._new_session,
            add_tooltip='New session (Ctrl+T)',
        )
        self._tabs.tabCloseRequested.connect(self._close_tab)
        self._tabs.tab_renamed.connect(self._on_tab_renamed)
        self.setCentralWidget(self._tabs)

    def _setup_toolbar(self) -> None:
        tb = QToolBar('Main')
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(tb)

        # Import actions (grouped under the Import button)
        self._act_load = QAction('Import Data\u2026', self)
        self._act_load.setShortcut(QKeySequence.StandardKey.Open)
        self._act_load.setToolTip('Import measurement data into the active session (Ctrl+O)')
        self._act_load.triggered.connect(self._on_load)
        # Register action on the main window so the Ctrl+O shortcut is active
        # even though the action is not placed directly on the toolbar.
        self.addAction(self._act_load)

        self._act_import = QAction('Import Fit Results\u2026', self)
        self._act_import.setToolTip('Import fit results from JSON for re-plotting')
        self._act_import.triggered.connect(self._on_import)

        self._act_load_style = QAction('Load Style Template\u2026', self)
        self._act_load_style.setToolTip('Load plot style settings from a JSON file')
        self._act_load_style.triggered.connect(self._on_load_style)

        import_menu = QMenu(self)
        import_menu.addAction(self._act_load)
        import_menu.addAction(self._act_import)
        import_menu.addSeparator()
        import_menu.addAction(self._act_load_style)

        self._import_btn = self._make_menu_button('Import', import_menu, 'Import / Load options')
        tb.addWidget(self._import_btn)

        self._act_demo = QAction('Demo IDA', self)
        self._act_demo.setToolTip('Load bundled IDA demo data and run a fit with default settings')
        self._act_demo.triggered.connect(self._on_load_demo)
        tb.addAction(self._act_demo)

        self._act_simulate = QAction('Simulate…', self)
        self._act_simulate.setToolTip('Open the forward-simulation applet for experiment design (no data needed)')
        self._act_simulate.triggered.connect(self._on_simulate)
        tb.addAction(self._act_simulate)

        tb.addSeparator()

        self._act_fit = QAction('Run Fit', self)
        self._act_fit.setShortcut(QKeySequence('Ctrl+R'))
        self._act_fit.setToolTip('Run multi-start fitting in the active session (Ctrl+R)')
        self._act_fit.triggered.connect(self._on_run_fit)
        tb.addAction(self._act_fit)

        tb.addSeparator()

        # Export actions
        self._act_export = QAction('Export Results (JSON)', self)
        self._act_export.setShortcut(QKeySequence.StandardKey.Save)
        self._act_export.setToolTip('Export fit results to JSON (Ctrl+S)')
        self._act_export.triggered.connect(self._on_export)

        self._act_export_txt = QAction('Export Results (TXT)', self)
        self._act_export_txt.setToolTip('Export fit results as a human-readable text report')
        self._act_export_txt.triggered.connect(self._on_export_txt)

        self._act_export_csv = QAction('Export Results (CSV)', self)
        self._act_export_csv.setToolTip('Export fit results as a CSV table, one row per parameter')
        self._act_export_csv.triggered.connect(self._on_export_csv)

        self._act_export_raw = QAction('Export Raw Data\u2026', self)
        self._act_export_raw.setToolTip('Export the currently loaded replicas and concentrations to TXT or CSV')
        self._act_export_raw.triggered.connect(self._on_export_raw)

        self._act_save_plot = QAction('Save Plot', self)
        self._act_save_plot.setToolTip('Export the current plot as PNG or SVG')
        self._act_save_plot.triggered.connect(self._on_save_plot)

        self._act_export_dists = QAction('Save Distributions Plot\u2026', self)
        self._act_export_dists.setToolTip('Save the distributions plot (with layout + size options) as PNG')
        self._act_export_dists.triggered.connect(self._on_save_distributions_plot)

        self._act_save_style = QAction('Save Style Template', self)
        self._act_save_style.setToolTip('Save current plot style settings to a JSON file')
        self._act_save_style.triggered.connect(self._on_save_style)

        self._act_export_all = QAction('Export All…', self)
        self._act_export_all.setShortcut(QKeySequence('Ctrl+Shift+E'))
        self._act_export_all.setToolTip('Export selected artefacts (raw data, results, plots, style) into one folder')
        self._act_export_all.triggered.connect(self._on_export_all)

        export_menu = QMenu(self)
        export_menu.addAction(self._act_export_all)
        export_menu.addSeparator()
        export_menu.addAction(self._act_export)
        export_menu.addAction(self._act_export_txt)
        export_menu.addAction(self._act_export_csv)
        export_menu.addAction(self._act_export_raw)
        export_menu.addSeparator()
        export_menu.addAction(self._act_save_plot)
        export_menu.addAction(self._act_export_dists)
        export_menu.addSeparator()
        export_menu.addAction(self._act_save_style)

        self._export_btn = self._make_menu_button('Export', export_menu, 'Export / Save options')
        tb.addWidget(self._export_btn)

    @staticmethod
    def _make_menu_button(label: str, menu: QMenu, tooltip: str) -> QToolButton:
        """Create a toolbar button that pops a menu without the Qt auto-arrow.

        The default ``InstantPopup`` mode draws a dropdown triangle in the
        bottom-right corner of the button; hiding the ``menu-indicator`` via
        stylesheet gives a clean text-only button while preserving the menu.
        """
        btn = QToolButton()
        btn.setText(label)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(False)
        btn.setMenu(menu)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        btn.setStyleSheet('QToolButton::menu-indicator { image: none; width: 0; }')
        return btn

    def _setup_menus(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu('&File')

        self._act_new = QAction('New Session', self)
        self._act_new.setShortcut(QKeySequence('Ctrl+T'))
        self._act_new.setToolTip('Open a new fitting session tab (Ctrl+T)')
        self._act_new.triggered.connect(self._new_session)
        file_menu.addAction(self._act_new)

        file_menu.addAction(self._act_load)
        file_menu.addSeparator()
        file_menu.addAction(self._act_fit)
        file_menu.addAction(self._act_simulate)
        file_menu.addSeparator()
        file_menu.addAction(self._act_export_all)
        file_menu.addSeparator()
        file_menu.addAction(self._act_export)
        file_menu.addAction(self._act_export_txt)
        file_menu.addAction(self._act_export_csv)
        file_menu.addAction(self._act_export_raw)
        file_menu.addAction(self._act_import)
        file_menu.addAction(self._act_save_plot)
        file_menu.addAction(self._act_export_dists)
        file_menu.addSeparator()
        file_menu.addAction(self._act_save_style)
        file_menu.addAction(self._act_load_style)
        file_menu.addSeparator()
        quit_act = QAction('&Quit', self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # View menu
        view_menu = mb.addMenu('&View')
        close_tab_act = QAction('Close Tab', self)
        close_tab_act.setShortcut(QKeySequence('Ctrl+W'))
        close_tab_act.triggered.connect(lambda: self._close_tab(self._tabs.currentIndex()))
        view_menu.addAction(close_tab_act)

        # Help menu
        help_menu = mb.addMenu('&Help')
        self._act_check_updates = QAction('Check for updates…', self)
        self._act_check_updates.setToolTip('Query GitHub for a newer release of SupraSimFit')
        self._act_check_updates.triggered.connect(lambda: self._run_update_check(silent=False))
        help_menu.addAction(self._act_check_updates)

    def _setup_statusbar(self) -> None:
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage('Ready')

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _new_session(self) -> None:
        session = FittingSession()
        session.title_changed.connect(lambda title, s=session: self._rename_tab(s, title))
        session.status_message.connect(self._statusbar.showMessage)
        idx = self._tabs.addTab(session, 'Untitled')
        self._tabs.setCurrentIndex(idx)

    def _close_tab(self, idx: int) -> None:
        if self._tabs.count() == 1:
            # Keep at least one tab
            QMessageBox.information(self, 'Cannot Close', 'At least one session must remain open.')
            return
        widget = self._tabs.widget(idx)
        self._tabs.removeTab(idx)
        if widget:
            widget.deleteLater()

    def _rename_tab(self, session: FittingSession, title: str) -> None:
        idx = self._tabs.indexOf(session)
        if idx >= 0:
            self._tabs.setTabText(idx, title)

    def _on_tab_renamed(self, index: int, name: str) -> None:
        session = self._tabs.widget(index)
        if isinstance(session, FittingSession):
            session.set_custom_tab_name(name)

    # ------------------------------------------------------------------
    # Toolbar / menu slots
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        session = self.active_session()
        if session:
            session._data_panel.load_file()

    def _on_load_demo(self) -> None:
        session = self.active_session()
        if session:
            session.load_demo_ida()

    def _on_simulate(self) -> None:
        """Open (or re-raise) the non-modal forward-simulation applet."""
        if self._sim_window is None:
            from gui.simulation.simulation_window import SimulationWindow

            self._sim_window = SimulationWindow(self)
        self._sim_window.show()
        self._sim_window.raise_()
        self._sim_window.activateWindow()

    def _on_run_fit(self) -> None:
        session = self.active_session()
        if session:
            session.run_fit()

    def _on_export(self) -> None:
        session = self.active_session()
        if session:
            session.export_results()

    def _on_export_txt(self) -> None:
        session = self.active_session()
        if session:
            session.export_results_txt()

    def _on_export_csv(self) -> None:
        session = self.active_session()
        if session:
            session.export_results_csv()

    def _on_export_raw(self) -> None:
        session = self.active_session()
        if session:
            session.export_raw_data()

    def _on_import(self) -> None:
        session = self.active_session()
        if session:
            session.import_results()

    def _on_save_plot(self) -> None:
        session = self.active_session()
        if session:
            session.export_plot()

    def _on_save_distributions_plot(self) -> None:
        session = self.active_session()
        if session:
            session.save_distributions_plot()

    def _on_export_all(self) -> None:
        session = self.active_session()
        if session:
            session.open_export_multiple_dialog(select_all_default=True)

    def _on_save_style(self) -> None:
        session = self.active_session()
        if session:
            session.save_style_template()

    def _on_load_style(self) -> None:
        session = self.active_session()
        if session:
            session.load_style_template()

    # ------------------------------------------------------------------
    # Update check
    # ------------------------------------------------------------------

    def _run_update_check(self, *, silent: bool) -> None:
        """Spawn an :class:`UpdateCheckWorker`.

        Parameters
        ----------
        silent : bool
            ``True`` for the background startup check — errors are swallowed,
            and the dialog is not opened (only the title-bar hint appears).
            ``False`` for the user-triggered menu action — errors raise a
            :class:`QMessageBox.warning` and the dialog opens when a newer
            release is found.
        """
        # Refuse to start a second check while one is in flight.
        # Note: completion slots null this out *before* the worker is
        # deleteLater'd, so isRunning() is never called on a deleted QObject.
        if self._update_worker is not None and self._update_worker.isRunning():
            return
        if not silent:
            self._act_check_updates.setEnabled(False)
            self._statusbar.showMessage('Checking for updates…')

        worker = UpdateCheckWorker(self)
        self._update_worker = worker
        worker.finished.connect(lambda info: self._on_update_check_done(info, silent=silent))
        worker.error.connect(lambda msg: self._on_update_check_error(msg, silent=silent))
        worker.start()

    def _on_update_check_done(self, info: dict, *, silent: bool) -> None:
        self._discard_update_worker()
        if not silent:
            self._act_check_updates.setEnabled(True)
            self._statusbar.showMessage('Ready')
        if is_newer(info['latest_version'], __version__):
            self._set_title('(updates available)')
            if not silent:
                UpdateAvailableDialog(info, self).exec()
        elif not silent:
            QMessageBox.information(
                self,
                'Up to date',
                f"You're on the latest version ({__version__}).",
            )

    def _on_update_check_error(self, msg: str, *, silent: bool) -> None:
        self._discard_update_worker()
        if silent:
            return  # Offline / rate-limited / etc. — stay quiet.
        self._act_check_updates.setEnabled(True)
        self._statusbar.showMessage('Ready')
        QMessageBox.warning(self, 'Could not check for updates', msg)

    def _discard_update_worker(self) -> None:
        """Clear the worker reference and schedule the QObject for deletion.

        Must run before ``deleteLater()`` so a later call to
        :meth:`_run_update_check` doesn't try to invoke ``isRunning()`` on
        an already-deleted C++ object.
        """
        worker = self._update_worker
        self._update_worker = None
        if worker is not None:
            worker.deleteLater()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Block until any in-flight update check finishes before closing.

        The startup check can still be running when the window closes (e.g.
        the GitHub request is waiting on its 5 s socket timeout). Destroying
        a running ``QThread`` aborts the app with
        ``QThread: Destroyed while thread is still running``. ``wait()`` is
        bounded by the worker's own 5 s timeout, so this is a short pause.
        """
        worker = self._update_worker
        if worker is not None and worker.isRunning():
            worker.wait()
        super().closeEvent(event)


def _app_icon_path() -> str | None:
    """Locate the bundled app icon for both source runs and PyInstaller bundles."""
    import os
    import sys

    base = getattr(sys, '_MEIPASS', None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ('AppIcon.ico', 'AppIcon.png'):
        p = os.path.join(base, 'assets', name)
        if os.path.exists(p):
            return p
    return None


def launch() -> None:
    """Entry point — create the QApplication and launch the main window."""
    import sys

    # Force English locale globally: dot as decimal separator, comma as thousands
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

    # Set org/app identity before QApplication so QSettings (used by
    # gui.preferences) lands in the correct per-user location.
    QCoreApplication.setOrganizationName(ORG_NAME)
    QCoreApplication.setApplicationName(APP_NAME)

    # Windows: pin an explicit AppUserModelID so the taskbar groups under us, not python.exe.
    if sys.platform == 'win32':
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f'{ORG_NAME}.{APP_NAME}')
        except Exception:
            pass

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(_APP_QSS)

    icon_path = _app_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))

    # Block non-focused spinboxes from stealing wheel events while the
    # user scrolls the sidebar. Kept alive on the app so Qt doesn't GC it.
    app._spinbox_wheel_filter = _SpinBoxWheelRedirect(app)  # type: ignore[attr-defined]
    app.installEventFilter(app._spinbox_wheel_filter)  # type: ignore[attr-defined]

    window = FittingMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    launch()
