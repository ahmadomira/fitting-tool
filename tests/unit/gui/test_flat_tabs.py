"""Flat/minimal tab widget behaviour (``gui/widgets/flat_tabs.py``).

Behavioural contracts for the shared ``FlatTabWidget``: the inline "+" lives
after the last tab (not in a corner) and invokes its callback, and the bar can
never be emptied. Visual fidelity (colours, underline) is checked by eye against
the spec, not asserted here.
"""

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QTabBar, QWidget

from gui.widgets.flat_tabs import FlatTabBar, FlatTabWidget


def _lay_out(tabs, size=None):
    """Show *tabs* off-screen so Qt resolves real geometry.

    These contracts are geometric — the inline "+" sits after the last tab,
    overflow scrolls — and Qt only computes ``tabRect()`` and child-widget
    positions once a widget is shown. ``WA_DontShowOnScreen`` gets that layout
    without putting a window on the screen of whoever is running the suite,
    and without needing a display server.
    """
    if size is not None:
        tabs.resize(*size)
    tabs.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    tabs.show()


def test_flat_config(qapp):
    tabs = FlatTabWidget()
    assert tabs.objectName() == 'flatTabs'
    assert tabs.documentMode()
    assert isinstance(tabs.tabBar(), FlatTabBar)
    assert not tabs.tabBar().expanding()  # left-aligned, not stretched
    assert not tabs.tabBar().drawBase()  # no platform base line
    # Overflow: keep full labels and wheel-scroll (don't elide); chevrons hidden.
    assert tabs.usesScrollButtons()
    assert tabs.tabBar().elideMode() == Qt.TextElideMode.ElideNone


def test_inline_add_button_after_last_tab_and_calls_back(qapp):
    calls = []
    tabs = FlatTabWidget(add_callback=lambda: calls.append(1))
    tabs.addTab(QWidget(), 'A')
    tabs.addTab(QWidget(), 'B')
    _lay_out(tabs, (400, 80))
    qapp.processEvents()
    try:
        # Inline, not a corner widget (spec §5.1): the "+" is a child of the bar.
        assert tabs.cornerWidget(Qt.Corner.TopRightCorner) is None
        btn = tabs.tabBar()._add_button
        assert btn is not None and btn.parent() is tabs.tabBar()
        # Glued to the right of the last tab.
        assert btn.x() >= tabs.tabBar().tabRect(tabs.count() - 1).right()
        # Clicking runs the callback (the app wires this to "new session").
        btn.click()
        assert calls == [1]
    finally:
        tabs.close()


def _close_button(bar, i):
    for side in (QTabBar.ButtonPosition.LeftSide, QTabBar.ButtonPosition.RightSide):
        b = bar.tabButton(i, side)
        if b is not None:
            return b
    return None


def test_never_zero_tabs_hides_the_sole_close_button(qapp):
    tabs = FlatTabWidget(closable=True)
    tabs.addTab(QWidget(), 'only')
    _lay_out(tabs)
    qapp.processEvents()
    try:
        bar = tabs.tabBar()
        # Sole tab: its close control is hidden so it cannot be closed to empty.
        assert not _close_button(bar, 0).isVisible()
        # With a second tab, both close controls are shown again.
        tabs.addTab(QWidget(), 'second')
        qapp.processEvents()
        assert _close_button(bar, 0).isVisible()
        assert _close_button(bar, 1).isVisible()
    finally:
        tabs.close()


def test_plot_style_widget_has_no_add_button_and_is_fixed(qapp):
    # The plot tabs (Fit Curve / Distributions) are fixed: no "+", not closable,
    # not movable.
    tabs = FlatTabWidget(white_pane=True)
    tabs.addTab(QWidget(), 'Fit Curve')
    tabs.addTab(QWidget(), 'Distributions')
    assert tabs.tabBar()._add_button is None
    assert not tabs.tabsClosable()
    assert not tabs.isMovable()


def test_editable_double_click_commits_rename(qapp):
    renamed = []
    tabs = FlatTabWidget(editable=True)
    tabs.tab_renamed.connect(lambda i, name: renamed.append((i, name)))
    tabs.addTab(QWidget(), 'Untitled')
    _lay_out(tabs)
    qapp.processEvents()
    try:
        tabs.tabBarDoubleClicked.emit(0)  # double-click tab 0 → inline editor
        editor = tabs._editor
        assert editor is not None
        editor.setText('Control run')
        editor._finish(True)  # commit (Enter / focus-out)
        assert renamed == [(0, 'Control run')]
        assert tabs._editor is None  # editor torn down
    finally:
        tabs.close()


def test_editable_rename_escape_cancels(qapp):
    renamed = []
    tabs = FlatTabWidget(editable=True)
    tabs.tab_renamed.connect(lambda i, name: renamed.append((i, name)))
    tabs.addTab(QWidget(), 'Untitled')
    _lay_out(tabs)
    qapp.processEvents()
    try:
        tabs.tabBarDoubleClicked.emit(0)
        tabs._editor.setText('X')
        tabs._editor._finish(False)  # Escape → cancel, no rename
        assert renamed == []
        assert tabs._editor is None
    finally:
        tabs.close()


def test_wheel_scrolls_overflowing_tabs(qapp):
    tabs = FlatTabWidget()
    for i in range(20):
        tabs.addTab(QWidget(), f'tryptamine_{i}')
    _lay_out(tabs, (360, 60))
    qapp.processEvents()
    bar = tabs.tabBar()

    def wheel(d):
        ev = QWheelEvent(
            QPointF(150, 20),
            QPointF(150, 20),
            QPoint(0, d),
            QPoint(0, d),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        qapp.sendEvent(bar, ev)
        qapp.processEvents()

    try:
        start = bar.tabRect(0).x()
        wheel(-120)  # wheel away → scroll toward later tabs (strip slides left)
        scrolled = bar.tabRect(0).x()
        assert scrolled < start
        wheel(120)  # wheel back
        assert bar.tabRect(0).x() > scrolled
    finally:
        tabs.close()
