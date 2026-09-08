"""Reusable live controls for the simulation applet.

These are generic and assay-agnostic — they render whatever :class:`SimKnob` /
concentration / noise configuration they are given, so the same widgets serve
every assay.

* :class:`ParameterControl` — one knob: a labelled ``min``/value/``max`` triple
  around a slider.  Log slider for association constants (with an optional
  ``Ka ⟷ log₁₀ Ka`` display toggle), linear otherwise; a nM/µM/mM/M selector for
  concentrations; scientific-notation entry for large-magnitude quantities;
  negatives allowed for signal coefficients (e.g. a calibration intercept).
* :class:`ConcentrationInput` — the titrant vector, via explicit / linear /
  start-step / log modes.
* :class:`NoiseControl` — optional Gaussian measurement noise (fraction of the
  signal range) with replicas and a seed.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.units import Q_
from gui.simulation.sim_knob import SimKnob
from gui.widgets.numeric_inputs import NoScrollDoubleSpinBox, NoScrollSpinBox, SciLineEdit

_SLIDER_STEPS = 1000

# nM/µM/mM/M selector for concentration knobs/inputs.  The label set is a UI
# choice; the conversion factors are always derived from Pint (_display_factor),
# never hardcoded.
_CONC_LABELS: tuple[str, ...] = ('nM', 'µM', 'mM', 'M')
_CONC_DEFAULT = 'µM'


def _unit_text(unit) -> str:
    """Short Unicode unit label for a static (non-selectable) unit."""
    return f'{unit:~P}'


def _labelled(field: QWidget, caption: str, *, before: bool) -> QWidget:
    """Wrap a numeric field with a small muted ``min``/``max`` caption."""
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(3)
    lbl = QLabel(caption)
    lbl.setStyleSheet('color: palette(mid);')
    lbl.setToolTip(f'{caption} of the slider range')
    if before:
        lay.addWidget(lbl)
        lay.addWidget(field)
    else:
        lay.addWidget(field)
        lay.addWidget(lbl)
    return box


class ParameterControl(QWidget):
    """A single live knob: slider + labelled editable bounds + exact value.

    Internals are kept in **base units** (M / M⁻¹ / au / au·M⁻¹); for concentration
    knobs a unit selector only rescales the *display*, and for association constants
    an optional ``log₁₀`` toggle shows/edits the exponent instead (the slider is
    geometric either way, so log₁₀ is just a linear readout of the same handle).
    ``changed`` fires on every user adjustment.
    """

    changed = pyqtSignal()

    def __init__(self, knob: SimKnob, parent=None):
        super().__init__(parent)
        self._knob = knob
        self._log = knob.log
        self._value = float(knob.default)
        self._vmin = float(knob.vmin)
        self._vmax = float(knob.vmax)
        self._scale = _display_factor(_CONC_DEFAULT, knob.unit) if knob.is_concentration else 1.0
        self._log10_display = False  # association constants only: show log₁₀(Ka)

        # Signal coefficients (and log knobs in log₁₀ view) may be negative.
        allow_neg = knob.allows_negative or knob.log

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        # Row 0: name (+ log₁₀ toggle for Ka) | value | unit.
        name = QLabel(knob.label)
        name.setTextFormat(Qt.TextFormat.RichText)
        name.setToolTip(knob.tooltip)
        name_cell = QWidget()
        name_lay = QHBoxLayout(name_cell)
        name_lay.setContentsMargins(0, 0, 0, 0)
        name_lay.setSpacing(4)
        name_lay.addWidget(name)
        if self._log:
            self._log_toggle: QCheckBox | None = QCheckBox('log₁₀')
            self._log_toggle.setToolTip('Show and edit this constant as log₁₀(Ka)')
            self._log_toggle.toggled.connect(self._on_log10_toggled)
            name_lay.addWidget(self._log_toggle)
        else:
            self._log_toggle = None
        name_lay.addStretch()

        self._value_spin = SciLineEdit(allow_negative=allow_neg)
        self._value_spin.setToolTip(knob.tooltip)
        self._value_spin.setMinimumWidth(90)
        self._value_spin.setStyleSheet('border: 1px solid palette(highlight); border-radius: 3px;')
        vf = self._value_spin.font()
        vf.setBold(True)
        self._value_spin.setFont(vf)

        grid.addWidget(name_cell, 0, 0)
        grid.addWidget(self._value_spin, 0, 1)
        if knob.is_concentration:
            self._unit_combo: QComboBox | None = QComboBox()
            self._unit_combo.addItems(list(_CONC_LABELS))
            self._unit_combo.setCurrentText(_CONC_DEFAULT)
            self._unit_combo.currentTextChanged.connect(self._on_unit_changed)
            grid.addWidget(self._unit_combo, 0, 2)
            self._unit_label: QLabel | None = None
        else:
            self._unit_combo = None
            self._unit_label = QLabel(_unit_text(knob.unit))
            grid.addWidget(self._unit_label, 0, 2)

        # Row 1: min (labelled) | slider | max (labelled).
        self._min_spin = SciLineEdit(allow_negative=allow_neg)
        self._max_spin = SciLineEdit(allow_negative=allow_neg)
        for sb in (self._min_spin, self._max_spin):
            sb.setMinimumWidth(74)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, _SLIDER_STEPS)

        grid.addWidget(_labelled(self._min_spin, 'min', before=True), 1, 0)
        grid.addWidget(self._slider, 1, 1)
        grid.addWidget(_labelled(self._max_spin, 'max', before=False), 1, 2)

        self._sync_all()

        self._slider.valueChanged.connect(self._on_slider)
        self._value_spin.value_changed.connect(self._on_value_edit)
        self._min_spin.value_changed.connect(self._on_bounds_edit)
        self._max_spin.value_changed.connect(self._on_bounds_edit)

    # -- public API ----------------------------------------------------------

    def value(self) -> float:
        """Current value in the base unit (M / M⁻¹ / au / au·M⁻¹)."""
        return self._value

    def quantity(self):
        """Current value as a pint Quantity (for assay conditions)."""
        return Q_(self._value, self._knob.unit)

    def state(self) -> dict:
        """Serializable knob state for settings save."""
        return {'value': self._value, 'min': self._vmin, 'max': self._vmax}

    def load_state(self, state: dict) -> None:
        self._vmin = float(state.get('min', self._vmin))
        self._vmax = float(state.get('max', self._vmax))
        self._value = _clamp(float(state.get('value', self._value)), self._vmin, self._vmax)
        self._sync_all()
        self.changed.emit()

    # -- display <-> base transform (unit scale, or log₁₀ for Ka) ------------

    def _to_display(self, base: float) -> float:
        if self._knob.is_concentration:
            return base / self._scale
        if self._log and self._log10_display:
            return math.log10(base) if base > 0 else 0.0
        return base

    def _from_display(self, shown: float) -> float:
        if self._knob.is_concentration:
            return shown * self._scale
        if self._log and self._log10_display:
            return 10.0**shown
        return shown

    # -- slider <-> value mapping (always in base units) --------------------

    def _pos_to_value(self, pos: int) -> float:
        f = pos / _SLIDER_STEPS
        if self._log and self._vmin > 0 and self._vmax > self._vmin:
            lo, hi = math.log10(self._vmin), math.log10(self._vmax)
            return 10 ** (lo + f * (hi - lo))
        return self._vmin + f * (self._vmax - self._vmin)

    def _value_to_pos(self, v: float) -> int:
        if self._vmax <= self._vmin:
            return 0
        if self._log and self._vmin > 0 and self._vmax > self._vmin:
            lo, hi = math.log10(self._vmin), math.log10(self._vmax)
            f = (math.log10(max(v, self._vmin)) - lo) / (hi - lo)
        else:
            f = (v - self._vmin) / (self._vmax - self._vmin)
        return int(round(_clamp(f, 0.0, 1.0) * _SLIDER_STEPS))

    # -- sync helpers (SciLineEdit.setValue never emits) --------------------

    def _sync_all(self) -> None:
        self._sync_bounds_spins()
        self._sync_value_spin()
        self._sync_slider()

    def _sync_value_spin(self) -> None:
        self._value_spin.setValue(self._to_display(self._value))

    def _sync_bounds_spins(self) -> None:
        self._min_spin.setValue(self._to_display(self._vmin))
        self._max_spin.setValue(self._to_display(self._vmax))

    def _sync_slider(self) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self._value_to_pos(self._value))
        self._slider.blockSignals(False)

    # -- slots ---------------------------------------------------------------

    def _on_slider(self, pos: int) -> None:
        self._value = self._pos_to_value(pos)
        self._sync_value_spin()
        self.changed.emit()

    def _on_value_edit(self) -> None:
        self._value = _clamp(self._from_display(self._value_spin.value()), self._vmin, self._vmax)
        self._sync_value_spin()  # reflect any clamping / canonical formatting
        self._sync_slider()
        self.changed.emit()

    def _on_bounds_edit(self) -> None:
        vmin = self._from_display(self._min_spin.value())
        vmax = self._from_display(self._max_spin.value())
        if self._log:
            vmin = max(vmin, 1e-12)  # log slider needs a positive floor
        if vmax <= vmin:
            vmax = vmin * 10 if self._log else vmin + abs(vmin or 1.0)
        self._vmin, self._vmax = vmin, vmax
        self._value = _clamp(self._value, vmin, vmax)
        self._sync_all()
        self.changed.emit()

    def _on_unit_changed(self, label: str) -> None:
        self._scale = _display_factor(label, self._knob.unit)
        self._sync_value_spin()
        self._sync_bounds_spins()

    def _on_log10_toggled(self, on: bool) -> None:
        self._log10_display = on
        if self._unit_label is not None:
            # log₁₀ of a dimensioned Ka is only defined on its numeric value: the
            # displayed quantity is log₁₀(K / M⁻¹), which is dimensionless.
            self._unit_label.setText('log₁₀(K/M⁻¹)' if on else _unit_text(self._knob.unit))
        # Value unchanged — only its representation flips; no recompute needed.
        self._sync_value_spin()
        self._sync_bounds_spins()


class TitrantInput(QWidget):
    """Titrant setup: a maximum concentration (0 → max in N points), or an explicit vector.

    The applet needs only two ways to define the titration — scan from zero up to a
    chosen concentration, or use exactly the concentrations the user lists.  The maximum
    is a normal concentration knob (slider + bounds + unit), so it looks and behaves like
    the fixed-concentration controls; ticking *Custom concentration vector* swaps in an
    explicit list.  ``spec()`` returns ``(mode, kwargs)`` in M, ready for
    :func:`core.simulation.build_concentration_vector`.
    """

    changed = pyqtSignal()

    N_POINTS = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self._max_knob: SimKnob | None = None
        self._max_control: ParameterControl | None = None
        self._vector_scale = _display_factor(_CONC_DEFAULT, 'M')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # The maximum-concentration knob is rebuilt per assay (its label + default follow
        # the titrant); it sits in this holder so the rest of the row persists.
        self._max_holder = QWidget()
        self._max_layout = QVBoxLayout(self._max_holder)
        self._max_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._max_holder)

        self._custom = QCheckBox('Custom concentration vector')
        self._custom.setToolTip('Simulate at exactly these concentrations instead of a 0 → max scan')
        self._custom.toggled.connect(self._on_custom_toggled)
        layout.addWidget(self._custom)

        self._vector = QLineEdit()
        self._vector.setPlaceholderText('e.g. 0, 1, 2.5, 5, 10')
        self._vector.textChanged.connect(self.changed)
        self._vector_unit = QComboBox()
        self._vector_unit.addItems(list(_CONC_LABELS))
        self._vector_unit.setCurrentText(_CONC_DEFAULT)
        self._vector_unit.currentTextChanged.connect(self._on_vector_unit_changed)
        self._vector_row = QWidget()
        vrow = QHBoxLayout(self._vector_row)
        vrow.setContentsMargins(0, 0, 0, 0)
        vrow.addWidget(self._vector, 1)
        vrow.addWidget(self._vector_unit)
        layout.addWidget(self._vector_row)

        self._on_custom_toggled(False)

    # -- public API ----------------------------------------------------------

    def set_titrant(self, label_html: str, tooltip: str, default_max_m: float) -> None:
        """(Re)build the maximum-concentration knob for a new assay's titrant."""
        if self._max_control is not None:
            self._max_control.setParent(None)
            self._max_control.deleteLater()
        default = default_max_m if default_max_m > 0 else 1e-5
        self._max_knob = SimKnob(
            key='titrant_max',
            label=label_html,
            tooltip=tooltip,
            unit=Q_(1, 'M').units,
            log=False,
            default=default,
            vmin=0.0,
            vmax=default * 5,
            is_condition=False,
            section='concentration',
        )
        self._max_control = ParameterControl(self._max_knob)
        self._max_control.setEnabled(not self._custom.isChecked())
        self._max_control.changed.connect(self.changed)
        self._max_layout.addWidget(self._max_control)

    def spec(self) -> tuple[str, dict]:
        if self._custom.isChecked():
            return 'explicit', {'values': [v * self._vector_scale for v in _parse_floats(self._vector.text())]}
        stop = self._max_control.value() if self._max_control is not None else 0.0
        return 'linear', {'start': 0.0, 'stop': stop, 'n': self.N_POINTS}

    def set_explicit(self, values_m) -> None:
        """Switch to a custom vector from Molar values (used on data import)."""
        self._vector.blockSignals(True)
        self._vector.setText(', '.join(f'{v / self._vector_scale:.17g}' for v in values_m))
        self._vector.blockSignals(False)
        self._custom.setChecked(True)  # emits toggled → changed

    def state(self) -> dict:
        return {
            'max': self._max_control.value() if self._max_control is not None else 0.0,
            'custom': self._custom.isChecked(),
            'vector': self._vector.text(),
            'vector_unit': self._vector_unit.currentText(),
        }

    def load_state(self, s: dict) -> None:
        if self._max_control is not None and 'max' in s:
            m = float(s['max'])
            self._max_control.load_state({'value': m, 'min': 0.0, 'max': (m if m > 0 else 1e-5) * 5})
        self._vector_unit.blockSignals(True)
        self._vector_unit.setCurrentText(s.get('vector_unit', _CONC_DEFAULT))
        self._vector_unit.blockSignals(False)
        self._vector_scale = _display_factor(self._vector_unit.currentText(), 'M')
        self._vector.blockSignals(True)
        self._vector.setText(s.get('vector', ''))
        self._vector.blockSignals(False)
        self._custom.setChecked(bool(s.get('custom', False)))

    # -- internals -----------------------------------------------------------

    def _on_custom_toggled(self, on: bool) -> None:
        if self._max_control is not None:
            self._max_control.setEnabled(not on)
        self._vector_row.setVisible(on)
        self.changed.emit()

    def _on_vector_unit_changed(self) -> None:
        old = self._vector_scale
        self._vector_scale = _display_factor(self._vector_unit.currentText(), 'M')
        ratio = old / self._vector_scale
        # Preserve the physical vector across a unit switch (convert, don't reinterpret).
        try:
            values = _parse_floats(self._vector.text())
        except ValueError:
            values = []
        if values:
            self._vector.blockSignals(True)
            self._vector.setText(', '.join(f'{v * ratio:.17g}' for v in values))
            self._vector.blockSignals(False)
        self.changed.emit()


class NoiseControl(QWidget):
    """Optional Gaussian measurement noise: enable, fraction-of-range, replicas, seed."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)

        self._enable = QCheckBox('Add measurement noise')
        self._enable.toggled.connect(self._on_toggle)
        form.addRow(self._enable)

        self._frac = NoScrollDoubleSpinBox()
        self._frac.setRange(0.0, 100.0)
        self._frac.setDecimals(2)
        self._frac.setValue(5.0)
        self._frac.setSuffix(' %')
        self._frac.valueChanged.connect(self.changed)
        form.addRow('Noise', self._frac)

        self._replicas = NoScrollSpinBox()
        self._replicas.setRange(1, 50)
        self._replicas.setValue(3)
        self._replicas.valueChanged.connect(self.changed)
        form.addRow('Replicas', self._replicas)

        self._seed = NoScrollSpinBox()
        self._seed.setRange(0, 1_000_000)
        self._seed.setValue(0)
        self._seed.valueChanged.connect(self.changed)
        form.addRow('Seed', self._seed)

        self._on_toggle(False)

    def _on_toggle(self, on: bool) -> None:
        for w in (self._frac, self._replicas, self._seed):
            w.setEnabled(on)
        self.changed.emit()

    def settings(self) -> dict:
        return {
            'enabled': self._enable.isChecked(),
            'frac': self._frac.value() / 100.0,
            'replicas': self._replicas.value(),
            'seed': self._seed.value(),
        }

    def load_settings(self, s: dict) -> None:
        for w, val in (
            (self._enable, bool(s.get('enabled', False))),
            (self._frac, float(s.get('frac', 0.05)) * 100.0),
            (self._replicas, int(s.get('replicas', 3))),
            (self._seed, int(s.get('seed', 0))),
        ):
            w.blockSignals(True)
            w.setChecked(val) if w is self._enable else w.setValue(val)
            w.blockSignals(False)
        self._on_toggle(self._enable.isChecked())


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _display_factor(label: str, base) -> float:
    """Pint-derived scale: magnitude of ``1 <label>`` expressed in ``base`` units.

    ``base`` is a pint ``Unit`` (a knob's canonical unit) or a unit string. The
    shared registry (``core.units``) is the single source of every factor.
    """
    return float(Q_(1, label).to(base).magnitude)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _parse_floats(text: str) -> list[float]:
    """Parse a comma/space-separated list of floats; raise ValueError if any token is bad."""
    tokens = [t for t in text.replace(',', ' ').split() if t]
    if not tokens:
        return []
    try:
        return [float(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(f'Invalid concentration value in: {text!r}') from exc
