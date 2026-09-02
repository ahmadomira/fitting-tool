"""Two-level (category → subtype) assay selector behaviour.

Verifies the dependent dropdowns react correctly and that exactly one
``assay_type_changed`` is emitted per selection — the contract the fitting
session and demo loader rely on.
"""

import pytest

from core.assays import AssayType
from core.assays.registry import ASSAY_REGISTRY
from core.units import Q_
from gui.widgets.assay_conditions import ConditionField
from gui.widgets.assay_config_panel import AssayConfigPanel, _UnitWidget


@pytest.fixture
def panel(qapp):
    return AssayConfigPanel()


def _direct_binding_types():
    return [at for at, m in ASSAY_REGISTRY.items() if m.category == 'Direct Binding']


def test_default_is_gda_without_emission(qapp):
    emitted = []
    p = AssayConfigPanel()
    p.assay_type_changed.connect(emitted.append)
    # Construction must not fire the signal (matches prior single-combo init).
    assert p.current_assay_type() is AssayType.GDA
    assert emitted == []


def test_set_assay_type_selects_category_and_subtype_once(panel):
    emitted = []
    panel.assay_type_changed.connect(emitted.append)

    panel.set_assay_type(AssayType.IDA)

    assert panel.current_assay_type() is AssayType.IDA
    assert panel._main_combo.currentData() == 'Competitive Displacement'
    assert panel._sub_combo.currentData() is AssayType.IDA
    assert emitted == [AssayType.IDA]  # exactly one emission


def test_changing_category_repopulates_subtypes_and_emits_once(panel):
    emitted = []
    panel.assay_type_changed.connect(emitted.append)

    panel._main_combo.setCurrentIndex(panel._main_combo.findData('Direct Binding'))

    direct = _direct_binding_types()
    sub_data = [panel._sub_combo.itemData(i) for i in range(panel._sub_combo.count())]
    assert sub_data == direct  # subtype combo now lists exactly the category's assays
    assert panel.current_assay_type() is direct[0]  # first subtype applied
    assert emitted == [direct[0]]  # one emission for the category switch, not two


def test_changing_subtype_within_category_emits_that_assay(panel):
    panel.set_assay_type(AssayType.DBA_HtoD)  # Direct Binding
    emitted = []
    panel.assay_type_changed.connect(emitted.append)

    panel._sub_combo.setCurrentIndex(panel._sub_combo.findData(AssayType.DBA_H2G))

    assert panel.current_assay_type() is AssayType.DBA_H2G
    assert emitted == [AssayType.DBA_H2G]


class TestUnitWidgetInputConversion:
    """The primary user-input boundary: _UnitWidget always returns a base-unit
    Quantity (M / M⁻¹) regardless of the display unit selected (G13)."""

    def test_concentration_default_returns_base_molar(self, qapp):
        w = _UnitWidget(ConditionField('h0', '[Host]', '', 50e-6, 'M', 'concentration'))
        v = w.value()
        assert v.units == Q_(1, 'M').units
        assert v.to('M').magnitude == pytest.approx(50e-6)

    # Switching a field's display unit must never alter the physical quantity —
    # for *every* offered unit, including a coarse one like 'M'. Regression for a
    # fixed 3-decimal spinbox that silently rounded a small value to 0 on switch
    # (e.g. 50 µM shown as M → 0.000), zeroing the condition and corrupting fits.
    @pytest.mark.parametrize('base_value', [1e-9, 50e-6])  # smallest (where the bug bit) + typical
    def test_switching_display_unit_preserves_quantity(self, qapp, base_value):
        w = _UnitWidget(ConditionField('h0', '[Host]', '', base_value, 'M', 'concentration'))
        for idx, (label, _) in enumerate(w._units):  # includes 'M', where the bug bit
            w._combo.setCurrentIndex(idx)
            got = w.value().to('M').magnitude
            assert got == pytest.approx(base_value, rel=1e-9), (
                f'switching to {label!r} changed {base_value} M to {got} M'
            )

    def test_spinbox_value_interpreted_in_display_unit(self, qapp):
        w = _UnitWidget(ConditionField('h0', '[Host]', '', 1e-6, 'M', 'concentration'))
        w._combo.setCurrentText('µM')
        w._spinbox.setValue(5.0)  # 5 µM
        assert w.value().to('M').magnitude == pytest.approx(5e-6)

    def test_binding_constant_returns_inverse_molar(self, qapp):
        w = _UnitWidget(ConditionField('Ka_dye', 'Ka', '', 1e6, 'M⁻¹', 'binding_constant'))
        assert w.value().to('1/M').magnitude == pytest.approx(1e6)
