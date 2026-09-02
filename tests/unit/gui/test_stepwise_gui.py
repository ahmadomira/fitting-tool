"""GUI integration for the stepwise HG2 / H2G assays.

Confirms the new assay types are selectable, map to the right assay class and
condition fields, carry display labels and help text, and that the values the
config panel produces construct a valid assay (the full GUI → fit path).
"""

import numpy as np
import pytest

from core.assays import AssayType, H2GAssay, HG2Assay
from core.units import Q_, Quantity
from gui.assay_descriptions import ASSAY_DESCRIPTIONS
from gui.parameter_descriptions import PARAMETER_DESCRIPTIONS
from gui.plotting.labels import fmt_param
from gui.widgets.assay_conditions import ASSAY_CONDITIONS, assay_class, condition_fields

STEPWISE = {AssayType.DBA_HG2: HG2Assay, AssayType.DBA_H2G: H2GAssay}


class TestRegistration:
    @pytest.mark.parametrize('assay_type, expected_cls', STEPWISE.items())
    def test_selectable_and_mapped(self, assay_type, expected_cls):
        assert assay_type in ASSAY_CONDITIONS
        assert assay_class(assay_type) is expected_cls

    @pytest.mark.parametrize('assay_type', STEPWISE)
    def test_single_host_condition(self, assay_type):
        fields = condition_fields(assay_type)
        assert [f.key for f in fields] == ['h0']
        assert fields[0].unit_type == 'concentration'


class TestLabelsAndHelp:
    def test_every_stepwise_param_has_label_and_description(self):
        """Each stepwise parameter renders a real HTML label, not the raw-key fallback."""
        for key in ('Ka_HG', 'Ka_HG2', 'Ka_H2G', 'I_G', 'I_H', 'I_HG', 'I_HG2', 'I_H2G'):
            assert fmt_param(key) != key, key
            assert key in PARAMETER_DESCRIPTIONS, key

    @pytest.mark.parametrize('assay_type', STEPWISE)
    def test_assay_has_description(self, assay_type):
        assert assay_type.name in ASSAY_DESCRIPTIONS


class TestConfigPanelToAssay:
    """Selecting a stepwise assay in the panel yields conditions that build a
    valid assay — the same hand-off the fit worker performs."""

    @pytest.mark.parametrize('assay_type, expected_cls', STEPWISE.items())
    def test_panel_conditions_build_assay(self, qapp, assay_type, expected_cls):
        from gui.widgets.assay_config_panel import AssayConfigPanel

        panel = AssayConfigPanel()
        panel.set_assay_type(assay_type)

        conditions = panel.current_conditions()
        assert set(conditions) == {'h0'}
        assert isinstance(conditions['h0'], Quantity)

        x = Q_(np.linspace(1e-6, 1e-3, 10), 'M')
        y = Q_(np.ones(10), 'au')
        assay = expected_cls(x_data=x, y_data=y, **conditions)
        assert assay.h0.to('M').magnitude > 0
