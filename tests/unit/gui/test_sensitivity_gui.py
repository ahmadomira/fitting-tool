"""GUI smoke tests for the Ka-sensitivity panel and widget.

Pins the wiring the compute-layer tests can't reach:

- :class:`SensitivityPanel` rebuilds its per-input ±% rows (titrant + every
  concentration / binding-constant condition, including ``Ka_dye``) when the
  assay type changes, hides the heatmap controls outside heatmap mode, filters
  0 % inputs out of the emitted config, and disables itself for ``DYE_ALONE``.
- :class:`SensitivityWidget` renders a synthetic JOINT result and a synthetic
  HEATMAP result without raising, and restyles/clears cleanly.

Require a ``QApplication`` (the ``qapp`` fixture).
"""

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from core.assays.registry import AssayType  # noqa: E402
from core.pipeline.sensitivity import (  # noqa: E402
    TITRANT,
    SensitivityConfig,
    SensitivityMode,
    SensitivityResult,
)

# ---------------------------------------------------------------------------
# SensitivityPanel
# ---------------------------------------------------------------------------


def test_set_assay_type_rebuilds_input_rows(qapp):
    """The ±% rows track the assay's perturbable inputs (titrant + conditions)."""
    from gui.widgets.sensitivity_panel import SensitivityPanel

    panel = SensitivityPanel()

    panel.set_assay_type(AssayType.IDA)
    assert set(panel._delta_spins) == {TITRANT, 'Ka_dye', 'h0', 'd0'}
    # Heatmap axis combos offer exactly those inputs.
    assert panel._x_combo.count() == len(panel._delta_spins)

    panel.set_assay_type(AssayType.GDA)
    assert set(panel._delta_spins) == {TITRANT, 'Ka_dye', 'h0', 'g0'}

    # A single-condition assay: just the titrant + its one fixed concentration.
    panel.set_assay_type(AssayType.DBA_HtoD)
    assert set(panel._delta_spins) == {TITRANT, 'fixed_conc'}


def test_heatmap_controls_hidden_outside_heatmap_mode(qapp):
    """The heatmap axis group shows only in Heatmap mode; samples row is inverse."""
    from gui.widgets.sensitivity_panel import SensitivityPanel

    panel = SensitivityPanel()
    panel.set_assay_type(AssayType.IDA)

    # Default mode is Joint.
    assert panel._heatmap_group.isHidden()
    assert not panel._samples_row.isHidden()

    # Switch to Heatmap (combo index 2).
    panel._mode_combo.setCurrentIndex(2)
    assert not panel._heatmap_group.isHidden()
    assert panel._samples_row.isHidden()

    # Back to Joint.
    panel._mode_combo.setCurrentIndex(0)
    assert panel._heatmap_group.isHidden()
    assert not panel._samples_row.isHidden()


def test_current_config_drops_zero_percent_inputs(qapp):
    """Inputs left at 0 % are excluded from the emitted delta_pct."""
    from gui.widgets.sensitivity_panel import SensitivityPanel

    panel = SensitivityPanel()
    panel.set_assay_type(AssayType.IDA)

    for key, spin in panel._delta_spins.items():
        spin.setValue(7.0 if key == TITRANT else 0.0)

    cfg = panel.current_config()
    assert isinstance(cfg, SensitivityConfig)
    assert cfg.mode is SensitivityMode.JOINT
    assert cfg.delta_pct == {TITRANT: 7.0}


def test_panel_disabled_for_dye_alone(qapp):
    """DYE_ALONE has no Ka to analyse, so the whole panel is disabled."""
    from gui.widgets.sensitivity_panel import SensitivityPanel

    panel = SensitivityPanel()
    panel.set_assay_type(AssayType.IDA)
    assert panel.isEnabled()

    panel.set_assay_type(AssayType.DYE_ALONE)
    assert not panel.isEnabled()


# ---------------------------------------------------------------------------
# SensitivityWidget
# ---------------------------------------------------------------------------


def _joint_result():
    rng = np.random.default_rng(0)
    samples = 2e6 * (1 + 0.05 * rng.standard_normal(40))
    return SensitivityResult(
        mode=SensitivityMode.JOINT,
        assay_type='IDA',
        ka_keys=('Ka_guest',),
        baseline_ka={'Ka_guest': 2e6},
        histograms={'joint': {'Ka_guest': samples}},
        x_offsets_pct=None,
        y_offsets_pct=None,
        ka_grid=None,
        n_success=40,
        n_total=40,
        config=SensitivityConfig(SensitivityMode.JOINT, 40, {TITRANT: 5.0}, seed=0),
    )


def _heatmap_result():
    x = np.linspace(-5.0, 5.0, 3)
    y = np.linspace(-5.0, 5.0, 3)
    grid = np.array(
        [
            [1.8e6, 1.9e6, 2.0e6],
            [1.9e6, 2.0e6, 2.1e6],
            [2.0e6, np.nan, 2.2e6],  # one failed-fit cell (transparent path)
        ]
    )
    return SensitivityResult(
        mode=SensitivityMode.HEATMAP,
        assay_type='IDA',
        ka_keys=('Ka_guest',),
        baseline_ka={'Ka_guest': 2e6},
        histograms=None,
        x_offsets_pct=x,
        y_offsets_pct=y,
        ka_grid={'Ka_guest': grid},
        n_success=8,
        n_total=9,
        config=SensitivityConfig(
            SensitivityMode.HEATMAP, 1, {TITRANT: 5.0, 'h0': 5.0}, x_key=TITRANT, y_key='h0', n_steps=3
        ),
    )


def test_widget_renders_joint_result(qapp):
    from gui.plotting.sensitivity_widget import SensitivityWidget

    w = SensitivityWidget()
    w.update_result(_joint_result())
    # Landed on the results page (not placeholder/progress).
    assert w._stack.currentWidget() is w._results_host

    # Restyling with a linear Ka axis must also render cleanly.
    w.apply_style({'distribution': {'ka_scale': 'linear'}})
    assert w._stack.currentWidget() is w._results_host


def test_widget_renders_heatmap_result_with_nan_cell(qapp):
    from gui.plotting.sensitivity_widget import SensitivityWidget

    w = SensitivityWidget()
    w.update_result(_heatmap_result())  # must not raise despite the NaN cell
    assert w._stack.currentWidget() is w._results_host


def test_widget_progress_and_clear(qapp):
    from gui.plotting.sensitivity_widget import SensitivityWidget

    w = SensitivityWidget()
    w.show_running()
    assert w._stack.currentWidget() is w._progress_page
    w.set_progress(3, 10)
    assert w._progress_bar.value() == 3

    w.update_result(_joint_result())
    w.clear()
    assert w._stack.currentWidget() is w._placeholder
