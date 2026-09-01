"""Widget tests for FitSummaryWidget — requires a QApplication."""

import numpy as np
import pytest

pytest.importorskip('PyQt6')


@pytest.fixture
def minimal_fit_result():
    from core.pipeline.fit_pipeline import FitResult

    x = np.linspace(0, 1e-4, 20)
    return FitResult(
        parameters={'Ka_guest': 1e6, 'I0': 100.0, 'I_dye_free': 5e4, 'I_dye_bound': 8e4},
        rmse=0.005,
        r_squared=0.998,
        n_passing=87,
        n_total=100,
        x_fit=x,
        y_fit=x * 1.05 + 0.01,
        assay_type='GDA',
        model_name='equilibrium_4param',
    )


def test_update_result_metric_labels_nonempty(qapp, minimal_fit_result):
    from gui.plotting.fit_summary_widget import FitSummaryWidget

    widget = FitSummaryWidget()
    widget.update_result(minimal_fit_result)

    assert widget._rmse_label.text() not in ('', '—')
    assert widget._r2_label.text() not in ('', '—')
    assert widget._passing_label.text() not in ('', '—')


def test_clear_resets_state(qapp, minimal_fit_result):
    from gui.plotting.fit_summary_widget import FitSummaryWidget

    widget = FitSummaryWidget()
    widget.update_result(minimal_fit_result)
    widget.clear()

    assert widget._table.rowCount() == 0
    assert widget._rmse_label.text() == '—'
    assert widget._r2_label.text() == '—'
    assert widget._passing_label.text() == '—'


def test_unknown_assay_type_does_not_crash(qapp):
    from core.pipeline.fit_pipeline import FitResult
    from gui.plotting.fit_summary_widget import FitSummaryWidget

    x = np.linspace(0, 1e-4, 5)
    result = FitResult(
        parameters={'slope': 1.5},
        rmse=0.01,
        r_squared=0.99,
        n_passing=1,
        n_total=1,
        x_fit=x,
        y_fit=x * 1.5,
        assay_type='UNKNOWN_TYPE',
        model_name='linear',
    )
    widget = FitSummaryWidget()
    widget.update_result(result)  # must not raise

    assert widget._table.rowCount() == 1


@pytest.fixture
def full_fit_result():
    """A FitResult carrying the ensemble pool (parameter + quality samples)."""
    from core.pipeline.fit_pipeline import FitResult

    x = np.linspace(0, 1e-4, 20)
    samples = {
        'Ka_guest': np.array([1.0e6, 1.1e6, 0.9e6]),
        'I0': np.array([100.0, 90.0, 110.0]),
        'I_dye_free': np.array([5.0e4, 5.1e4, 4.9e4]),
        'I_dye_bound': np.array([8.0e4, 8.1e4, 7.9e4]),
    }
    quality = {
        'rmse': np.array([0.006, 0.005, 0.007]),
        'r_squared': np.array([0.997, 0.998, 0.996]),
    }
    return FitResult(
        # Representative = index 1 (best RMSE / R²).
        parameters={'Ka_guest': 1.1e6, 'I0': 90.0, 'I_dye_free': 5.1e4, 'I_dye_bound': 8.1e4},
        rmse=0.005,
        r_squared=0.998,
        n_passing=3,
        n_total=10,
        x_fit=x,
        y_fit=x * 1.05,
        assay_type='GDA',
        model_name='equilibrium_4param',
        parameter_samples=samples,
        quality_samples=quality,
        representative_index=1,
    )


def test_seven_columns_merged_stats_and_ranges(qapp, full_fit_result):
    from gui.plotting.fit_summary_widget import _COLUMN_HEADERS, FitSummaryWidget

    widget = FitSummaryWidget()
    widget.update_result(full_fit_result)

    assert widget._table.columnCount() == 7
    headers = [widget._table.horizontalHeaderItem(c).text() for c in range(7)]
    assert headers == list(_COLUMN_HEADERS)
    # Merged 'central ± spread' cells (Median ± MAD, Mean ± SD) render one '±'.
    for col in (2, 3):
        assert '±' in widget._table.cellWidget(0, col).text()
    # 68% Range (col 4) and Min-Max Range (col 5) both render '[lo, hi]'.
    for col in (4, 5):
        t = widget._table.cellWidget(0, col).text()
        assert t.startswith('[') and t.endswith(']') and ',' in t


def test_log10_ka_row_computed_in_log_space(qapp, full_fit_result):
    """A log₁₀(Ka) row follows the Ka row; its Estimate is log₁₀ of the
    representative Ka (checked independently), not derived from a Ka spread."""
    from gui.plotting.fit_summary_widget import FitSummaryWidget

    widget = FitSummaryWidget()
    widget.update_result(full_fit_result)

    # GDA has one log-scale key (Ka_guest) -> 4 params + 1 log row = 5 rows.
    assert widget._table.rowCount() == 5
    labels = [widget._table.cellWidget(r, 0).text() for r in range(5)]
    log_rows = [i for i, t in enumerate(labels) if t.startswith('log₁₀')]
    assert len(log_rows) == 1
    est = widget._table.cellWidget(log_rows[0], 1).text()
    assert abs(float(est) - np.log10(1.1e6)) < 0.05  # representative Ka = 1.1e6


def test_rep_combo_offers_named_choices_before_plot_selection(qapp, full_fit_result):
    from gui.plotting.fit_summary_widget import FitSummaryWidget

    widget = FitSummaryWidget()
    widget.update_result(full_fit_result)

    # Fresh fit, no plot pick yet: Best / Median / Worst only (no 'Selected').
    names = [widget._rep_combo.itemText(i).split(' ·')[0] for i in range(widget._rep_combo.count())]
    assert names == ['Best', 'Median', 'Worst']


def test_representative_selector_emits_pool_index(qapp, full_fit_result):
    from gui.plotting.fit_summary_widget import FitSummaryWidget

    widget = FitSummaryWidget()
    widget.update_result(full_fit_result)
    captured = []
    widget.representative_selected.connect(captured.append)

    # 'Worst' = lowest R² = pool index 2 (r² = [0.997, 0.998, 0.996]).
    widget._rep_combo.setCurrentIndex(2)
    assert captured == [2]


def test_selected_item_is_sticky_across_switches(qapp, full_fit_result):
    """The plot-picked fit becomes a sticky 'Selected' item that survives
    switching the representative to Best, so the user can compare and go back."""
    from dataclasses import replace

    from gui.plotting.fit_summary_widget import FitSummaryWidget

    widget = FitSummaryWidget()
    widget.update_result(full_fit_result)

    # Pick pool index 2 from the plot -> a 'Selected' item appears, and the
    # combo reflects the reported representative.
    widget.note_plot_selection(2)
    widget.update_result(replace(full_fit_result, representative_index=2))
    assert any(widget._rep_combo.itemText(i).startswith('Selected') for i in range(widget._rep_combo.count()))
    assert widget._rep_combo.currentData() == 2

    # Switch the representative to Best (index 1): 'Selected' (2) must remain.
    widget.update_result(replace(full_fit_result, representative_index=1))
    sel = [i for i in range(widget._rep_combo.count()) if widget._rep_combo.itemText(i).startswith('Selected')]
    assert len(sel) == 1
    assert widget._rep_combo.itemData(sel[0]) == 2  # still points at the plot pick


def test_new_fit_clears_sticky_selection(qapp, full_fit_result):
    """A different fit (new id) resets the sticky 'Selected' choice."""
    from dataclasses import replace

    from gui.plotting.fit_summary_widget import FitSummaryWidget

    widget = FitSummaryWidget()
    widget.update_result(full_fit_result)
    widget.note_plot_selection(2)
    widget.update_result(replace(full_fit_result, representative_index=2))
    assert any(widget._rep_combo.itemText(i).startswith('Selected') for i in range(widget._rep_combo.count()))

    fresh = replace(full_fit_result, id='different-fit-id', representative_index=1)
    widget.update_result(fresh)
    names = [widget._rep_combo.itemText(i).split(' ·')[0] for i in range(widget._rep_combo.count())]
    assert 'Selected' not in names
