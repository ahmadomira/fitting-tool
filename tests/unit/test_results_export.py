"""Tests for the TXT report and CSV table exports in gui.session (no Qt required).

These guard the *reported* numbers, which are the point of the exports: the
estimate is a real fit, the range is the pool's own extent, and log₁₀ rows are
computed in log space. They also pin that a result with no pool says so rather
than emitting a fabricated range.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from core.pipeline.fit_pipeline import FitResult
from core.units import Q_
from gui.session import export_results_csv, export_results_txt


def _result(*, pool: bool = True) -> FitResult:
    rng = np.random.default_rng(0)
    x = np.linspace(1e-6, 5e-5, 10)
    samples = None
    if pool:
        samples = {
            'Ka_guest': rng.normal(1.24e6, 9e4, 60).clip(1e5),
            'I_0': rng.normal(1000.0, 20.0, 60),
        }
    return FitResult(
        parameters={'Ka_guest': Q_(1.24e6, '1/M'), 'I_0': Q_(1000.0, 'au')},
        rmse=12.3,
        r_squared=0.9962,
        n_passing=60 if pool else 1,
        n_total=100,
        x_fit=Q_(x, 'M'),
        y_fit=Q_(x, 'au'),
        assay_type='IDA',
        model_name='equilibrium_4param',
        source_file='IDA_system.txt',
        parameter_samples=samples,
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def test_txt_report_states_what_it_reports(tmp_path: Path) -> None:
    """The old report printed '± Uncert.' without saying what the ± was."""
    path = tmp_path / 'r.txt'
    export_results_txt([_result()], path)
    text = path.read_text(encoding='utf-8')

    assert 'Estimate = the single best fit (highest R²) of 60 accepted fits.' in text
    assert 'Range    = (min, max) across those 60 fits.' in text
    assert '± Uncert.' not in text
    # Every column of the summary table is carried, across the two blocks.
    for header in ('Estimate', 'Range (min, max)', 'Median ± MAD', 'Mean ± SD', '68% Range [p16, p84]', 'Units'):
        assert header in text


def test_txt_report_range_is_the_pool_extent(tmp_path: Path) -> None:
    result = _result()
    path = tmp_path / 'r.txt'
    export_results_txt([result], path)
    line = next(ln for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip().startswith('Ka_guest'))

    pool = result.parameter_samples['Ka_guest']
    assert f'{np.min(pool):.4g}' in line
    assert f'{np.max(pool):.4g}' in line


def test_txt_report_without_pool_says_so(tmp_path: Path) -> None:
    path = tmp_path / 'r.txt'
    export_results_txt([_result(pool=False)], path)
    text = path.read_text(encoding='utf-8')

    assert 'Range    = unavailable: no fit pool is stored for this result.' in text
    assert 'SPREAD ACROSS THE ACCEPTED-FIT POOL' not in text


def test_csv_is_one_row_per_reported_parameter(tmp_path: Path) -> None:
    result = _result()
    path = tmp_path / 'r.csv'
    export_results_csv([result], path)
    rows = _csv_rows(path)

    # Two parameters, plus the log₁₀ twin of the association constant.
    assert [r['parameter'] for r in rows] == ['Ka_guest', 'log10(Ka_guest)', 'I_0']
    assert {r['unit'] for r in rows} == {'1 / molar', '', 'au'}
    assert all(r['n_passing'] == '60' and r['assay_type'] == 'IDA' for r in rows)


def test_csv_values_match_the_pool(tmp_path: Path) -> None:
    result = _result()
    path = tmp_path / 'r.csv'
    export_results_csv([result], path)
    row = next(r for r in _csv_rows(path) if r['parameter'] == 'Ka_guest')

    pool = result.parameter_samples['Ka_guest']
    assert float(row['estimate']) == pytest.approx(1.24e6)
    assert float(row['min']) == pytest.approx(np.min(pool))
    assert float(row['max']) == pytest.approx(np.max(pool))
    assert float(row['median']) == pytest.approx(np.median(pool))
    # The reported estimate is a real fit, so it lies inside the pool's extent.
    assert float(row['min']) <= float(row['estimate']) <= float(row['max'])


def test_csv_log_row_is_computed_in_log_space(tmp_path: Path) -> None:
    """log₁₀ statistics come from the per-fit log values, never log of a spread."""
    result = _result()
    path = tmp_path / 'r.csv'
    export_results_csv([result], path)
    row = next(r for r in _csv_rows(path) if r['parameter'] == 'log10(Ka_guest)')

    logs = np.log10(result.parameter_samples['Ka_guest'])
    assert float(row['median']) == pytest.approx(np.median(logs))
    assert float(row['std']) == pytest.approx(np.std(logs, ddof=1))
    # Independent check that this is not log10 of the linear-space spread.
    linear_std = np.std(result.parameter_samples['Ka_guest'], ddof=1)
    assert float(row['std']) != pytest.approx(np.log10(linear_std))


def test_csv_leaves_pool_columns_blank_without_a_pool(tmp_path: Path) -> None:
    path = tmp_path / 'r.csv'
    export_results_csv([_result(pool=False)], path)
    rows = _csv_rows(path)

    assert [r['parameter'] for r in rows] == ['Ka_guest', 'I_0']  # no log twin without a pool
    for row in rows:
        assert row['estimate']
        assert row['min'] == row['max'] == row['median'] == ''


def test_csv_indexes_multiple_results(tmp_path: Path) -> None:
    path = tmp_path / 'r.csv'
    export_results_csv([_result(), _result()], path)
    rows = _csv_rows(path)

    assert {r['result_index'] for r in rows} == {'0', '1'}
