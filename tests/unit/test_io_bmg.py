"""Tests for the BMG plate-reader XLSX importer and raw data writers.

The BMG reader:
    * recognises the ``Microplate End point`` sheet,
    * parses the plate grid into long-format replicas with placeholder
      concentrations ``1..N``,
    * flags the DataFrame via ``df.attrs`` so the GUI can prompt.

The measurement writers round-trip the resulting MeasurementSet through
the existing TxtReader / CsvReader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook

from core.data_processing.measurement_set import MeasurementSet
from core.io import load_measurements
from core.io.formats.bmg_reader import is_bmg_workbook, parse_bmg_workbook
from core.io.formats.measurement_writer import (
    write_measurements_csv,
    write_measurements_txt,
)

_BUNDLED_BMG = Path('data/01_ZBeta_MDAP_Dopamine_pH2.xlsx')


def _write_structured_xlsx(path: Path) -> None:
    """Write a minimal non-BMG .xlsx so the dispatcher's fallback path is exercised."""
    wb = Workbook()
    ws = wb.active
    ws.append(['concentration', 'signal', 'replica'])
    for conc, sig, rep in [
        (0.0, 100.0, 0),
        (1e-6, 200.0, 0),
        (2e-6, 300.0, 0),
        (0.0, 102.0, 1),
        (1e-6, 198.0, 1),
        (2e-6, 305.0, 1),
    ]:
        ws.append([conc, sig, rep])
    wb.save(path)


class TestBMGDetection:
    def test_structured_xlsx_is_not_bmg(self, tmp_path):
        path = tmp_path / 'structured.xlsx'
        _write_structured_xlsx(path)
        wb = load_workbook(path, data_only=True, read_only=True)
        try:
            assert is_bmg_workbook(wb) is False
        finally:
            wb.close()


def _write_synthetic_bmg_xlsx(path: Path, n_rows: int = 3, n_cols: int = 4) -> None:
    """A minimal BMG-shaped workbook: metadata rows, a column-number header
    row (col A empty, cols B.. = 1..N), then one row per plate-row letter.

    Signals are ``row*100 + col`` so every cell is identifiable in an assertion.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = 'Microplate End point'
    ws.append(['Test ID', 1234])
    ws.append(['Test Name', 'synthetic plate'])
    ws.append([])
    ws.append([None] + list(range(1, n_cols + 1)))
    for r in range(n_rows):
        ws.append([chr(ord('A') + r)] + [float(r * 100 + c + 1) for c in range(n_cols)])
    wb.save(path)


class TestBMGParseSynthetic:
    """Parser coverage that does not depend on the (untracked) real export.

    Without this the whole BMG path is unverified in a clean checkout: both
    real-file tests skip when ``data/01_ZBeta_MDAP_Dopamine_pH2.xlsx`` is absent.
    """

    def test_detects_and_parses_a_plate_grid(self, tmp_path):
        path = tmp_path / 'synthetic_bmg.xlsx'
        _write_synthetic_bmg_xlsx(path, n_rows=3, n_cols=4)

        wb = load_workbook(path, data_only=True, read_only=True)
        try:
            assert is_bmg_workbook(wb) is True
            df, meta = parse_bmg_workbook(wb)
        finally:
            wb.close()

        # One replica per plate row, one titration point per plate column.
        assert set(df['replica'].unique()) == {0, 1, 2}
        assert sorted(df['concentration'].unique().tolist()) == [1.0, 2.0, 3.0, 4.0]
        # Concentrations are positional placeholders, flagged for the GUI prompt.
        assert df.attrs.get('bmg_placeholder_concentrations') is True
        # Cell values land in the right (replica, column) slot.
        rep1 = df[df['replica'] == 1].sort_values('concentration')
        assert rep1['signal'].tolist() == pytest.approx([101.0, 102.0, 103.0, 104.0])
        assert isinstance(meta, dict)

    def test_sheet_without_a_column_header_row_raises(self, tmp_path):
        """A Microplate sheet whose header row is missing must fail loudly."""
        wb = Workbook()
        ws = wb.active
        ws.title = 'Microplate End point'
        ws.append(['Test ID', 1234])
        ws.append(['A', 1.0, 2.0])
        path = tmp_path / 'no_header.xlsx'
        wb.save(path)

        wb2 = load_workbook(path, data_only=True, read_only=True)
        try:
            with pytest.raises(ValueError, match='header row'):
                parse_bmg_workbook(wb2)
        finally:
            wb2.close()


class TestBMGParse:
    def test_shape_and_placeholders(self):
        if not _BUNDLED_BMG.exists():
            pytest.skip('bundled BMG fixture not available')
        wb = load_workbook(_BUNDLED_BMG, data_only=True, read_only=True)
        try:
            df, meta = parse_bmg_workbook(wb)
        finally:
            wb.close()

        # 8 plate rows × 12 plate columns = 96 data rows
        assert set(df['replica'].unique()) == set(range(8))
        assert len(df[df['replica'] == 0]) == 12
        assert df['concentration'].min() == 1.0
        assert df['concentration'].max() == 12.0
        # Sentinel attrs so the GUI prompt fires
        assert df.attrs.get('bmg_placeholder_concentrations') is True
        assert isinstance(meta, dict)
        # Test name should be captured from the Microplate sheet metadata
        joined = ' '.join(str(v) for v in meta.values()).lower()
        assert 'zeolite' in joined or 'test_name' in {k.lower() for k in meta}
        # Golden values — plate row A, columns 1–3: 208319, 163967, 127277
        rep0 = df[df['replica'] == 0].sort_values('concentration')
        assert rep0['signal'].iloc[0] == pytest.approx(208319.0)
        assert rep0['signal'].iloc[1] == pytest.approx(163967.0)
        assert rep0['signal'].iloc[2] == pytest.approx(127277.0)


class TestXlsxDispatcher:
    def test_structured_xlsx_still_works(self, tmp_path):
        """XlsxReader must fall back to the structured path for non-BMG files."""
        path = tmp_path / 'structured.xlsx'
        _write_structured_xlsx(path)
        df = load_measurements(path)
        assert set(df['replica'].unique()) == {0, 1}
        assert df.attrs.get('bmg_placeholder_concentrations') is not True


class TestMeasurementWriters:
    def _build_ms(self) -> MeasurementSet:
        df = pd.DataFrame(
            [
                {'concentration': 0.0, 'signal': 100.0, 'replica': 0},
                {'concentration': 1e-6, 'signal': 200.0, 'replica': 0},
                {'concentration': 2e-6, 'signal': 300.0, 'replica': 0},
                {'concentration': 0.0, 'signal': 102.0, 'replica': 1},
                {'concentration': 1e-6, 'signal': 198.0, 'replica': 1},
                {'concentration': 2e-6, 'signal': 305.0, 'replica': 1},
            ]
        )
        return MeasurementSet.from_dataframe(df)

    def test_txt_round_trip(self, tmp_path):
        ms = self._build_ms()
        path = tmp_path / 'export.txt'
        write_measurements_txt(ms, path)
        reloaded = load_measurements(path)
        assert len(reloaded) == 6
        assert set(reloaded['replica'].unique()) == {0, 1}
        # Concentrations and signals survive the round trip
        rep0 = reloaded[reloaded['replica'] == 0].sort_values('concentration')
        assert rep0['concentration'].tolist() == pytest.approx([0.0, 1e-6, 2e-6])
        assert rep0['signal'].tolist() == pytest.approx([100.0, 200.0, 300.0])

    def test_csv_round_trip(self, tmp_path):
        ms = self._build_ms()
        path = tmp_path / 'export.csv'
        write_measurements_csv(ms, path)
        reloaded = load_measurements(path)
        assert len(reloaded) == 6
        assert set(reloaded['replica'].unique()) == {0, 1}
        rep1 = reloaded[reloaded['replica'] == 1].sort_values('concentration')
        assert rep1['signal'].tolist() == pytest.approx([102.0, 198.0, 305.0])


class TestBMGRoundTripAfterConcentrationFix:
    """Load BMG, replace placeholder concentrations, export, reload."""

    def test_full_round_trip(self, tmp_path):
        if not _BUNDLED_BMG.exists():
            pytest.skip('bundled BMG fixture not available')
        df = load_measurements(_BUNDLED_BMG)
        ms = MeasurementSet.from_dataframe(
            df,
            source_file=str(_BUNDLED_BMG),
            bmg_placeholder_concentrations=True,
        )
        assert ms.metadata.get('bmg_placeholder_concentrations') is True

        # Rebuild with real concentrations (1 nM–12 nM as an example)
        real_conc = np.array([i * 1e-9 for i in range(1, ms.n_points + 1)])
        rows = []
        for rep_idx, (_rid, signal) in enumerate(ms.iter_replicas(active_only=False)):
            for c, s in zip(real_conc, signal):
                rows.append({'concentration': c, 'signal': float(s), 'replica': rep_idx})
        fixed = MeasurementSet.from_dataframe(pd.DataFrame(rows))

        path = tmp_path / 'bmg_roundtrip.txt'
        write_measurements_txt(fixed, path)
        reloaded_df = load_measurements(path)
        reloaded = MeasurementSet.from_dataframe(reloaded_df)

        assert reloaded.n_replicas == fixed.n_replicas
        assert reloaded.n_points == fixed.n_points
        np.testing.assert_allclose(reloaded.concentrations, fixed.concentrations)
        np.testing.assert_allclose(reloaded.signals, fixed.signals)
