"""P4: I/O round-trip tests.

Verify that measurement data survives write→read cycles and that
the reader/writer handle edge cases correctly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_processing.measurement_set import MeasurementSet
from core.io import load_measurements, save_results
from core.io.formats.csv_reader import CsvReader
from core.io.formats.measurement_writer import write_measurements_txt
from core.io.formats.txt import TxtReader


class TestSelfDescribingUnits:
    """A declared concentration unit survives from file to a molar MeasurementSet."""

    def test_txt_units_header_converts_uM_to_M(self, tmp_path):
        """A '# units: concentration=µM' header makes the reader tag the frame so
        MeasurementSet.from_dataframe converts the grid to molar."""
        data = '# units: concentration=µM, signal=au\nvar\tsignal\n0.1\t100\n0.5\t200\n1.0\t300\n'
        p = tmp_path / 'uM.txt'
        p.write_text(data)

        df = load_measurements(p)
        assert df.attrs.get('concentration_unit') == 'µM'

        ms = MeasurementSet.from_dataframe(df)
        np.testing.assert_allclose(ms.concentrations, [1e-7, 5e-7, 1e-6], rtol=1e-12)

    def test_measurement_writer_txt_round_trips_in_M(self, tmp_path):
        """A molar MeasurementSet written to TXT reads back unchanged (M)."""
        ms = MeasurementSet(
            concentrations=np.array([1e-7, 5e-7, 1e-6]),
            signals=np.array([[100.0, 200.0, 300.0]]),
            replica_ids=('0',),
        )
        out = tmp_path / 'out.txt'
        write_measurements_txt(ms, out)

        ms2 = MeasurementSet.from_dataframe(load_measurements(out))
        np.testing.assert_allclose(ms2.concentrations, ms.concentrations, rtol=1e-12)


class TestTxtReader:
    """TxtReader correctly parses measurement files."""

    def test_single_replica(self, tmp_path):
        """Single replica file loads correctly."""
        data = 'var\tsignal\n0.0\t100.0\n1e-6\t200.0\n2e-6\t300.0\n'
        p = tmp_path / 'single.txt'
        p.write_text(data)

        reader = TxtReader()
        df = reader.read(p)

        assert len(df) == 3
        assert list(df.columns) == ['concentration', 'signal', 'replica']
        assert (df['replica'] == 0).all()
        assert df['concentration'].iloc[0] == pytest.approx(0.0)
        assert df['signal'].iloc[-1] == pytest.approx(300.0)

    def test_multi_replica(self, tmp_path):
        """Multi-replica file with repeated headers is parsed correctly."""
        data = 'var\tsignal\n0.0\t100.0\n1e-6\t200.0\nvar\tsignal\n0.0\t110.0\n1e-6\t210.0\nvar\tsignal\n0.0\t105.0\n1e-6\t205.0\n'
        p = tmp_path / 'multi.txt'
        p.write_text(data)

        reader = TxtReader()
        df = reader.read(p)

        assert len(df) == 6
        assert set(df['replica'].unique()) == {0, 1, 2}
        # 2 rows per replica
        for r in range(3):
            assert len(df[df['replica'] == r]) == 2

    def test_skips_comment_lines(self, tmp_path):
        """Lines starting with # are ignored."""
        data = '# comment\nvar\tsignal\n0.0\t100.0\n# another comment\n1e-6\t200.0\n'
        p = tmp_path / 'comments.txt'
        p.write_text(data)

        reader = TxtReader()
        df = reader.read(p)

        assert len(df) == 2

    def test_single_data_row(self, tmp_path):
        """A file with exactly one data row loads as a 1-row replica."""
        data = 'var\tsignal\n1e-6\t100\n'
        p = tmp_path / 'one_row.txt'
        p.write_text(data)

        reader = TxtReader()
        df = reader.read(p)

        assert len(df) == 1
        assert df['concentration'].iloc[0] == pytest.approx(1e-6)
        assert df['signal'].iloc[0] == pytest.approx(100.0)
        assert df['replica'].iloc[0] == 0

    def test_empty_file_raises(self, tmp_path):
        """Empty file raises ValueError."""
        p = tmp_path / 'empty.txt'
        p.write_text('')

        reader = TxtReader()
        with pytest.raises(ValueError, match='No data found'):
            reader.read(p)

    def test_loads_real_gda_data(self):
        """Load the actual GDA data file from the data/ directory."""
        gda_path = Path('data/GDA_system.txt')
        if not gda_path.exists():
            pytest.skip('GDA_system.txt not found in data/')

        df = load_measurements(gda_path)
        assert len(df) > 0
        assert 'concentration' in df.columns
        assert 'signal' in df.columns
        assert 'replica' in df.columns

    def test_concentration_header_variant(self, tmp_path):
        """Accepts 'concentration' as header name."""
        data = 'concentration\tsignal\n0.0\t100.0\n1e-6\t200.0\n'
        p = tmp_path / 'conc_header.txt'
        p.write_text(data)

        reader = TxtReader()
        df = reader.read(p)
        assert len(df) == 2


class TestCsvReader:
    """CsvReader handles varied CSV dialects and headers."""

    def test_standard_comma_dot(self, tmp_path):
        """Comma-separated, dot-decimal CSV with canonical headers."""
        data = 'concentration,signal\n0.0,100.0\n1e-6,200.0\n2e-6,300.0\n'
        p = tmp_path / 'std.csv'
        p.write_text(data)

        df = CsvReader().read(p)
        assert len(df) == 3
        assert df['concentration'].iloc[1] == pytest.approx(1e-6)
        assert df['signal'].iloc[-1] == pytest.approx(300.0)
        assert (df['replica'] == 0).all()

    def test_european_semicolon_comma(self, tmp_path):
        """Semicolon-delimited, comma-decimal CSV (Patrick-style)."""
        data = 'conc CB;Int (455 nm) cut 513\n0;29,29\n10,61;234,95\n20,97;416,78\n'
        p = tmp_path / 'euro.csv'
        p.write_text(data)

        df = CsvReader().read(p)
        assert len(df) == 3
        assert df['concentration'].iloc[1] == pytest.approx(10.61)
        assert df['signal'].iloc[2] == pytest.approx(416.78)

    def test_fuzzy_headers_via_name_match(self, tmp_path):
        """'conc CB' and 'Int (...)' are detected via token-startswith."""
        data = 'conc CB,Int (455 nm) cut 513\n0.0,29.29\n1.0,234.95\n2.0,416.78\n'
        p = tmp_path / 'fuzzy.csv'
        p.write_text(data)

        df = CsvReader().read(p)
        assert len(df) == 3
        assert df['signal'].iloc[0] == pytest.approx(29.29)

    def test_headerless_inferred_by_monotonicity(self, tmp_path):
        """CSV without a header row: concentration inferred as monotonic column."""
        data = '0.0,100.0\n1.0,250.0\n2.0,380.0\n3.0,470.0\n4.0,540.0\n'
        p = tmp_path / 'headerless.csv'
        p.write_text(data)

        df = CsvReader().read(p)
        assert len(df) == 5
        assert df['concentration'].iloc[0] == pytest.approx(0.0)
        assert df['signal'].iloc[-1] == pytest.approx(540.0)

    def test_wide_format_replicas(self, tmp_path):
        """Wide format: one conc column + multiple signal columns → replicas."""
        data = 'concentration,rep0,rep1,rep2\n0.0,100.0,105.0,98.0\n1e-6,200.0,210.0,195.0\n'
        p = tmp_path / 'wide.csv'
        p.write_text(data)

        df = CsvReader().read(p)
        assert set(df['replica'].unique()) == {0, 1, 2}
        assert len(df) == 6

    def test_unparseable_raises(self, tmp_path):
        """Non-numeric content raises ValueError."""
        data = 'name,color\nalice,red\nbob,blue\n'
        p = tmp_path / 'bad.csv'
        p.write_text(data)

        with pytest.raises(ValueError, match='Cannot'):
            CsvReader().read(p)

    def test_loads_patrick_file(self):
        """Real-world European CSV from data/patrick_data_origin_exp/ loads."""
        path = Path('data/patrick_data_origin_exp/CB7 azo 1 I 455 nm cut 513.csv')
        if not path.exists():
            pytest.skip(f'{path} not found')

        df = CsvReader().read(path)
        assert len(df) == 15
        assert df['concentration'].is_monotonic_increasing
        assert df['concentration'].iloc[0] == pytest.approx(0.0)
        assert df['concentration'].iloc[-1] == pytest.approx(128.08511)
        assert df['signal'].iloc[-1] == pytest.approx(1291.17225)


class TestTxtWriter:
    """TxtWriter correctly serializes fit results."""

    def test_write_results_via_public_api(self, tmp_path):
        """save_results() public API works end-to-end."""
        results = {'Ka_dye': 5e5, 'Ka_dye_uncertainty': 1e4}
        p = tmp_path / 'api_results.txt'
        save_results(results, p)

        assert p.exists()
        content = p.read_text()
        assert 'Ka_dye' in content


class TestIODataIntegrity:
    """Measurement data survives load with correct types and values."""

    def test_dtypes_are_numeric(self, tmp_path):
        """Loaded data has numeric dtypes, not strings."""
        data = 'var\tsignal\n0.0\t100.0\n1e-6\t200.0\n'
        p = tmp_path / 'types.txt'
        p.write_text(data)

        df = load_measurements(p)
        assert pd.api.types.is_numeric_dtype(df['concentration'])
        assert pd.api.types.is_numeric_dtype(df['signal'])
        assert pd.api.types.is_integer_dtype(df['replica'])
