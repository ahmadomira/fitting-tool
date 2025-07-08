from pathlib import Path

from .io_base import Reader
from .measurement_sets import MeasurementSet

# ── registry ---------------------------------------------------------------
_readers = {}


def register_reader(ext: str):
    def decorator(cls):
        _readers[ext.lower()] = cls()
        return cls

    return decorator


# ── XLSX reader ------------------------------------------------------------
@register_reader(".xlsx")
class ClarioStarXlsxReader(Reader):
    def read(self, path: Path) -> MeasurementSet:
        import pandas as pd

        xls = pd.ExcelFile(path)
        df_raw = xls.parse("Microplate End point", header=None)

        # locate block with the actual numbers (8 rows × 12 cols)
        start_row = df_raw[df_raw.iloc[:, 0] == "A"].index[0]
        block = df_raw.iloc[start_row : start_row + 8, 0:13]

        tidy = (
            block.set_index(0)  # Set the DataFrame index using existing A-H columns
            .rename_axis("well_row")
            .stack()
            .reset_index()
            .rename(columns={"level_1": "well_col", 0: "signal"})
        )

        # ── metadata -------------------------------------------------------
        proto_df = xls.parse("Protocol Information", header=None).dropna(how="all")

        # two possible layouts:
        #   1) Column‑0 = key, Column‑1 = value     (most recent BMG firmware)
        #   2) Column‑0 = "key: value"              (older exports; no second column)
        if proto_df.shape[1] > 1 and proto_df.iloc[:, 1].notna().any():
            # layout 1 ─ separate columns
            meta_df = proto_df.iloc[:, :2].dropna(how="any")
            meta_df.iloc[:, 0] = (
                meta_df.iloc[:, 0]
                .astype(str)
                .str.replace(r":?$", "", regex=True)  # strip trailing colon
                .str.strip()
            )
            meta = (
                meta_df.set_index(meta_df.columns[0])[meta_df.columns[1]]
                .astype(str)
                .str.strip()
                .to_dict()
            )
        else:
            # layout 2 ─ everything in column 0 separated by “:”
            meta_series = proto_df.iloc[:, 0].dropna().astype(str)
            parts = meta_series.str.split(":", n=1, expand=True)
            parts[0] = parts[0].str.strip()
            parts[1] = parts[1].str.strip()
            meta = parts.set_index(0)[1].to_dict()

        tidy["channel"] = (
            "FI"  # only one channel in our current use case, can be extended to, e.g. multiple wavelengths
        )
        tidy["time_s"] = 0.0  # endpoint ⇒ single time point (no kinetic assay)

        return MeasurementSet.from_dataframe(tidy, meta)


@register_reader(".parquet")
class ParquetReader(Reader):
    """
    Load a MeasurementSet from a Parquet file previously written by
    ParquetWriter.  Expects:
        • a tidy table (one row per point)
        • attrs dict JSON-encoded under schema metadata key b"attrs"
    """

    def read(self, path: Path) -> MeasurementSet:
        import json

        import pyarrow.parquet as pq

        table = pq.read_table(path)
        meta_json = (table.schema.metadata or {}).get(b"attrs")
        meta = json.loads(meta_json) if meta_json else {}

        df = table.to_pandas()  # index already flat
        return MeasurementSet.from_dataframe(df, meta)


# ── CSV reader -------------------------------------------------------------
@register_reader(".csv")
class ClarioStarCsvReader(Reader):
    def read(self, path: Path) -> MeasurementSet: ...


# ── TXT reader -------------------------------------------------------------
@register_reader(".txt")
class ClarioStarTxtReader(Reader):
    def read(self, path: Path) -> MeasurementSet: ...
