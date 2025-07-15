"""
XLSX file reader for ClarioStar microplate exports.

This module provides a Reader implementation for .xlsx files exported from ClarioStar
microplate readers, returning a tidy pandas DataFrame and associated metadata.

Classes
-------
ClarioStarXlsxReader : Reader
    Reads .xlsx files and returns (tag, DataFrame).
"""

from pathlib import Path

from ...base import Reader
from ...registry import register_reader

# TODO: implement an XLSX writer for measurement & fit data and expand this reader to support it
#       (currently only reads ClarioStar plate reader exports)
# ── XLSX reader ------------------------------------------------------------
@register_reader(".xlsx")
class ClarioStarXlsxReader(Reader):
    """
    Reader for BMG Labtech's ClarioStar plate reader .xlsx files, returning a tidy DataFrame and metadata of the well plate.
    """

    def read(self, path: Path):
        """
        Parse a ClarioStar .xlsx file and return its tag and data as a DataFrame.

        Parameters
        ----------
        path : pathlib.Path or str
            Path to the .xlsx file.

        Returns
        -------
        tag : str
            Object type tag, e.g. 'mset'.
        df : pandas.DataFrame
            Tidy DataFrame with data and metadata in attrs['attrs'].
        """
        import json

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

        # embed metadata for the dispatcher
        tidy.attrs["attrs"] = json.dumps({**meta, "object_type": "mset"})
        return "mset", tidy
