"""
Writer for FitResult objects to Parquet files.

This module provides a Writer implementation for saving FitResult objects
as .parquet files, embedding metadata in the file's Arrow schema.

Classes
-------
FitParquetWriter : Writer
    Serializes FitResult to .parquet format.
"""

from pathlib import Path

from ...base import Writer
from ...fit_result import FitResult
from ...registry import register_writer


@register_writer("fit", ".parquet")
class FitParquetWriter(Writer):
    """
    Writer for FitResult objects to .parquet files.
    """

    def write(self, obj: FitResult, path: Path):
        """
        Write a FitResult object to a .parquet file, embedding metadata.

        Parameters
        ----------
        obj : FitResult
            The FitResult object to serialize.
        path : pathlib.Path or str
            Path to the .parquet file.
        """
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        df = obj.to_dataframe()
        table = pa.Table.from_pandas(df, preserve_index=False)

        meta = {**obj.meta, "stats": obj.stats, "ref_hash": obj.ref_hash}
        meta_json = json.dumps(meta).encode()

        schema = table.schema.with_metadata({b"attrs": meta_json})
        pq.write_table(table.cast(schema), path)
