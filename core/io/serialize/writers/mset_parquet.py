"""
Writer for MeasurementSet objects to Parquet files.

This module provides a Writer implementation for saving MeasurementSet objects
as .parquet files, embedding metadata in the file's Arrow schema.

Classes
-------
MsetParquetWriter : Writer
    Serializes MeasurementSet to .parquet format.

Examples
--------
>>> writer = MsetParquetWriter()
>>> writer.write(mset, 'file.parquet')
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ...base import Writer
from ...measurement_set import MeasurementSet
from ...registry import register_writer


@register_writer("mset", ".parquet")
class MsetParquetWriter(Writer):
    """
    Writer for MeasurementSet objects to .parquet files.
    """

    def write(self, obj: MeasurementSet, path: Path):
        """
        Write a MeasurementSet object to a .parquet file, embedding metadata.

        Parameters
        ----------
        obj : MeasurementSet
            The MeasurementSet object to serialize.
        path : pathlib.Path or str
            Path to the .parquet file.
        """
        import json

        df = obj.to_dataframe()
        table = pa.Table.from_pandas(df, preserve_index=False)

        meta_json = json.dumps(obj.meta).encode()
        schema = table.schema.with_metadata({b"attrs": meta_json})
        pq.write_table(table.cast(schema), path)
