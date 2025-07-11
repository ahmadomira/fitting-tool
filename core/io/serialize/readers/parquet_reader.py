"""
Parquet file reader for MeasurementSet and FitResult objects.

This module provides a Reader implementation for .parquet files, returning
a tidy pandas DataFrame and associated metadata.

Classes
-------
ParquetReader : Reader
    Reads .parquet files and returns (tag, DataFrame).

Examples
--------
>>> reader = ParquetReader()
>>> tag, df = reader.read_raw(Path('data/example.parquet'))
>>> print(tag)
'mset'
>>> print(df.head())
  well_row  well_col  ...
0        A         1  ...

"""

from ...base import Reader
from ...registry import register_reader


@register_reader(".parquet")
class ParquetReader(Reader):
    """
    Reader for .parquet files.

    Returns a tidy DataFrame and metadata from a .parquet file.
    See module docstring for usage examples.
    """

    def read_raw(self, path):
        """
        Parse a .parquet file and return its tag and data as a DataFrame.

        Parameters
        ----------
        path : pathlib.Path or str
            Path to the .parquet file.

        Returns
        -------
        tag : str
            Object type tag, e.g. 'mset', 'fit'.
        df : pandas.DataFrame
            Tidy DataFrame with data and metadata in attrs['attrs'].
        """

        import json

        import pyarrow.parquet as pq

        table = pq.read_table(path)
        meta_json = (table.schema.metadata or {}).get(b"attrs", b"{}")
        meta = json.loads(meta_json)
        tag = meta.get("object_type", "mset")

        df = table.to_pandas()
        df.attrs["attrs"] = json.dumps(meta)

        return tag, df  # tidy DataFrame
