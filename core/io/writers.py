import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .io_base import Writer
from .measurement_sets import MeasurementSet

# ── registry ----------------------------------------------------------------

_writers = {}


def register_writer(ext: str):
    def decorator(cls):
        _writers[ext.lower()] = cls()
        return cls

    return decorator

# ── Parquet writer -----------------------------------------------------------
@register_writer(".parquet")
class ParquetWriter(Writer):
    def write(self, mset: MeasurementSet, path: Path):
        """
        Save as Parquet and embed the meta-dict in the file's Arrow schema.

        Reload snippet:
        >>> import pyarrow.parquet as pq, json
        >>> tbl = pq.read_table("file.parquet")
        >>> meta = json.loads(tbl.schema.metadata[b"attrs"])
        >>> df = tbl.to_pandas()
        >>> mset = MeasurementSet.from_dataframe(df, meta)
        """
        df = mset.tidy
        table = pa.Table.from_pandas(df, preserve_index=False)

        # add JSON-encoded attrs to schema metadata
        meta_json = json.dumps(mset.meta).encode()
        schema_with_meta = table.schema.with_metadata(
            {**(table.schema.metadata or {}), b"attrs": meta_json}
        )
        pq.write_table(table.cast(schema_with_meta), Path(path))

# ── NetCDF writer ---------------------------------------------------------
@register_writer(".nc")
class NetCDFWriter(Writer):
    """
    Loss-less export (dims, coords, attrs, dtypes) of a MeasurementSet using xarray's NetCDF backend.
    The Dataset already carries attrs (= meta), so nothing extra to embed.
    """
    def write(self, mset: MeasurementSet, path: Path):
        # Optional but nicer on disk: gzip compression for the big data var
        encoding = {"signal": {"zlib": True, "complevel": 4}}

        # Try h5netcdf first (pure-Python, smooth on Windows), fall back to default
        try:
            mset.data.to_netcdf(path, engine="h5netcdf", encoding=encoding)
        except ModuleNotFoundError:
            mset.data.to_netcdf(path, encoding=encoding)