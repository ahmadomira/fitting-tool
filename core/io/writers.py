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
