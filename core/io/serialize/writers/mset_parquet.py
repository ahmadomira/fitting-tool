from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ...base import Writer
from ...measurement_set import MeasurementSet
from ...registry import register_writer


@register_writer("mset", ".parquet")
class MsetParquetWriter(Writer):
    """
    Save MeasurementSet as Parquet and embed the meta-dict in the file's Arrow schema.

    Reload snippet:
    >>> import pyarrow.parquet as pq, json
    >>> tbl = pq.read_table("file.parquet")
    >>> meta = json.loads(tbl.schema.metadata[b"attrs"])
    >>> df = tbl.to_pandas()
    >>> mset = MeasurementSet.from_dataframe(df, meta)
    """

    def write(self, obj: MeasurementSet, path: Path):
        import json

        df = obj.to_dataframe()
        table = pa.Table.from_pandas(df, preserve_index=False)

        meta_json = json.dumps(obj.meta).encode()
        schema = table.schema.with_metadata({b"attrs": meta_json})
        pq.write_table(table.cast(schema), path)
