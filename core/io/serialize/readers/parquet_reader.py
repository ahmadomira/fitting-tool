from ...base import Reader
from ...registry import register_reader


@register_reader(".parquet")
class ParquetReader(Reader):
    def read_raw(self, path):
        import json

        import pyarrow.parquet as pq

        table = pq.read_table(path)
        meta_json = (table.schema.metadata or {}).get(b"attrs", b"{}")
        meta = json.loads(meta_json)
        tag = meta.get("object_type", "mset")
        
        df = table.to_pandas()
        df.attrs["attrs"] = json.dumps(meta)
        
        return tag, df  # tidy DataFrame
