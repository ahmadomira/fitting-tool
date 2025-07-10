from ...base import Reader
from ...registry import register_reader


@register_reader(".nc")
class NetcdfReader(Reader):
    def read_raw(self, path):
        import xarray as xr

        ds = xr.open_dataset(path)
        ds.load()
        tag = ds.attrs.get("object_type", "mset")
        return tag, ds
