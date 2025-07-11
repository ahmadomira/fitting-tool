from pathlib import Path

from ...base import Writer
from ...measurement_set import MeasurementSet
from ...registry import register_writer


@register_writer("mset", ".nc")
class MsetNetcdfWriter(Writer):
    """
    Loss-less export (dims, coords, attrs, dtypes) of a MeasurementSet using xarray's NetCDF backend.
    The Dataset already carries attrs (= meta), so nothing extra to embed.
    """

    def write(self, obj: MeasurementSet, path: Path):
        # Optional but nicer on disk: gzip compression for the big data var
        encoding = {"signal": {"zlib": True, "complevel": 4}}

        # Try h5netcdf first (pure-Python, smooth on Windows), fall back to default
        try:
            obj.to_dataset().to_netcdf(path, engine="h5netcdf", encoding=encoding)
        except ModuleNotFoundError:
            obj.to_dataset().to_netcdf(path, encoding=encoding)
