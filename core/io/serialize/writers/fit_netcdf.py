from pathlib import Path

from ...base import Writer
from ...fit_result import FitResult
from ...registry import register_writer


@register_writer("fit", ".nc")
class FitNetcdfWriter(Writer):
    def write(self, obj: FitResult, path: Path):
        # params already carry meta tag

        # Try h5netcdf first (pure-Python, smooth on Windows), fall back to default
        try:
            obj.to_dataset().to_netcdf(path, engine="h5netcdf")
        except ModuleNotFoundError:
            obj.to_dataset().to_netcdf(path)
