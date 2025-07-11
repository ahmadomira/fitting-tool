"""
Writer for FitResult objects to NetCDF files.

This module provides a Writer implementation for saving FitResult objects
as .nc (NetCDF) files, using h5netcdf if available.

Classes
-------
FitNetcdfWriter : Writer
    Serializes FitResult to .nc (NetCDF) format.
"""

from pathlib import Path

from ...base import Writer
from ...fit_result import FitResult
from ...registry import register_writer


@register_writer("fit", ".nc")
class FitNetcdfWriter(Writer):
    """
    Writer for FitResult objects to .nc (NetCDF) files.
    """

    def write(self, obj: FitResult, path: Path):
        """
        Write a FitResult object to a .nc (NetCDF) file.

        Parameters
        ----------
        obj : FitResult
            The FitResult object to serialize.
        path : pathlib.Path or str
            Path to the .nc file.
        """
        # params already carry meta tag

        # Try h5netcdf first (pure-Python, smooth on Windows), fall back to default
        try:
            obj.to_dataset().to_netcdf(path, engine="h5netcdf")
        except ModuleNotFoundError:
            obj.to_dataset().to_netcdf(path)
