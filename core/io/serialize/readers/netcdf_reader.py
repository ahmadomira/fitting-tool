"""
NetCDF file reader for MeasurementSet and FitResult objects.

This module provides a Reader implementation for .nc (NetCDF) files, returning
an xarray.Dataset and associated metadata.

Classes
-------
NetcdfReader : Reader
    Reads .nc files and returns (tag, xarray.Dataset).
"""

from ...base import Reader
from ...registry import register_reader


@register_reader(".nc")
class NetcdfReader(Reader):
    """
    Reader for .nc (NetCDF) files.

    Returns an xarray.Dataset and metadata from a .nc file.
    """

    def read(self, path):
        """
        Parse a .nc (NetCDF) file and return its tag and data as an xarray.Dataset.

        Parameters
        ----------
        path : pathlib.Path or str
            Path to the .nc file.

        Returns
        -------
        tag : str
            Object type tag, e.g. 'mset', 'fit'.
        ds : xarray.Dataset
            Dataset with data and metadata in attrs.
        """
        import xarray as xr

        ds = xr.open_dataset(path)
        ds.load()
        tag = ds.attrs.get("object_type", "mset")
        return tag, ds
