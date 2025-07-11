"""
I/O serialization plug-ins for the fitting-tool.

This subpackage contains all reader and writer plug-ins for supported file formats.
Each plug-in registers itself with the global registry, enabling automatic discovery
and use by the core I/O system.

Supported formats
-----------------
Readers:
- Parquet (.parquet): ParquetReader
- NetCDF (.nc): NetcdfReader
- XLSX (.xlsx): ClarioStarXlsxReader

Writers:
- Parquet (.parquet):
    - MsetParquetWriter (MeasurementSet)
    - FitParquetWriter (FitResult)
- NetCDF (.nc):
    - FitNetcdfWriter (FitResult)

Usage
-----
Plug-ins are registered automatically on import. To add support for a new format,
implement a Reader or Writer and decorate it with `@register_reader` or `@register_writer`.

See each module for details and usage examples.
"""

from .readers.netcdf_reader import NetcdfReader
from .readers.parquet_reader import ParquetReader
from .readers.xlsx_reader import ClarioStarXlsxReader
from .writers.fit_netcdf import FitNetcdfWriter
from .writers.fit_parquet import FitParquetWriter
from .writers.mset_parquet import MsetParquetWriter

__all__ = [
    "ParquetReader",
    "MsetParquetWriter",
    "FitParquetWriter",
    "NetcdfReader",
    "FitNetcdfWriter",
    "ClarioStarXlsxReader",
]
