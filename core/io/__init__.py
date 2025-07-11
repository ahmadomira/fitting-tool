"""
I/O subsystem for the fitting-tool: unified serialization and deserialization of measurement and fit data.

Overview
--------
The `core.io` package provides a flexible, extensible framework for reading and writing
measurement and model-fit data in a variety of formats. It is organized around a few key concepts:

- **Base interfaces**: Abstract base classes define the contract for serializable objects (`Serializable`),
  file readers (`Reader`), and file writers (`Writer`).

- **Domain objects**: Concrete data containers such as `MeasurementSet` (for plate data) and `FitResult`
  (for model fit outputs) implement the `Serializable` interface, providing conversion to/from
  tabular and n-dimensional representations.

- **Registries**: The system uses registries to map (object type, file extension) pairs to the appropriate
  reader or writer implementation. This enables plugin support for new formats without modifying core logic.

- **Serializers**: Reader and writer plugins for specific formats (e.g., Parquet, NetCDF, XLSX) are
  discovered and registered automatically. Each plug-in implements the low-level details for a single format.

Typical usage
-------------
- To save a `MeasurementSet` or `FitResult`, call `core.io.save(obj, path)`.
- To load an object, call `core.io.load(path)`; the correct reader and domain object are selected automatically.

Submodules
----------
- `base`           : Abstract base classes for serialization.
- `fit_result`     : Model fit result container.
- `measurement_set`: Measurement data container.
- `registry`       : Global registries and save/load entry points.
- `serialize`      : Reader/writer plugins for various file formats.
- `serialize_boot` : Bootstraps the serialization system by loading all registered readers/writers.
- `serialize/readers` and `serialize/writers`: Specific file format readers and writers for measurement data and model fit results (e.g., XLSX, NetCDF, Parquet, etc.).

The domain objects (`MeasurementSet`, `FitResult`) define I/O data structure, metadata handling, and data processing methods. They implement the `Serializable` interface, allowing conversion to/from data formats like DataFrame and xarray.Dataset.

See the documentation in each submodule for details and usage examples.
"""

# ensure all plug-ins register
from . import serialize_boot  # noqa: E402  (import after registry)
from .fit_result import FitResult
from .measurement_set import MeasurementSet
from .registry import load, save

_ = serialize_boot  # to prevent unused import removal by linters

__all__ = ["load", "save", "MeasurementSet", "FitResult"]
