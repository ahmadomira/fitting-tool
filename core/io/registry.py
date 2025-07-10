"""
Global registries for serialisation plug-ins.

   (tag, ext) → writer      e.g. ('mset', '.parquet')
        ext   → reader      e.g. '.parquet'
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple

from .base import Reader, Serializable, Writer

WriterKey = Tuple[str, str]  # (tag, '.ext')
ReaderKey = str  # '.ext'

_writers: Dict[WriterKey, Writer] = {}
_readers: Dict[ReaderKey, Reader] = {}


def register_writer(tag: str, ext: str):
    ext = ext.lower()

    def decorator(cls):
        _writers[(tag, ext)] = cls()
        return cls

    return decorator


def register_reader(ext: str):
    ext = ext.lower()

    def decorator(cls):
        _readers[ext] = cls()
        return cls

    return decorator


def save(obj: Serializable, path: Path):
    tag = obj.tag()
    ext = path.suffix.lower()
    try:
        _writers[(tag, ext)].write(obj, path)
    except KeyError as err:
        raise ValueError(f"No writer for object '{tag}' to *{ext} files") from err


def load(path: Path):
    ext = Path(path).suffix.lower()
    try:
        tag, payload = _readers[ext].read_raw(path)  # returns (tag, df|ds|...)
    except KeyError as err:
        raise ValueError(f"No reader for *{ext} files") from err

    from .fit_result import FitResult
    from .measurement_set import MeasurementSet

    factory = {
        "mset": MeasurementSet.from_serialisable,
        "fit": FitResult.from_serialisable,
    }

    obj = factory[tag](
        payload
    )  # data object (MeasurementSet or FitResult, pd.DataFrame or xr.Dataset)

    # For signaling future I/O structure changes
    # e.g. if we rename "signal" to "fluorescence" in v2, old files can be detected and migrated automatically.
    # if we later overhaul the layout, we bump SCHEMA_VERSION, adjust the loader accordingly, and we’ll know exactly which files need migration.
    EXPECTED_VERSION = {
        "mset": MeasurementSet.SCHEMA_VERSION,
        "fit": FitResult.SCHEMA_VERSION,
    }

    if obj.meta.get("schema_version") != EXPECTED_VERSION[tag]:
        warnings.warn(
            f"{tag} schema v{obj.meta.get('schema_version')} "
            f"!= expected v{EXPECTED_VERSION[tag]}",
            UserWarning,
        )

    return obj
