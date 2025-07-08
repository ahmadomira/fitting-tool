from pathlib import Path

from .measurement_sets import MeasurementSet
from .readers import _readers
from .writers import _writers


def load(path: Path) -> MeasurementSet:
    ext = path.suffix.lower()
    try:
        return _readers[ext].read(path)
    except KeyError:
        raise ValueError(f"No reader for *{ext} files")


def save(mset: MeasurementSet, path: Path):
    ext = path.suffix.lower()
    try:
        return _writers[ext].write(mset, path)
    except KeyError:
        raise ValueError(f"No writer for *{ext} files")
