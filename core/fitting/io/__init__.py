from pathlib import Path

from .measurement_sets import MeasurementSet
from .readers import _readers


def load(path: Path) -> MeasurementSet:
    ext = path.suffix.lower()
    try:
        return _readers[ext].read(path)
    except KeyError:
        raise ValueError(f"No reader for *{ext} files")
