from abc import ABC, abstractmethod
from pathlib import Path

from .measurement_sets import MeasurementSet


class Reader(ABC):
    @abstractmethod
    def read(self, path: Path) -> MeasurementSet: ...


class Writer(ABC):
    @abstractmethod
    def write(self, mset: MeasurementSet, path: Path) -> None: ...
