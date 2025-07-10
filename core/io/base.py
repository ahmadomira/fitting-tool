from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import xarray as xr

Tag = str  # e.g. "mset", "fit"
Payload = Any  # pd.DataFrame | xr.Dataset | …


class Serializable(ABC):
    """Common contract for anything that can be saved / loaded."""

    # --- identity ---------------------------------------------------------
    @abstractmethod
    def tag(self) -> str:
        """Return a short string identifier, e.g. 'mset', 'fit'."""
        pass

    # --- tabular and n-D views -------------------------------------------
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Return a long DataFrame (one row per point)."""
        pass

    @abstractmethod
    def to_dataset(self) -> xr.Dataset:
        """Return a multi-dimensional Dataset."""
        pass


class Reader(ABC):
    """Low-level file parser that returns unwrapped data + tag."""

    @abstractmethod
    def read_raw(self, path: Path) -> Tuple[Tag, Payload]:
        """
        Parse *path* and return (tag, payload).

        tag      : "mset" | "fit" | other future object types
        payload  : tidy DataFrame or xarray.Dataset carrying the metadata
        """
        ...


class Writer(ABC):
    """Serialises a single object type to a single file format."""

    @abstractmethod
    def write(self, obj: Serializable, path: Path) -> None:
        """
        Persist *obj* (MeasurementSet, FitResult, …) to *path*.
        The implementing class is registered for exactly one:
            (obj.tag(), path.suffix)
        """
        pass
