"""
Base interfaces for serialization in the fitting-tool core.io module.

Defines abstract base classes for objects that can be saved/loaded, and for file readers/writers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import xarray as xr

Tag = str  # e.g. "mset", "fit"
Payload = Any  # pd.DataFrame | xr.Dataset | …


class Serializable(ABC):
    """
    Common contract for anything that can be saved or loaded.

    Methods
    -------
    tag() : str
        Return a short string identifier, e.g. 'mset', 'fit'.
    to_dataframe() : pandas.DataFrame
        Return a long DataFrame (one row per point).
    to_dataset() : xarray.Dataset
        Return a multi-dimensional Dataset.
    """

    # --- identity ---------------------------------------------------------
    @abstractmethod
    def tag(self) -> str:
        """
        Return a short string identifier for the object.

        Returns
        -------
        str
            Identifier string, e.g. 'mset', 'fit'.
        """
        pass

    # --- tabular and n-D views -------------------------------------------
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a long-format DataFrame representation of the object.

        Returns
        -------
        pandas.DataFrame
            Long-format table, one row per data point.
        """
        pass

    @abstractmethod
    def to_dataset(self) -> xr.Dataset:
        """
        Return a multi-dimensional xarray.Dataset representation of the object.

        Returns
        -------
        xarray.Dataset
            Multi-dimensional dataset.
        """
        pass


class Reader(ABC):
    """
    Low-level file parser that returns unwrapped data and tag.

    Methods
    -------
    read_raw(path: Path) -> Tuple[str, Any]
        Parse the file at the given path and return (tag, payload).
    """

    @abstractmethod
    def read(self, path: Path) -> Tuple[Tag, Payload]:
        """
        Parse *path* and return (tag, payload).

        Parameters
        ----------
        path : pathlib.Path
            Path to the file to parse.

        Returns
        -------
        tag : str
            Object type tag, e.g. 'mset', 'fit', etc.
        payload : Any
            Tidy DataFrame or xarray.Dataset carrying the metadata.
        """
        ...


class Writer(ABC):
    """
    Serializes a single object type to a single file format.

    Methods
    -------
    write(obj: Serializable, path: Path) -> None
        Persist the object to the given path.
    """

    @abstractmethod
    def write(self, obj: Serializable, path: Path) -> None:
        """
        Persist *obj* (MeasurementSet, FitResult, …) to *path*.

        Parameters
        ----------
        obj : Serializable
            The object to serialize.
        path : pathlib.Path
            The file path to write to.
        """
        pass
