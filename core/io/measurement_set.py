"""
Canonical representation and interface for measurement data in the fitting-tool.

This module defines the MeasurementSet dataclass, which stores plate or plate-series data
as an xarray.Dataset and associated metadata. Provides methods for conversion to DataFrame
and xarray.Dataset, and for constructing from stored formats.

Classes
-------
MeasurementSet : Serializable
    Stores measurement data and provides serialization helpers.

Examples
--------
>>> ms = MeasurementSet(ds, meta)
>>> df = ms.to_dataframe()
>>> ds2 = ms.to_dataset()
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .base import Serializable

# channel: only one channel in our current use case (FI), can be extended to, e.g., multiple wavelengths
# time_s: for kinetic assays
REQUIRED_COLS = {"well_row", "well_col", "channel", "time_s", "signal"}


@dataclass
class MeasurementSet(Serializable):
    """
    Canonical representation of one plate (or plate-series) worth of data.

    See module docstring for usage examples.
    """

    ds: xr.Dataset
    meta: dict

    # data structure version
    SCHEMA_VERSION: int = 1

    def __post_init__(self):
        # Ensure meta has object_type key
        if "object_type" not in self.meta:
            self.meta["object_type"] = "mset"

        if "schema_version" not in self.meta:
            self.meta["schema_version"] = self.SCHEMA_VERSION

        self.ds.attrs.update(self.meta)

    # handy aliases
    @property
    def data(self) -> xr.Dataset:
        """
        Return the underlying xarray.Dataset.

        Returns
        -------
        xarray.Dataset
            The measurement data as a dataset.
        """
        return self.ds

    @property
    def tidy(self) -> pd.DataFrame:
        """
        Return a long-format DataFrame (one row per point).

        Returns
        -------
        pandas.DataFrame
            Long-format table of measurement data.
        """
        return self.ds.to_dataframe().reset_index()

    # --- Serializable interface -------------------------------------------
    def tag(self) -> str:
        """
        Return the object type tag for this measurement set.

        Returns
        -------
        str
            The string 'mset'.
        """
        return "mset"

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the measurement data as a long-format DataFrame.

        Returns
        -------
        pandas.DataFrame
            Long-format table of measurement data.
        """
        return self.tidy

    def to_dataset(self) -> xr.Dataset:
        """
        Return the measurement data as an xarray.Dataset.

        Returns
        -------
        xarray.Dataset
            The measurement data as a dataset.
        """
        return self.ds

    # ---------- factory helpers -------------------------------------------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, meta: dict):
        """
        Construct a MeasurementSet from a DataFrame and metadata.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing measurement data.
        meta : dict
            Metadata dictionary.

        Returns
        -------
        MeasurementSet
            New MeasurementSet instance.
        """
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

        df = df.copy()
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")

        # pull out optional concentration column
        conc_col = None
        if "concentration" in df.columns:
            conc_col = (
                df[["well_col", "concentration"]]
                .drop_duplicates("well_col")
                .sort_values("well_col")["concentration"]
                .values
            )

        da = (
            df.set_index(["well_row", "well_col", "channel", "time_s"])
            .sort_index()["signal"]
            .to_xarray()
            .rename("signal")
        )
        meta = {**meta, "object_type": "mset"}
        ds = xr.Dataset({"signal": da})
        if conc_col is not None:
            R, C = ds.sizes["well_row"], ds.sizes["well_col"]
            conc_2d = np.broadcast_to(conc_col, (R, C))
            ds = ds.assign_coords(
                concentration=(
                    ("well_row", "well_col"),
                    conc_2d,
                    {"unit": meta.get("concentration_unit", "µM")},
                )
            )
        ds.attrs.update(meta)
        return cls(ds, meta)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        """
        Construct a MeasurementSet from an xarray.Dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing measurement data and metadata.

        Returns
        -------
        MeasurementSet
            New MeasurementSet instance.
        """
        meta = dict(ds.attrs)
        return cls(ds, meta)

    @classmethod
    def from_serialisable(cls, payload):
        """
        Construct a MeasurementSet from a serializable payload (DataFrame or Dataset).

        Parameters
        ----------
        payload : pandas.DataFrame or xarray.Dataset
            Serializable measurement data.

        Returns
        -------
        MeasurementSet
            New MeasurementSet instance.

        Raises
        ------
        TypeError
            If the payload is neither a DataFrame nor a Dataset.
        """
        if isinstance(payload, pd.DataFrame):
            return cls.from_dataframe(payload, json.loads(payload.attrs["attrs"]))
        elif isinstance(payload, xr.Dataset):
            return cls.from_dataset(payload)
        else:
            raise TypeError("Unsupported payload")

    # ----------------------------------------------------------------------
    #   Concentration handling
    # ----------------------------------------------------------------------
    def set_concentration(self, values, unit: str = "µM"):
        """
        Attach or overwrite the 'concentration' coordinate.

        Parameters
        ----------
        values : scalar, 1‑D, 2‑D array‑like or xarray.DataArray
            Scalar          – broadcast to all wells  
            1‑D length C    – one value per column  
            2‑D R×C         – one value per well  
            DataArray       – must be broadcast‑compatible
        unit : str
            Stored in ``coord.attrs["unit"]`` (default 'µM').

        Returns
        -------
        MeasurementSet
            ``self`` (fluent interface)
        """
        arr = values if isinstance(values, np.ndarray) else np.asarray(values)
        R = self.ds.sizes["well_row"]
        C = self.ds.sizes["well_col"]

        # normalise to 2‑D array R×C
        if arr.ndim == 0:                       # scalar
            arr = np.broadcast_to(arr, (R, C))
        elif arr.ndim == 1:                     # per‑column
            if arr.size != C:
                raise ValueError("1‑D concentration vector length ≠ number of columns")
            arr = np.broadcast_to(arr, (R, C))
        elif arr.ndim == 2:
            if arr.shape != (R, C):
                raise ValueError("2‑D concentration matrix shape mismatch")
        else:
            raise ValueError("values must be 0‑, 1‑, or 2‑D")

        da = xr.DataArray(
            arr,
            dims=("well_row", "well_col"),
            coords={
                "well_row": self.ds.coords["well_row"],
                "well_col": self.ds.coords["well_col"],
            },
            attrs={"unit": unit},
        )
        self.ds = self.ds.assign_coords(concentration=da)
        return self

    def concentration(self, well_row: str | None = None, well_col: int | None = None):
        """
        Access the concentration coordinate.

        Returns
        -------
        xarray.DataArray or float
            Full 2‑D array, 1‑D vector, or scalar depending on arguments.
        """
        if "concentration" not in self.ds.coords:
            raise ValueError("No concentration coordinate set.")
        da = self.ds.coords["concentration"]
        if well_row is not None:
            da = da.sel(well_row=well_row)
        if well_col is not None:
            da = da.sel(well_col=well_col)
        return da

    def has_concentration(self, complete: bool = True) -> bool:
        """
        Check if a concentration coordinate exists (and is complete).

        Parameters
        ----------
        complete : bool, optional
            If True, also require that the coordinate contains no NaN.

        Returns
        -------
        bool
        """
        if "concentration" not in self.ds.coords:
            return False
        if complete:
            return not np.isnan(self.ds.coords["concentration"].values).any()
        return True

    # ---- convenience ----------------------------------------------------
    def row(self, label: str, channel="FI", time_s=0):
        """1-D DataArray for a whole row (12 wells)."""
        return (
            self.ds["signal"]
            .sel(well_row=label, channel=channel, time_s=time_s)
            .sortby("well_col")
        )

    def well(self, label: str, channel="FI", time_s=0):
        """Scalar signal of a single well, e.g. label='A3'."""
        row, col = label[0], int(label[1:])
        return (
            self.ds["signal"]
            .sel(well_row=row, well_col=col, channel=channel, time_s=time_s)
            .item()
        )

    @property
    def plate(self):
        """2-D DataArray (8 × 12) for the default channel/time."""
        return (
            self.ds["signal"]
            .sel(channel="FI", time_s=0)
            .sortby(["well_row", "well_col"])
        )
