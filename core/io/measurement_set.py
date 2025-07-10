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
      - self.ds : xarray.Dataset, dims: well_row, well_col, channel, time_s
      - self.meta : dict, any (meta-)data that is *not* necessarily numeric
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
        return self.ds

    @property
    def tidy(self) -> pd.DataFrame:
        """Return a long DataFrame (one row per point)."""
        return self.ds.to_dataframe().reset_index()

    # --- Serializable interface -------------------------------------------
    def tag(self) -> str:
        return "mset"

    def to_dataframe(self) -> pd.DataFrame:
        return self.tidy

    def to_dataset(self) -> xr.Dataset:
        return self.ds

    # ---------- factory helpers -------------------------------------------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, meta: dict):
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

        df = df.copy()
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")

        da = (
            df.set_index(["well_row", "well_col", "channel", "time_s"])
            .sort_index()["signal"]
            .to_xarray()
            .rename("signal")
        )
        meta = {**meta, "object_type": "mset"}
        ds = xr.Dataset({"signal": da})
        ds.attrs.update(meta)
        return cls(ds, meta)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        meta = dict(ds.attrs)
        return cls(ds, meta)

    @classmethod
    def from_serialisable(cls, payload):
        if isinstance(payload, pd.DataFrame):
            return cls.from_dataframe(payload, json.loads(payload.attrs["attrs"]))
        elif isinstance(payload, xr.Dataset):
            return cls.from_dataset(payload)
        else:
            raise TypeError("Unsupported payload")

    # ----------------------------------------------------------------------
    #   Concentration handling
    # ----------------------------------------------------------------------
    def with_concentration(self, values, name: str = "concentration", unit: str = "µM"):
        """
        Return a *new* MeasurementSet with a concentration coordinate attached.

        Parameters
        ----------
        values : scalar, 1‑D or 2‑D array‑like
            • scalar                       → broadcast to every well
            • length‑12 1‑D array/Series   → one value per column (well_col)
            • 8×12 2‑D array/DataFrame     → one value per well
            • xarray.DataArray            → must be broadcast‑compatible
        name : str, optional
            Coordinate name to use (default: "concentration").
            Useful when multiple titrations exist.
        unit : str, optional
            Saved in ``da.attrs["unit"]`` for reference.

        Returns
        -------
        MeasurementSet
            A *new* instance containing the extra coordinate.
        """

        # ---- normalise input to xarray.DataArray --------------------------------
        if isinstance(values, xr.DataArray):
            da = values
        else:
            arr = np.asarray(values)

            if arr.ndim == 0:
                # scalar: broadcast later
                da = xr.DataArray(arr)
            elif arr.ndim == 1:
                if arr.size != 12:
                    raise ValueError(
                        "1‑D concentration vector must have length 12 (one per column)."
                    )
                da = xr.DataArray(
                    arr,
                    dims=("well_col",),
                    coords={"well_col": self.ds.coords["well_col"]},
                )
            elif arr.ndim == 2:
                if arr.shape != (8, 12):
                    raise ValueError("2‑D concentration matrix must be shape (8, 12).")
                da = xr.DataArray(
                    arr,
                    dims=("well_row", "well_col"),
                    coords={
                        "well_row": self.ds.coords["well_row"],
                        "well_col": self.ds.coords["well_col"],
                    },
                )
            else:
                raise ValueError("`values` must be scalar, 1‑D or 2‑D.")
        da.attrs["unit"] = unit

        # ---- attach as coordinate (auto‑broadcast) ------------------------------
        new_ds = self.ds.assign_coords({name: da})
        return MeasurementSet(new_ds, self.meta)

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

    #   In‑place mutator for concentration
    def add_concentration(self, values, name: str = "concentration", unit: str = "µM"):
        """
        Mutating wrapper around :py:meth:`with_concentration`.

        Alters *this* MeasurementSet by attaching the concentration coordinate
        and returns ``self`` so you can chain calls.

        Examples
        --------
        >>> mset.add_concentration([0, 1, 2, 4, 8, 16, 32, 50, 65, 80, 90, 100])
        >>> mset.concentration("A3")
        2.0
        """
        # delegate to functional version, then replace the internal Dataset
        self.ds = self.with_concentration(values, name, unit).ds
        return self

    # ----------------------------------------------------------------------
    #   Accessors for concentration
    # ----------------------------------------------------------------------
    def concentration(self, well_label: str, name: str = "concentration"):
        """
        Return the *scalar* concentration for an individual well (e.g. 'B7').

        Works whether the coordinate was supplied as:

        * scalar                    (broadcast to all wells)
        * 1‑D per‑column vector     (dims: well_col)
        * 2‑D 8×12 matrix           (dims: well_row, well_col)

        Parameters
        ----------
        well_label : str
            Row‑column label such as 'A3', 'H12'.
        name : str, optional
            Coordinate name (default "concentration").

        Returns
        -------
        float
        """
        row, col = well_label[0], int(well_label[1:])
        coord = self.ds.coords[name]

        # Try the most specific selection possible given the coordinate's dims
        sel_kwargs = {}
        if "well_row" in coord.dims:
            sel_kwargs["well_row"] = row
        if "well_col" in coord.dims:
            sel_kwargs["well_col"] = col

        return coord.sel(**sel_kwargs).item()

    def row_concentration(self, row_label: str, name: str = "concentration"):
        """
        Return the 1‑D concentration vector (length‑12) for an entire row.

        This is typically the X‑axis for a replica series.

        Parameters
        ----------
        row_label : str
            'A' … 'H'
        name : str, optional
            Coordinate name (default "concentration").

        Returns
        -------
        xarray.DataArray
            dims: well_col
        """
        coord = self.ds.coords[name]

        if "well_row" in coord.dims:
            return coord.sel(well_row=row_label).sortby("well_col")
        else:
            # coord is 1‑D → identical for every row
            return coord.sortby("well_col")
