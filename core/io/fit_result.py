"""
Container and interface for model-fit outputs in the fitting-tool.

This module defines the FitResult dataclass, which stores the results of a model fit,
including fitted parameters, statistics, and metadata. Provides methods for conversion
to DataFrame and xarray.Dataset, and for constructing from stored formats.

Classes
-------
FitResult : Serializable
    Stores model fit results and provides serialization helpers.

Examples
--------
>>> fr = FitResult(params, stats, ref_hash)
>>> df = fr.to_dataframe()
>>> ds = fr.to_dataset()
"""

import json
from dataclasses import dataclass, field

import pandas as pd
import xarray as xr

from .base import Serializable


@dataclass(frozen=True)
class FitResult(Serializable):
    """
    Stores model-fit outputs, including parameters, statistics, and metadata.

    See module docstring for usage examples.
    """

    params: xr.Dataset
    stats: dict
    ref_hash: str
    meta: dict = field(default_factory=dict)

    # data structure version
    SCHEMA_VERSION = 1

    def __post_init__(self):
        full_meta = {
            **self.meta,
            "object_type": "fit",
            "schema_version": self.SCHEMA_VERSION,
        }
        object.__setattr__(self, "meta", full_meta)
        self.params.attrs.update(full_meta)

    def tag(self) -> str:
        """
        Return the object type tag for this result.

        Returns
        -------
        str
            The string 'fit'.
        """
        return "fit"

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a long-format DataFrame with one row per fitted unit.

        Returns
        -------
        pandas.DataFrame
            DataFrame with fitted parameters and statistics.
        """
        df = self.params.to_dataframe().reset_index()
        df["ref_hash"] = self.ref_hash
        df["method"] = self.meta.get("method", "")
        for k, v in self.stats.items():
            df[k] = v
        return df

    def to_dataset(self) -> xr.Dataset:
        """
        Return the fitted parameters as an xarray.Dataset.

        Returns
        -------
        xarray.Dataset
            Dataset of fitted parameters.
        """
        return self.params

    # ---------- constructors from stored formats -------------------------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, meta: dict):
        """
        Construct a FitResult from a DataFrame and metadata.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing fit results.
        meta : dict
            Metadata dictionary.

        Returns
        -------
        FitResult
            New FitResult instance.
        """
        ds = xr.Dataset.from_dataframe(df.set_index(df.columns[:1].tolist()))
        return cls(
            params=ds,
            stats=meta.get("stats", {}),
            ref_hash=meta.get("ref_hash", ""),
            meta=meta,
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        """
        Construct a FitResult from an xarray.Dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing fit results.

        Returns
        -------
        FitResult
            New FitResult instance.
        """
        meta = dict(ds.attrs)
        return cls(
            params=ds,
            stats=meta.get("stats", {}),
            ref_hash=meta.get("ref_hash", ""),
            meta=meta,
        )

    @classmethod
    def from_serialisable(cls, payload):
        """
        Construct a FitResult from a serializable payload (DataFrame or Dataset).

        Parameters
        ----------
        payload : pandas.DataFrame or xarray.Dataset
            Payload containing fit results.

        Returns
        -------
        FitResult
            New FitResult instance.

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
