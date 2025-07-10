import json
from dataclasses import dataclass, field

import pandas as pd
import xarray as xr

from .base import Serializable




@dataclass(frozen=True)
class FitResult(Serializable):
    """
    Container for model‑fit outputs.

    params : xarray.Dataset      # fitted coefficients per unit (row / well / plate)
    stats  : dict                # goodness‑of‑fit metrics (AIC, R², …)
    ref_hash : str               # SHA‑1 of the originating MeasurementSet
    meta   : dict                # method, options, timestamp, etc.
    """
    
    params: xr.Dataset
    stats: dict
    ref_hash: str
    meta: dict = field(default_factory=dict)
    
    # data structure version
    SCHEMA_VERSION = 1

    # ---------- automatic tag & version injection ----------------
    def __post_init__(self):
        full_meta = {
            **self.meta,
            "object_type": "fit",
            "schema_version": self.SCHEMA_VERSION,
        }
        object.__setattr__(self, "meta", full_meta)
        self.params.attrs.update(full_meta)

    # ---------- Serializable interface -----------------
    def tag(self) -> str:
        return "fit"

    def to_dataframe(self) -> pd.DataFrame:
        """Long table: one row per fitted unit (+ stats replicated)."""
        df = self.params.to_dataframe().reset_index()
        df["ref_hash"] = self.ref_hash
        df["method"] = self.meta.get("method", "")
        for k, v in self.stats.items():  # broadcast global stats
            df[k] = v
        return df

    def to_dataset(self) -> xr.Dataset:
        return self.params

    # ---------- constructors from stored formats -------------------------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, meta: dict):
        ds = xr.Dataset.from_dataframe(df.set_index(df.columns[:1].tolist()))
        return cls(
            params=ds,
            stats=meta.get("stats", {}),
            ref_hash=meta.get("ref_hash", ""),
            meta=meta,
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        meta = dict(ds.attrs)
        return cls(
            params=ds,
            stats=meta.get("stats", {}),
            ref_hash=meta.get("ref_hash", ""),
            meta=meta,
        )

    @classmethod
    def from_serialisable(cls, payload):
        if isinstance(payload, pd.DataFrame):
            return cls.from_dataframe(payload, json.loads(payload.attrs["attrs"]))
        elif isinstance(payload, xr.Dataset):
            return cls.from_dataset(payload)
        else:
            raise TypeError("Unsupported payload")
