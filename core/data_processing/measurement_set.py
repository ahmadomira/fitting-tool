"""In-memory container for multi-replica measurement data.

MeasurementSet holds a shared concentration grid and all replica signals
in a 2D numpy array.  An internal boolean mask tracks which replicas are
"active" (included for fitting / averaging) without ever mutating the raw
data.

Typical construction::

    from core.io import load_measurements
    from core.data_processing import MeasurementSet

    df = load_measurements("data/GDA_system.txt")
    ms = MeasurementSet.from_dataframe(df, assay_type="GDA")
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Tuple, Type

import numpy as np
import pandas as pd

from core.assays.base import BaseAssay
from core.units import Q_, Quantity

logger = logging.getLogger(__name__)


@dataclass
class MeasurementSet:
    """Multi-replica measurement data with non-destructive replica masking.

    Attributes
    ----------
    concentrations : np.ndarray
        Shared concentration grid, shape ``(n_points,)``.  Read-only after
        construction.
    signals : np.ndarray
        Replica signals, shape ``(n_replicas, n_points)``.  Read-only after
        construction.
    replica_ids : tuple[str, ...]
        Immutable identifiers for each replica row.
    metadata : dict[str, Any]
        Free-form provenance bag (``source_file``, ``assay_type``, ``units``,
        ``date``, etc.).
    id : str
        Auto-generated UUID (hex).
    processing_log : list[dict[str, Any]]
        Chronological log of preprocessing steps applied.
    """

    concentrations: np.ndarray
    signals: np.ndarray
    replica_ids: tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    processing_log: list[Dict[str, Any]] = field(default_factory=list)

    # Internal active mask — not part of public __init__
    _active_mask: np.ndarray = field(init=False, repr=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate shapes, freeze raw data, initialise active mask."""
        self.concentrations = np.array(self.concentrations, dtype=np.float64)
        self.signals = np.array(self.signals, dtype=np.float64)

        if self.concentrations.ndim != 1:
            raise ValueError(f'concentrations must be 1-D, got shape {self.concentrations.shape}')
        if self.signals.ndim != 2:
            raise ValueError(f'signals must be 2-D (n_replicas, n_points), got shape {self.signals.shape}')
        if self.signals.shape[1] != self.concentrations.shape[0]:
            raise ValueError(
                f'signals columns ({self.signals.shape[1]}) must match concentrations length ({self.concentrations.shape[0]})'
            )
        if len(self.replica_ids) != self.signals.shape[0]:
            raise ValueError(
                f'replica_ids length ({len(self.replica_ids)}) must match signals rows ({self.signals.shape[0]})'
            )

        # Freeze raw data
        self.concentrations.flags.writeable = False
        self.signals.flags.writeable = False
        self.replica_ids = tuple(self.replica_ids)

        # All replicas start active
        self._active_mask = np.ones(self.signals.shape[0], dtype=bool)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        replica_col: str = 'replica',
        concentration_col: str = 'concentration',
        signal_col: str = 'signal',
        **metadata: Any,
    ) -> 'MeasurementSet':
        """Build a MeasurementSet from a long-form DataFrame.

        The DataFrame must contain columns for concentration, signal, and
        replica index.  All replicas must share the same concentration grid.

        Parameters
        ----------
        df : pd.DataFrame
            Long-form data with one row per (replica, concentration) pair.
        replica_col, concentration_col, signal_col : str
            Column names (defaults match ``load_measurements`` output).
        **metadata
            Arbitrary keyword arguments stored in ``metadata``.

        Returns
        -------
        MeasurementSet

        Raises
        ------
        ValueError
            If required columns are missing or replicas have mismatched grids.
        """
        for col in (replica_col, concentration_col, signal_col):
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")

        # Capture a reader-declared concentration unit before any reshaping that
        # could drop ``df.attrs`` (readers tag non-M files, e.g. an external µM
        # CSV, via ``df.attrs['concentration_unit']``; absent means M).
        declared_conc_unit = df.attrs.get('concentration_unit')

        # Sort for deterministic order
        df = df.sort_values([replica_col, concentration_col]).reset_index(drop=True)

        replica_labels = df[replica_col].unique()
        groups = {label: grp.sort_values(concentration_col) for label, grp in df.groupby(replica_col)}

        # Extract and validate shared concentration grid
        reference_conc = groups[replica_labels[0]][concentration_col].values
        for label, grp in groups.items():
            conc = grp[concentration_col].values
            # Check length first: np.allclose on unequal-length arrays raises a
            # cryptic "operands could not be broadcast together" error.
            if len(conc) != len(reference_conc):
                raise ValueError(
                    f"Replica '{label}' has {len(conc)} concentration points but replica "
                    f"'{replica_labels[0]}' has {len(reference_conc)} — all replicas must share "
                    'the same concentration grid.'
                )
            if not np.allclose(conc, reference_conc, rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"Replica '{label}' has a different concentration grid than replica '{replica_labels[0]}'"
                )

        # Build 2-D signals array
        signals = np.array([groups[label][signal_col].values for label in replica_labels])
        replica_ids = tuple(str(label) for label in replica_labels)

        # Convert to molar if a non-M unit was declared, so the stored grid is
        # always M. Pint validates the token and the dimensionality.
        if declared_conc_unit and declared_conc_unit != 'M':
            reference_conc = Q_(np.asarray(reference_conc, dtype=float), declared_conc_unit).to('M').magnitude

        return cls(
            concentrations=reference_conc,
            signals=signals,
            replica_ids=replica_ids,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_replicas(self) -> int:
        """Total number of replicas (active + dropped)."""
        return self.signals.shape[0]

    @property
    def n_active(self) -> int:
        """Number of currently active replicas."""
        return int(self._active_mask.sum())

    @property
    def n_points(self) -> int:
        """Number of concentration points per replica."""
        return self.concentrations.shape[0]

    @property
    def active_replica_ids(self) -> list[str]:
        """Replica IDs currently marked active."""
        return [rid for rid, active in zip(self.replica_ids, self._active_mask) if active]

    @property
    def dropped_replica_ids(self) -> list[str]:
        """Replica IDs currently marked inactive."""
        return [rid for rid, active in zip(self.replica_ids, self._active_mask) if not active]

    # ------------------------------------------------------------------
    # Replica management
    # ------------------------------------------------------------------

    def _replica_index(self, replica_id: str) -> int:
        """Return the row index for *replica_id*, or raise ValueError."""
        try:
            return self.replica_ids.index(replica_id)
        except ValueError:
            raise ValueError(f"Unknown replica_id '{replica_id}'. Available: {self.replica_ids}") from None

    def set_active(self, replica_id: str, active: bool) -> None:
        """Set the active state of a single replica.

        Parameters
        ----------
        replica_id : str
            Must be one of ``replica_ids``.
        active : bool
            ``True`` to include, ``False`` to exclude.
        """
        idx = self._replica_index(replica_id)
        self._active_mask[idx] = active

    def is_active(self, replica_id: str) -> bool:
        """Check whether *replica_id* is currently active."""
        return bool(self._active_mask[self._replica_index(replica_id)])

    def reset_active(self) -> None:
        """Mark all replicas active (undo all filtering)."""
        self._active_mask[:] = True

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def iter_replicas(self, *, active_only: bool = True) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate over replicas as ``(replica_id, signal_view)`` pairs.

        Parameters
        ----------
        active_only : bool
            If ``True`` (default), skip inactive replicas.

        Yields
        ------
        tuple[str, np.ndarray]
            ``(replica_id, signal)`` where *signal* is a read-only **view**
            into the underlying signals array (no copy).
        """
        for i, rid in enumerate(self.replica_ids):
            if active_only and not self._active_mask[i]:
                continue
            yield rid, self.signals[i]

    def get_replica_signal(self, replica_id: str) -> np.ndarray:
        """Return the signal array (read-only view) for a single replica.

        Parameters
        ----------
        replica_id : str
            Must be one of ``replica_ids``.

        Returns
        -------
        np.ndarray
            Shape ``(n_points,)``, read-only view.
        """
        return self.signals[self._replica_index(replica_id)]

    def average_signal(self, *, active_only: bool = True) -> np.ndarray:
        """Compute the mean signal across replicas.

        Parameters
        ----------
        active_only : bool
            If ``True`` (default), average only over active replicas.

        Returns
        -------
        np.ndarray
            Shape ``(n_points,)``.

        Raises
        ------
        ValueError
            If no replicas are selected.
        """
        mask = self._active_mask if active_only else np.ones(self.n_replicas, dtype=bool)
        if not mask.any():
            raise ValueError('No replicas selected for averaging')
        return self.signals[mask].mean(axis=0)

    # ------------------------------------------------------------------
    # Bridge to assay objects
    # ------------------------------------------------------------------

    def to_assay(
        self,
        assay_cls: Type[BaseAssay],
        *,
        conditions: Dict[str, Any],
        use_average: bool = True,
        replica_id: Optional[str] = None,
    ) -> BaseAssay:
        """Construct a :class:`BaseAssay` from this measurement set.

        Parameters
        ----------
        assay_cls : Type[BaseAssay]
            Concrete assay class (e.g. ``GDAAssay``).
        conditions : dict
            Assay-specific keyword arguments (``Ka_dye``, ``h0``, etc.).
        use_average : bool
            If ``True``, use the averaged active-replica signal.
        replica_id : str, optional
            Use this specific replica instead of the average.  Raises if the
            replica is inactive.

        Returns
        -------
        BaseAssay
            Ready for fitting via ``fit_assay()``.

        Raises
        ------
        ValueError
            If *replica_id* is specified but inactive, or if both
            *use_average* is False and *replica_id* is None.
        """
        if replica_id is not None:
            if not self.is_active(replica_id):
                raise ValueError(f"Replica '{replica_id}' is inactive.  Activate it first or choose an active replica.")
            y = np.array(self.get_replica_signal(replica_id))  # writeable copy
        elif use_average:
            y = self.average_signal(active_only=True)
        else:
            raise ValueError('Either set use_average=True or provide a replica_id')

        return assay_cls(
            x_data=Q_(np.array(self.concentrations), 'M'),
            y_data=Q_(y, 'au'),
            name=self.metadata.get('source_file', ''),
            **conditions,
        )

    # ------------------------------------------------------------------
    # In-place mutation
    # ------------------------------------------------------------------

    def set_concentrations(
        self,
        new_conc: Quantity,
        *,
        drop_metadata_keys: Tuple[str, ...] = (),
    ) -> None:
        """Replace this MeasurementSet's concentration grid in place.

        The blessed mutation path for the concentration vector. The
        read-only lock on ``concentrations`` is lifted transiently so the
        existing buffer can be overwritten; the lock is restored before
        returning, so external code still cannot mutate the array.

        Parameters
        ----------
        new_conc : pint.Quantity
            New concentration vector. Must have concentration
            dimensionality and length equal to ``self.n_points``.
            Converted to molar via Pint; the unit declaration done at
            the call site is the single source of truth for the
            conversion.
        drop_metadata_keys : tuple of str, optional
            Metadata keys to drop after the update (e.g. the BMG
            placeholder flag once real concentrations are supplied).
        """
        if not isinstance(new_conc, Quantity):
            raise TypeError(f'new_conc must be a pint Quantity, got {type(new_conc).__name__}')
        if new_conc.dimensionality != Q_(1.0, 'M').dimensionality:
            raise ValueError(f'new_conc must have concentration dimensionality, got {new_conc.dimensionality}')
        molar_values = np.asarray(new_conc.to('M').magnitude, dtype=np.float64)
        if molar_values.shape != (self.n_points,):
            raise ValueError(f'new_conc length {molar_values.shape} does not match n_points ({self.n_points},)')

        self.concentrations.flags.writeable = True
        try:
            self.concentrations[:] = molar_values
        finally:
            self.concentrations.flags.writeable = False

        for key in drop_metadata_keys:
            self.metadata.pop(key, None)
