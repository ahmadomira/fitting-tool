"""Base class for assay data containers.

This module defines the abstract base class for all assay types. Each assay
holds experimental data and knows how to compute predicted signals via its
forward model.

The BaseAssay class is a data container, not a fitter. Fitting is handled
by the optimizer module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Tuple

import numpy as np
import pint

from core.assays.registry import AssayMetadata, AssayType, get_metadata
from core.units import Quantity


@dataclass
class BaseAssay(ABC):
    """Abstract base class for assay data containers.

    Subclasses must implement:
    - assay_type: class attribute defining the AssayType
    - forward_model: compute predicted signal from parameters and conditions
    - get_conditions: return experimental conditions needed for forward model

    Attributes
    ----------
    x_data : Quantity
        Independent variable (e.g., titrant concentration in M).
    y_data : Quantity
        Observed signal values (in au).
    name : str
        Optional identifier for this dataset.
    metadata : Dict[str, Any]
        Additional metadata (source file, date, etc.).
    """

    x_data: Quantity
    y_data: Quantity
    name: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Subclasses must define this class attribute
    assay_type: AssayType = field(init=False)

    # Model identifier recorded on fit results; overridden per assay family
    # (e.g. ``'linear'`` for dye-alone, ``'equilibrium_hg2'`` for HG2).
    model_name: ClassVar[str] = 'equilibrium_4param'

    def __post_init__(self):
        """Validate data after initialization."""
        if not isinstance(self.x_data, Quantity):
            raise TypeError(f'x_data must be a pint Quantity, got {type(self.x_data).__name__}')
        if not isinstance(self.y_data, Quantity):
            raise TypeError(f'y_data must be a pint Quantity, got {type(self.y_data).__name__}')

        # Normalize to canonical units so .magnitude is always M / au. Give a
        # domain-clear error (x is a concentration, y is a signal) rather than a
        # bare pint DimensionalityError. The ``au = [signal]`` dimension means a
        # dimensionless or wrong-dimension y_data now fails here instead of being
        # silently accepted as a signal.
        try:
            object.__setattr__(self, 'x_data', self.x_data.to('M'))
        except pint.DimensionalityError as err:
            raise ValueError(
                f"x_data must be a concentration (e.g. M, µM, nM); got units '{self.x_data.units}'."
            ) from err
        try:
            object.__setattr__(self, 'y_data', self.y_data.to('au'))
        except pint.DimensionalityError as err:
            raise ValueError(f"y_data must be a signal in au; got units '{self.y_data.units}'.") from err

        if self.x_data.shape != self.y_data.shape:
            raise ValueError(f'x_data and y_data must have same shape, got {self.x_data.shape} and {self.y_data.shape}')

        if self.x_data.ndim != 1:
            raise ValueError('x_data and y_data must be one-dimensional')
        if not np.all(np.isfinite(self.x_data.magnitude)):
            raise ValueError('x_data concentrations must be finite')
        if np.any(self.x_data.magnitude < 0):
            raise ValueError('x_data concentrations must be nonnegative')
        if not np.all(np.isfinite(self.y_data.magnitude)):
            raise ValueError('y_data signals must be finite')

    @property
    def registry_metadata(self) -> AssayMetadata:
        """Get metadata from the assay registry."""
        return get_metadata(self.assay_type)

    @property
    def parameter_keys(self) -> Tuple[str, ...]:
        """Parameter names for this assay type."""
        return self.registry_metadata.parameter_keys

    @property
    def n_params(self) -> int:
        """Number of parameters to fit."""
        return len(self.parameter_keys)

    @property
    def n_points(self) -> int:
        """Number of data points."""
        return len(self.x_data)

    @abstractmethod
    def forward_model(self, params: np.ndarray, x: np.ndarray | None = None) -> Quantity:
        """Compute predicted signal from parameters.

        Parameters
        ----------
        params : np.ndarray
            Parameter values in the order defined by parameter_keys.
        x : np.ndarray, optional
            Titrant concentrations (M) to evaluate at. Defaults to the assay's
            own ``x_data`` (the measured points); pass a denser grid to render
            a smooth curve without changing the fit.

        Returns
        -------
        Quantity
            Predicted signal as a ``pint.Quantity`` in signal units — an
            implementation MUST return ``Q_(values, 'au')``, not a bare ndarray.
            Because ``au`` carries the ``[signal]`` dimension, :meth:`residuals`
            subtracts this from the ``au`` ``y_data``; returning a bare array
            raises a ``DimensionalityError`` there.
        """
        pass

    @abstractmethod
    def species(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Equilibrium speciation across ``x_data`` from the same solve as the signal.

        Returns the per-species free/complex concentrations underlying the
        forward model — e.g. ``{'H', 'D', 'HD'}`` for direct binding, or
        ``{'H', 'D', 'G', 'HD', 'HG'}`` for competitive assays.  Insertion order
        is the intended plotting/legend order.  Implementations MUST reuse the
        model's own ``*_species`` solve (the one the signal is built from) so the
        speciation and the plotted signal can never disagree.

        Parameters
        ----------
        params : np.ndarray
            Parameter values in ``parameter_keys`` order (as for
            :meth:`forward_model`).

        Returns
        -------
        Dict[str, np.ndarray]
            Species name → concentration array (M magnitudes), each the same
            length as ``x_data``; NaN where the equilibrium solve fails.
        """
        pass

    @abstractmethod
    def get_conditions(self) -> Dict[str, Any]:
        """Return experimental conditions needed for the forward model.

        Returns
        -------
        Dict[str, Any]
            Conditions like fixed concentrations, known binding constants, etc.
        """
        pass

    def residuals(self, params: np.ndarray) -> Quantity:
        """Compute residuals (observed - predicted).

        Parameters
        ----------
        params : np.ndarray
            Parameter values (bare floats from optimizer).

        Returns
        -------
        Quantity
            Residual values in signal units.
        """
        return self.y_data - self.forward_model(params)

    def sum_squared_residuals(self, params: np.ndarray) -> float:
        """Compute sum of squared residuals (SSR) for optimization.

        Parameters
        ----------
        params : np.ndarray
            Parameter values (bare floats from optimizer).

        Returns
        -------
        float
            Numerical sum of squared residuals in canonical au² units.
        """
        resid = self.residuals(params)
        return float(np.sum(resid.magnitude**2))

    def get_default_bounds(self) -> Dict[str, Tuple[Quantity, Quantity]]:
        """Get default parameter bounds as a name-keyed dictionary.

        Returns the ``default_bounds`` dictionary from the assay registry,
        keyed by parameter name.  Values are Quantity tuples.

        Returns
        -------
        Dict[str, Tuple[Quantity, Quantity]]
            ``{param_name: (lower, upper), ...}``
        """
        return dict(self.registry_metadata.default_bounds)

    def params_to_dict(self, params: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to named dictionary.

        Parameters
        ----------
        params : np.ndarray
            Parameter values in standard order.

        Returns
        -------
        Dict[str, float]
            Parameter names mapped to values.
        """
        return dict(zip(self.parameter_keys, params))

    def params_from_dict(self, param_dict: Dict[str, float]) -> np.ndarray:
        """Convert named dictionary to parameter array.

        Parameters
        ----------
        param_dict : Dict[str, float]
            Parameter names mapped to values.

        Returns
        -------
        np.ndarray
            Parameter values in standard order.
        """
        return np.array([param_dict[k] for k in self.parameter_keys])
