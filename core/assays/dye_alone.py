"""Dye-alone calibration assay.

This is a simple linear calibration where signal is proportional to
dye concentration:
    Signal = slope * [Dye] + intercept

Used to establish the linear response range of the dye.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict

import numpy as np

from core.assays.base import BaseAssay
from core.assays.registry import AssayType
from core.models.linear import linear_signal
from core.units import Q_, Quantity


@dataclass
class DyeAloneAssay(BaseAssay):
    """Dye-alone calibration data container.

    Attributes
    ----------
    x_data : np.ndarray
        Dye concentrations (M).
    y_data : np.ndarray
        Observed signal values.
    name : str
        Optional identifier for this dataset.
    metadata : Dict[str, Any]
        Additional metadata.

    Example
    -------
    >>> assay = DyeAloneAssay(
    ...     x_data=dye_conc,  # Dye concentrations in M
    ...     y_data=signal,
    ... )
    >>> slope, intercept, r2, rmse = assay.fit_linear()
    """

    assay_type: AssayType = field(init=False, default=AssayType.DYE_ALONE)
    model_name: ClassVar[str] = 'linear'

    def forward_model(self, params: np.ndarray, x: np.ndarray | None = None) -> Quantity:
        """Compute predicted signal from parameters.

        Parameters
        ----------
        params : np.ndarray
            [slope, intercept] as bare floats from optimizer.
        x : np.ndarray, optional
            Dye concentrations in M; defaults to the measured concentrations.

        Returns
        -------
        Quantity
            Predicted signal values in au.
        """
        slope, intercept = params
        xx = self.x_data.magnitude if x is None else np.asarray(x, dtype=float)
        result = linear_signal(slope, intercept, xx)
        return Q_(result, 'au')

    def species(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Free dye equals the titrant — a linear calibration has no equilibrium.

        There is nothing to speciate (no host, no complex); the only species is
        the dye itself, at its total concentration.  The simulation applet
        recognises this and shows a note instead of a trivial identity line.
        """
        return {'D': np.asarray(self.x_data.magnitude, dtype=float)}

    def get_conditions(self) -> Dict[str, Any]:
        """Return experimental conditions.

        Returns
        -------
        Dict[str, Any]
            Empty dict for dye-alone (no fixed conditions).
        """
        return {}

    def fit_linear(self):
        """Fit linear model using simple linear regression.

        Returns
        -------
        Tuple[Quantity, Quantity, float, Quantity]
            (slope, intercept, r_squared, rmse) with proper units.
        """
        from core.optimizer.linear_fit import linear_regression

        slope, intercept, r_squared, rmse = linear_regression(
            self.x_data.magnitude,
            self.y_data.magnitude,
        )
        return Q_(slope, 'au/M'), Q_(intercept, 'au'), r_squared, Q_(rmse, 'au')
