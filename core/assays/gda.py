"""GDA (Guest Displacement Assay) data container.

In GDA, dye is titrated into a solution containing host and guest.
The dye competes with the guest for binding to the host, displacing it:
    H + D ⇌ HD  (Ka_dye known from DBA)
    H + G ⇌ HG  (Ka_guest to be fitted)

Titrant: Dye (d0 varies)
Fixed: Host (h0), Guest (g0)
Target: Ka_guest (association constant for host-guest)
Signal trend: ↑ Increases as dye binds and displaces guest
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from core.assays.base import BaseAssay
from core.assays.registry import AssayType
from core.models.equilibrium import gda_signal, gda_species
from core.units import Q_, Quantity


@dataclass
class GDAAssay(BaseAssay):
    """Guest Displacement Assay data container.

    Attributes
    ----------
    x_data : Quantity
        Dye concentrations (M) - the independent variable (titrant).
    y_data : Quantity
        Observed signal values (au).
    Ka_dye : Quantity
        Known association constant for host-dye binding (1/M).
    h0 : Quantity
        Total host concentration (M) - fixed.
    g0 : Quantity
        Total guest concentration (M) - fixed.
    """

    Ka_dye: Optional[Quantity] = None
    h0: Optional[Quantity] = None
    g0: Optional[Quantity] = None

    assay_type: AssayType = field(init=False, default=AssayType.GDA)

    def __post_init__(self):
        """Validate data and conditions."""
        super().__post_init__()

        if self.Ka_dye is None:
            raise ValueError('Ka_dye is required (known host-dye association constant)')
        if self.h0 is None:
            raise ValueError('h0 is required (total host concentration)')
        if self.g0 is None:
            raise ValueError('g0 is required (total guest concentration)')

        if not isinstance(self.Ka_dye, Quantity):
            raise TypeError(f'Ka_dye must be a pint Quantity, got {type(self.Ka_dye).__name__}')
        if not isinstance(self.h0, Quantity):
            raise TypeError(f'h0 must be a pint Quantity, got {type(self.h0).__name__}')
        if not isinstance(self.g0, Quantity):
            raise TypeError(f'g0 must be a pint Quantity, got {type(self.g0).__name__}')

        # Normalize to base units so .magnitude is always M / 1/M
        object.__setattr__(self, 'Ka_dye', self.Ka_dye.to('1/M'))
        object.__setattr__(self, 'h0', self.h0.to('M'))
        object.__setattr__(self, 'g0', self.g0.to('M'))

        for key in ('Ka_dye', 'h0', 'g0'):
            value = getattr(self, key).magnitude
            if np.ndim(value) != 0 or not np.isfinite(value):
                raise ValueError(f'{key} must be a finite scalar')

        if self.Ka_dye.magnitude <= 0:
            raise ValueError('Ka_dye must be positive')
        if self.h0.magnitude <= 0:
            raise ValueError('h0 must be positive')
        if self.g0.magnitude < 0:
            raise ValueError('g0 must be positive (or zero)')

    def forward_model(self, params: np.ndarray, x: np.ndarray | None = None) -> Quantity:
        """Compute predicted signal from parameters.

        Parameters
        ----------
        params : np.ndarray
            [Ka_guest, I0, I_dye_free, I_dye_bound] as bare floats
            from the optimizer, in canonical units.

        Returns
        -------
        Quantity
            Predicted signal values in au.
        """
        Ka_guest, I0, I_dye_free, I_dye_bound = params
        xx = self.x_data.magnitude if x is None else np.asarray(x, dtype=float)

        result = gda_signal(
            I0=I0,
            Ka_guest=Ka_guest,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            Ka_dye=self.Ka_dye.magnitude,
            h0=self.h0.magnitude,
            d0_values=xx,
            g0=self.g0.magnitude,
        )
        return Q_(result, 'au')

    def species(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Free host, dye, guest and both complexes (M) across the dye titration."""
        Ka_guest = params[0]
        return gda_species(Ka_guest, self.Ka_dye.magnitude, self.h0.magnitude, self.x_data.magnitude, self.g0.magnitude)

    def get_conditions(self) -> Dict[str, Any]:
        """Return experimental conditions.

        Returns
        -------
        Dict[str, Any]
            {'Ka_dye': ..., 'h0': ..., 'g0': ...}
        """
        return {
            'Ka_dye': self.Ka_dye,
            'h0': self.h0,
            'g0': self.g0,
        }
