"""H2G assay: stepwise 2:1 host–guest direct binding.

Guest is titrated into a fixed host concentration; two hosts bind one
guest in two successive steps:
    H + G  ⇌ HG    (Ka_HG)
    HG + H ⇌ H2G   (Ka_H2G)

Titrant: Guest (g0 varies)
Fixed:   Host (h0)
Target:  Ka_HG, Ka_H2G (stepwise association constants)

This uses the same mass-balance solver as :class:`~core.assays.hg2.HG2Assay`,
with host and guest roles exchanged inside the solver. Its guest-titration
curve differs. The two constants can be weakly constrained by the available
concentration range and species responses; a good fit does not prove uniqueness.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from core.assays.base import BaseAssay
from core.assays.registry import AssayType
from core.models.equilibrium import h2g_signal, h2g_species
from core.units import Q_, Quantity


@dataclass
class H2GAssay(BaseAssay):
    """Stepwise 2:1 host–guest direct-binding assay (two hosts bind one guest).

    Attributes
    ----------
    x_data : Quantity
        Guest concentrations (M) — the titrant.
    y_data : Quantity
        Observed signal values (au).
    h0 : Quantity
        Total (fixed) host concentration (M).
    """

    h0: Optional[Quantity] = None

    assay_type: AssayType = field(init=False, default=AssayType.DBA_H2G)
    model_name: ClassVar[str] = 'equilibrium_h2g'

    def __post_init__(self):
        """Validate data and the fixed host concentration."""
        super().__post_init__()

        if self.h0 is None:
            raise ValueError('h0 is required (total host concentration)')
        if not isinstance(self.h0, Quantity):
            raise TypeError(f'h0 must be a pint Quantity, got {type(self.h0).__name__}')

        # Normalize to base units so .magnitude is always M
        object.__setattr__(self, 'h0', self.h0.to('M'))

        if np.ndim(self.h0.magnitude) != 0 or not np.isfinite(self.h0.magnitude):
            raise ValueError('h0 must be a finite scalar')

        if self.h0.magnitude <= 0:
            raise ValueError('h0 must be positive')

    def forward_model(self, params: np.ndarray, x: np.ndarray | None = None) -> Quantity:
        """Compute predicted signal from parameters.

        Parameters
        ----------
        params : np.ndarray
            ``[Ka_HG, Ka_H2G, I0, I_G, I_H, I_HG, I_H2G]`` as bare floats
            from the optimizer, in canonical units.

        Returns
        -------
        Quantity
            Predicted signal values in au.
        """
        Ka_HG, Ka_H2G, I0, I_G, I_H, I_HG, I_H2G = params
        xx = self.x_data.magnitude if x is None else np.asarray(x, dtype=float)

        result = h2g_signal(
            I0=I0,
            Ka_HG=Ka_HG,
            Ka_H2G=Ka_H2G,
            I_G=I_G,
            I_H=I_H,
            I_HG=I_HG,
            I_H2G=I_H2G,
            h0=self.h0.magnitude,
            g0_values=xx,
        )
        return Q_(result, 'au')

    def species(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Free host, guest, HG and H2G complexes (M) across the guest titration."""
        Ka_HG, Ka_H2G = params[0], params[1]
        return h2g_species(Ka_HG, Ka_H2G, self.h0.magnitude, self.x_data.magnitude)

    def get_conditions(self) -> Dict[str, Any]:
        """Return experimental conditions.

        Returns
        -------
        Dict[str, Any]
            ``{'h0': ...}``
        """
        return {'h0': self.h0}
