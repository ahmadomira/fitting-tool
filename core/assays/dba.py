"""DBA (Direct Binding Assay) data containers.

DBA measures direct binding between host and dye:
    H + D ⇌ HD  (Ka_dye to be fitted)

Two titration modes:
- Host→Dye (DBA_HtoD): Host is titrated into fixed dye concentration
- Dye→Host (DBA_DtoH): Dye is titrated into fixed host concentration

Target: Ka_dye (association constant for host-dye)
Concentrations are analytical totals in the measured solution.
Signal depends on free-dye and complex brightness; binding can enhance or quench it.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from core.assays.base import BaseAssay
from core.assays.registry import AssayType
from core.models.equilibrium import dba_signal, dba_species
from core.units import Q_, Quantity


@dataclass
class DBAAssay(BaseAssay):
    """Direct Binding Assay data container.

    This class handles both Host→Dye and Dye→Host titrations.
    The mode is determined by the assay_type attribute.

    Attributes
    ----------
    x_data : Quantity
        Titrant concentrations (M) - host for HtoD, dye for DtoH.
    y_data : Quantity
        Observed signal values (au).
    fixed_conc : Quantity
        Fixed component concentration (M) - dye for HtoD, host for DtoH.
    mode : str
        Titration mode: 'HtoD' or 'DtoH'.
    """

    fixed_conc: Optional[Quantity] = None
    mode: str = 'DtoH'

    assay_type: AssayType = field(init=False)

    def __post_init__(self):
        """Validate data and set assay type based on mode."""
        super().__post_init__()

        if self.fixed_conc is None:
            raise ValueError('fixed_conc is required (fixed component concentration)')

        if not isinstance(self.fixed_conc, Quantity):
            raise TypeError(f'fixed_conc must be a pint Quantity, got {type(self.fixed_conc).__name__}')

        # Normalize to base units so .magnitude is always M
        object.__setattr__(self, 'fixed_conc', self.fixed_conc.to('M'))

        if np.ndim(self.fixed_conc.magnitude) != 0 or not np.isfinite(self.fixed_conc.magnitude):
            raise ValueError('fixed_conc must be a finite scalar')

        if self.fixed_conc.magnitude <= 0:
            raise ValueError('fixed_conc must be positive')

        if self.mode == 'HtoD':
            object.__setattr__(self, 'assay_type', AssayType.DBA_HtoD)
        elif self.mode == 'DtoH':
            object.__setattr__(self, 'assay_type', AssayType.DBA_DtoH)
        else:
            raise ValueError(f"mode must be 'HtoD' or 'DtoH', got '{self.mode}'")

    def forward_model(self, params: np.ndarray, x: np.ndarray | None = None) -> Quantity:
        """Compute predicted signal from parameters.

        Parameters
        ----------
        params : np.ndarray
            [Ka_dye, I0, I_dye_free, I_dye_bound] as bare floats
            from the optimizer, in canonical units.

        Returns
        -------
        Quantity
            Predicted signal values in au.
        """
        Ka_dye, I0, I_dye_free, I_dye_bound = params
        xx = self.x_data.magnitude if x is None else np.asarray(x, dtype=float)

        result = dba_signal(
            I0=I0,
            Ka_dye=Ka_dye,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            x_titrant=xx,
            y_fixed=self.fixed_conc.magnitude,
            mode=self.mode,
        )
        return Q_(result, 'au')

    def species(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Free host, free dye and host-dye complex (M) across the titration."""
        Ka_dye = params[0]
        return dba_species(Ka_dye, self.x_data.magnitude, self.fixed_conc.magnitude, mode=self.mode)

    def get_conditions(self) -> Dict[str, Any]:
        """Return experimental conditions.

        Returns
        -------
        Dict[str, Any]
            {'fixed_conc': ..., 'mode': ...}
        """
        return {
            'fixed_conc': self.fixed_conc,
            'mode': self.mode,
        }


# Convenience factory functions for clearer API
def create_dba_host_to_dye(
    host_conc: Quantity,
    signal: Quantity,
    dye_conc: Quantity,
    **kwargs,
) -> DBAAssay:
    """Create DBA assay for host-to-dye titration.

    Parameters
    ----------
    host_conc : Quantity
        Total host concentrations, converted to M - the titrant.
    signal : Quantity
        Observed signal values in au.
    dye_conc : Quantity
        Fixed total dye concentration, converted to M.
    **kwargs
        Additional arguments passed to DBAAssay.

    Returns
    -------
    DBAAssay
        Configured for HtoD mode.
    """
    return DBAAssay(
        x_data=host_conc,
        y_data=signal,
        fixed_conc=dye_conc,
        mode='HtoD',
        **kwargs,
    )


def create_dba_dye_to_host(
    dye_conc: Quantity,
    signal: Quantity,
    host_conc: Quantity,
    **kwargs,
) -> DBAAssay:
    """Create DBA assay for dye-to-host titration.

    Parameters
    ----------
    dye_conc : Quantity
        Total dye concentrations, converted to M - the titrant.
    signal : Quantity
        Observed signal values in au.
    host_conc : Quantity
        Fixed total host concentration, converted to M.
    **kwargs
        Additional arguments passed to DBAAssay.

    Returns
    -------
    DBAAssay
        Configured for DtoH mode.
    """
    return DBAAssay(
        x_data=dye_conc,
        y_data=signal,
        fixed_conc=host_conc,
        mode='DtoH',
        **kwargs,
    )
