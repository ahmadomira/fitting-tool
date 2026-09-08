"""Data-driven parameter rescaling for well-conditioned fitting.

Given an assay with raw concentration and signal data, we derive two
scaling constants from the data::

    c_max = max(|x_data|)  in M
    s_max = max(|y_data|)  in au

and rescale each fit parameter to an O(1) tilded value::

    theta_tilded[k] = theta_raw[k] * scale[k]

For a parameter whose unit is a product of concentration (M) and
``au`` exponents, the scale factor is derived mechanically from the
pint unit: each ``au`` component contributes ``s_max ** (-exp)`` and
each concentration component contributes ``(c_max / mag_in_M) **
(-exp)``, where ``mag_in_M`` is how many M the unit represents
(e.g. 1 for ``M``, 1e-6 for ``µM``).

This is an *exact* affine reparameterization. The minimizer of the
rescaled loss equals the minimizer of the raw loss under the
bijection above. Its purpose is numerical: it reduces differences due to
parameter units. It cannot remove structural degeneracy or guarantee
well-conditioned curvature or optimizer convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pint

from core.units import Q_, ureg

# Canonical dimensions used to classify a parameter's unit components.
# ``au`` carries its own ``[signal]`` dimension (see core/units.py), so a
# signal component is dimensionally distinct from a concentration component
# and from a genuinely dimensionless number — no reliance on the historical
# "au is dimensionless" coincidence.
_SIGNAL_DIM = ureg.Unit('au').dimensionality
_CONC_DIM = ureg.Unit('M').dimensionality


@dataclass(frozen=True)
class ParamScaler:
    """Per-parameter affine rescaler derived from assay data.

    Attributes
    ----------
    scales : np.ndarray
        Per-parameter multiplicative scale.
        ``theta_tilded = theta_raw * scales``.
    loss_scale : float
        Divisor for the objective value:
        ``L_tilded(theta_tilded) = L(theta_raw) / loss_scale``.
        Equal to ``s_max ** 2`` where ``s_max = max|y_data|``.
    """

    scales: np.ndarray
    loss_scale: float

    @classmethod
    def from_assay(cls, assay) -> 'ParamScaler':
        """Build a scaler from a ``BaseAssay`` instance.

        Raises
        ------
        ValueError
            If the data are degenerate (peak is 0, negative, or non-finite).
        """
        x_mag = np.asarray(assay.x_data.to('M').magnitude, dtype=float)
        y_mag = np.asarray(assay.y_data.to('au').magnitude, dtype=float)
        c_max = float(np.max(np.abs(x_mag))) if x_mag.size else 0.0
        s_max = float(np.max(np.abs(y_mag))) if y_mag.size else 0.0

        if not np.isfinite(c_max) or c_max <= 0:
            raise ValueError(f'Cannot rescale parameters: peak concentration is {c_max} (need positive, finite).')
        if not np.isfinite(s_max) or s_max <= 0:
            raise ValueError(f'Cannot rescale parameters: peak signal is {s_max} (need positive, finite).')

        defaults = assay.get_default_bounds()
        scales = np.array(
            [_scale_factor(defaults[k][0].units, c_max, s_max) for k in assay.parameter_keys],
            dtype=float,
        )
        return cls(scales=scales, loss_scale=s_max * s_max)

    def to_internal(self, theta_raw: np.ndarray) -> np.ndarray:
        """Raw parameters → tilded."""
        return np.asarray(theta_raw, dtype=float) * self.scales

    def to_external(self, theta_tilded: np.ndarray) -> np.ndarray:
        """Tilded parameters → raw."""
        return np.asarray(theta_tilded, dtype=float) / self.scales

    def bounds_to_internal(
        self,
        bounds_raw: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Rescale raw (lower, upper) bound tuples to tilded space."""
        return [(lo * s, hi * s) for (lo, hi), s in zip(bounds_raw, self.scales)]

    def wrap_objective(
        self,
        obj_raw: Callable[[np.ndarray], float],
    ) -> Callable[[np.ndarray], float]:
        """Return a tilded-parameter objective wrapping a raw-parameter one.

        ``obj_tilded(theta_tilded) = obj_raw(theta_raw) / loss_scale``
        """
        scales = self.scales
        loss_scale = self.loss_scale

        def obj_tilded(theta_tilded: np.ndarray) -> float:
            return obj_raw(theta_tilded / scales) / loss_scale

        return obj_tilded


def _scale_factor(unit: pint.Unit, c_max: float, s_max: float) -> float:
    """Compute the affine rescaling factor for a parameter unit.

    Each component of ``unit`` (walked via pint's ``_units`` map) is
    classified by its **dimension**, not by ``.dimensionless``:

    - ``[signal]`` components (``au``) contribute ``s_max ** (-exp)``;
    - ``[concentration]`` components (``M``/``µM``/…) contribute
      ``(c_max / mag_in_M) ** (-exp)``, where ``mag_in_M`` is the
      component's size in M (1 for ``M``, 1e-6 for ``µM``);
    - a genuinely dimensionless component (a pure number/ratio) is left
      unscaled (contributes 1.0).

    Because ``au`` has its own ``[signal]`` dimension, a signal component
    can no longer be confused with a ``1/M`` binding constant or with a
    dimensionless ratio. A component in any other dimension (e.g. time,
    length) is a programming error and raises ``ValueError``.
    """
    factor = 1.0
    for name, exp in unit._units.items():
        one = Q_(1.0, name)
        dim = one.dimensionality
        if dim == _SIGNAL_DIM:
            factor *= s_max ** (-exp)
        elif dim == _CONC_DIM:
            factor *= (c_max / one.to('M').magnitude) ** (-exp)
        elif one.dimensionless:
            continue
        else:
            raise ValueError(f"Cannot rescale: unit '{name}' is neither signal, concentration, nor dimensionless.")
    return factor
