"""Per-session application state for the unified fitting GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.assays.registry import AssayType
from core.data_processing.measurement_set import MeasurementSet
from core.pipeline.fit_pipeline import FitConfig, FitResult
from core.pipeline.sensitivity import SensitivityResult
from core.units import Quantity


@dataclass
class SessionState:
    """Mutable state for one fitting session (one tab).

    All widgets within a FittingSession share this state via the
    FittingSession coordinator — widgets never hold references to each other.
    """

    # Data
    measurement_set: Optional[MeasurementSet] = None
    source_file: Optional[str] = None

    # Assay configuration
    assay_type: AssayType = AssayType.GDA
    conditions: dict[str, Any] = field(default_factory=dict)

    # Fit configuration
    fit_config: FitConfig = field(default_factory=FitConfig)
    custom_bounds: Optional[dict[str, tuple[Quantity, Quantity]]] = None

    # Preprocessing steps spec — passed to apply_preprocessing()
    preprocessing_steps: list[dict] = field(default_factory=list)

    # Results
    fit_results: list[FitResult] = field(default_factory=list)
    dye_alone_result: Optional[FitResult] = None
    sensitivity_result: Optional[SensitivityResult] = None

    # Plot display unit (concentration unit used for the x-axis). Lives in
    # session state so it survives style-widget reconstructions and is the
    # source of truth for the value that ends up in style['axes']['x_unit'].
    display_unit: str = 'µM'

    def has_data(self) -> bool:
        return self.measurement_set is not None

    def clear_results(self) -> None:
        self.fit_results.clear()
        self.dye_alone_result = None
