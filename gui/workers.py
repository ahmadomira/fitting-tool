"""Background worker threads for long-running operations."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from core.assays.base import BaseAssay
from core.data_processing.measurement_set import MeasurementSet
from core.pipeline.fit_pipeline import FitConfig, PerReplicaFitError, fit_measurement_set
from core.pipeline.sensitivity import SensitivityConfig, run_sensitivity


class FitWorker(QThread):
    """Run :func:`fit_measurement_set` in a background thread.

    Prevents the GUI from freezing during multi-start L-BFGS-B optimization.

    Parameters
    ----------
    ms : MeasurementSet
        Measurement data (will use average signal).
    assay_cls : type[BaseAssay]
        Assay class to fit.
    conditions : dict
        Assay conditions (Ka_dye, h0, etc.).
    config : FitConfig
        Optimizer configuration.
    source_file : str, optional
        Original filename, stored in FitResult metadata.

    Signals
    -------
    finished(FitResult)
        Emitted on successful completion.
    error(str)
        Emitted if an exception is raised.
    """

    finished = pyqtSignal(object)  # FitResult
    error = pyqtSignal(str)

    def __init__(
        self,
        ms: MeasurementSet,
        assay_cls: type[BaseAssay],
        conditions: dict[str, Any],
        config: FitConfig,
        source_file: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._ms = ms
        self._assay_cls = assay_cls
        self._conditions = conditions
        self._config = config
        self._source_file = source_file

    def run(self) -> None:
        try:
            result = fit_measurement_set(
                self._ms,
                self._assay_cls,
                self._conditions,
                self._config,
            )
            self.finished.emit(result)
        except PerReplicaFitError as exc:
            details = '\n'.join(f'  - {rid}: {reason}' for rid, reason in exc.failures.items())
            self.error.emit(f'{exc}\n\nFailures per replica:\n{details}')
        except Exception as exc:
            self.error.emit(f'The fit could not be completed:\n{exc}')


class SensitivityWorker(QThread):
    """Run :func:`run_sensitivity` in a background thread.

    Keeps the GUI responsive while the sensitivity analysis re-runs the fit
    many times over perturbed inputs.

    Parameters
    ----------
    ms : MeasurementSet
        Measurement data (its averaged active signal is used).
    assay_cls : type[BaseAssay]
        Assay class to build each perturbed fit with.
    conditions : dict
        Baseline assay conditions (Ka_dye, h0, etc.) as pint Quantities.
    fit_config : FitConfig
        Per-fit optimizer configuration.
    sens_config : SensitivityConfig
        What to perturb and how.

    Signals
    -------
    progress(int, int)
        Emitted as ``(done, total)`` after every fit.
    finished(SensitivityResult)
        Emitted on successful completion.
    error(str)
        Emitted if the analysis cannot be completed.
    """

    progress = pyqtSignal(int, int)  # (done, total)
    finished = pyqtSignal(object)  # SensitivityResult
    error = pyqtSignal(str)

    def __init__(
        self,
        ms: MeasurementSet,
        assay_cls: type[BaseAssay],
        conditions: dict[str, Any],
        fit_config: FitConfig,
        sens_config: SensitivityConfig,
        parent=None,
    ):
        super().__init__(parent)
        self._ms = ms
        self._assay_cls = assay_cls
        self._conditions = conditions
        self._fit_config = fit_config
        self._sens_config = sens_config
        self._cancelled = False

    def cancel(self) -> None:
        """Request early termination; surfaced to the driver via ``should_cancel``."""
        self._cancelled = True

    def run(self) -> None:
        try:
            result = run_sensitivity(
                self._ms,
                self._assay_cls,
                self._conditions,
                self._fit_config,
                self._sens_config,
                progress=self.progress.emit,
                should_cancel=lambda: self._cancelled,
            )
            self.finished.emit(result)
        except ValueError as exc:
            self.error.emit(f'The sensitivity analysis could not run:\n{exc}')
        except Exception as exc:
            self.error.emit(f'The sensitivity analysis could not be completed:\n{exc}')
