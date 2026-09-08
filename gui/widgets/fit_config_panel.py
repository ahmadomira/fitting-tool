"""FitConfigPanel — optimizer configuration (n_trials, RMSE factor, min R²)."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QWidget,
)

from core.pipeline.fit_pipeline import FitConfig
from gui.widgets.info_button import InfoButton, InfoGroupBox
from gui.widgets.numeric_inputs import NoScrollDoubleSpinBox, NoScrollSpinBox

_TRIALS_HELP_HTML = """
<h3>Trials &mdash; Number of Starting Points</h3>
<p>Each trial starts a local fit from a different parameter guess.
More trials explore more of the allowed range and take longer; they do
not add experimental observations or guarantee the best possible fit.</p>
<p>Start with the default 100. If repeated runs disagree, increase trials
and check bounds, concentrations and the sampled binding range. Persistent
disagreement can reflect weak information or model mismatch as well as
optimization difficulty.</p>
"""

_RMSE_HELP_HTML = """
<h3>Trim by RMSE Factor</h3>
<p>Optional, off by default. After the R<sup>2</sup> filter, retain trials
with RMSE no greater than <i>factor &times; best RMSE</i> for that fit.
RMSE measures typical residual size in signal units; the factor is
unitless.</p>
<p>Tighter filtering narrows the accepted pool by selection. Its range is
a description of retained trials, not a confidence interval. Choose filters
for a stated purpose and inspect the fit and residuals.</p>
"""

_PER_REPLICA_HELP_HTML = """
<h3>Fit per Replica</h3>
<p><b>Average mode:</b> fit the mean of active replicas. The retained
trials describe repeated optimization of that one curve.</p>
<p><b>Per-replica mode (default):</b> fit each active replica separately,
then pool its accepted trials. Replicas with more accepted trials contribute
more entries; this does not mean they contain more independent observations.
Failed replicas are reported.</p>
<p>Use per-replica results to inspect disagreement between traces.
The report shows a representative fit and the minimum/maximum values in
the accepted pool. These are descriptive search results, not confidence
intervals or standard errors. Pool size does not replace the number of
independent experiments.</p>
<p>Reliable scientific uncertainty needs an analysis that accounts for
measurement noise, replica structure and uncertain fixed inputs.</p>
"""


_RESCALE_HELP_HTML = """
<h3>Rescale Parameters</h3>
<p>Uses data-derived scales to improve numerical conditioning during
optimization, then reports parameters in their physical units.</p>
<p>This invertible change of coordinates preserves the model and bounds.
It can improve convergence but cannot add information or resolve parameters
that the experiment cannot distinguish. Leave it on by default; turn it off
when comparing numerical behavior.</p>
"""


_R2_HELP_HTML = """
<h3>Minimum R<sup>2</sup></h3>
<p>Reject trials below this threshold (default 0.90).
<i>R<sup>2</sup> = 1 &minus; SSE/SST</i>, where SSE is the summed squared
residual and SST is the summed squared deviation from the data mean.
R<sup>2</sup> is unitless and undefined for a constant observed signal.</p>
<p>A value near one indicates agreement with the data, but does not
establish the binding model or parameter precision. If every trial fails,
inspect residuals, signal range, bounds and concentrations before relaxing
the threshold.</p>
"""


_SECTION_HELP_HTML = """
<h3>Fit Configuration</h3>
<p>Controls for the multi-start optimiser: how many trials to run, how
strictly to filter them, and whether to fit each replica
independently. See the <i>i</i> next to each setting for details.</p>
"""


class FitConfigPanel(InfoGroupBox):
    """Editor for :class:`~core.pipeline.fit_pipeline.FitConfig` parameters.

    Signals
    -------
    config_changed()
        Emitted whenever any field value changes.
    """

    config_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__('Fit Configuration', 'Fit Configuration', _SECTION_HELP_HTML, parent)
        self._setup_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_config(self) -> FitConfig:
        return FitConfig(
            n_trials=self._trials_spin.value(),
            rmse_threshold_factor=(self._rmse_spin.value() if self._rmse_check.isChecked() else None),
            min_r_squared=self._r2_spin.value(),
            rescale_parameters=self._rescale_check.isChecked(),
            per_replica=self._per_replica_check.isChecked(),
        )

    def set_config(self, config: FitConfig) -> None:
        self._trials_spin.setValue(config.n_trials)
        factor = config.rmse_threshold_factor
        self._rmse_check.setChecked(factor is not None)
        self._rmse_spin.setEnabled(factor is not None)
        if factor is not None:
            self._rmse_spin.setValue(factor)
        self._r2_spin.setValue(config.min_r_squared)
        self._rescale_check.setChecked(config.rescale_parameters)
        self._per_replica_check.setChecked(config.per_replica)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        form = QFormLayout(self)

        self._trials_spin = NoScrollSpinBox()
        self._trials_spin.setRange(10, 10_000)
        self._trials_spin.setSingleStep(10)
        self._trials_spin.setValue(100)
        self._trials_spin.setToolTip('Number of multi-start L-BFGS-B optimization trials.')
        self._trials_spin.valueChanged.connect(self.config_changed)
        form.addRow('Trials:', self._with_info(self._trials_spin, 'Trials', _TRIALS_HELP_HTML))

        # Optional RMSE trim — off by default; the R² floor is the primary gate.
        self._rmse_check = QCheckBox()
        self._rmse_check.setChecked(FitConfig().rmse_threshold_factor is not None)  # default: off
        self._rmse_check.setToolTip('Optionally trim the valid pool by RMSE relative to the best fit (off by default).')
        self._rmse_spin = NoScrollDoubleSpinBox()
        self._rmse_spin.setRange(1.0, 10.0)
        self._rmse_spin.setSingleStep(0.1)
        self._rmse_spin.setDecimals(2)
        self._rmse_spin.setValue(1.5)
        self._rmse_spin.setEnabled(self._rmse_check.isChecked())
        self._rmse_spin.setToolTip('Drop valid fits with RMSE > (best_RMSE × factor). Only applied when enabled.')
        self._rmse_check.toggled.connect(self._rmse_spin.setEnabled)
        self._rmse_check.toggled.connect(self.config_changed)
        self._rmse_spin.valueChanged.connect(self.config_changed)
        form.addRow('Trim by RMSE factor:', self._rmse_trim_row())

        self._r2_spin = NoScrollDoubleSpinBox()
        self._r2_spin.setRange(0.0, 1.0)
        self._r2_spin.setSingleStep(0.01)
        self._r2_spin.setDecimals(2)
        self._r2_spin.setValue(0.90)
        self._r2_spin.setToolTip('Minimum R² for a trial to pass the acceptance filter.')
        self._r2_spin.valueChanged.connect(self.config_changed)
        form.addRow('Min R²:', self._with_info(self._r2_spin, 'Min R²', _R2_HELP_HTML))

        self._rescale_check = QCheckBox()
        self._rescale_check.setChecked(FitConfig().rescale_parameters)
        self._rescale_check.setToolTip(
            'Rescale parameters to O(1) for the optimiser; results are shown in physical units.'
        )
        self._rescale_check.toggled.connect(self.config_changed)
        form.addRow(
            'Rescale parameters:',
            self._with_info(self._rescale_check, 'Rescale parameters for fitting', _RESCALE_HELP_HTML),
        )

        self._per_replica_check = QCheckBox()
        self._per_replica_check.setChecked(FitConfig().per_replica)
        self._per_replica_check.setToolTip(
            'Fit each replica separately and pool accepted trials; ranges are descriptive, not confidence intervals.'
        )
        self._per_replica_check.toggled.connect(self.config_changed)
        form.addRow(
            'Fit per replica:',
            self._with_info(self._per_replica_check, 'Fit per replica', _PER_REPLICA_HELP_HTML),
        )

    def _rmse_trim_row(self) -> QWidget:
        """Row: [enable checkbox] [factor spinbox] [info] for the optional trim."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(15)
        lay.addWidget(self._rmse_check)
        self._rmse_spin.setFixedWidth(120)
        lay.addWidget(self._rmse_spin)
        lay.addWidget(InfoButton('RMSE factor', _RMSE_HELP_HTML))
        lay.addStretch(1)
        return row

    @staticmethod
    def _with_info(widget: QWidget, title: str, html: str) -> QWidget:
        """Wrap ``widget`` in an HBox together with a trailing info button.

        The spinbox gets a uniform fixed width so the three Fit Config
        rows line up vertically regardless of the label column width,
        and the trailing stretch keeps the (spinbox, info-button) pair
        pinned to the left of the field column.
        """
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(15)
        if not isinstance(widget, QCheckBox):
            widget.setFixedWidth(120)
        lay.addWidget(widget)
        lay.addWidget(InfoButton(title, html))
        lay.addStretch(1)
        return row
