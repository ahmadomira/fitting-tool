"""SensitivityPanel — configure a Ka input-sensitivity run.

The panel exposes the knobs of :class:`~core.pipeline.sensitivity.SensitivityConfig`:
the analysis *mode* (joint / one-at-a-time / heatmap), how many random draws to
take, a ``±%`` variation for every perturbable input (the titrant plus each
concentration / binding-constant condition), the two heatmap axes, and an
optional fixed seed. It emits :attr:`run_requested` / :attr:`cancel_requested`;
the owning session runs the worker and calls :meth:`set_running` to flip the
button. For ``DYE_ALONE`` (no Ka to analyse) the whole panel is disabled.
"""

from __future__ import annotations

import re

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.assays.registry import ASSAY_REGISTRY, AssayType
from core.pipeline.sensitivity import TITRANT, SensitivityConfig, SensitivityMode, perturbable_inputs
from core.units import Q_
from gui.widgets.assay_conditions import ASSAY_CONDITIONS
from gui.widgets.info_button import InfoButton, InfoGroupBox
from gui.widgets.numeric_inputs import NoScrollDoubleSpinBox, NoScrollSpinBox

_SECTION_HELP_HTML = """
<h3>K<sub>a</sub> Sensitivity Analysis</h3>
<p>A fitted association constant is only as good as the <i>inputs</i> you feed
the model — the titrant concentrations and the fixed conditions
(<i>[Host]<sub>0</sub></i>, <i>[Dye]<sub>0</sub></i>, <i>K<sub>a,dye</sub></i>, …).
Each carries pipetting / calibration error, and the fit silently absorbs it by
shifting the reported K<sub>a</sub>.</p>
<p>This tool re-runs the fit many times with those inputs jittered inside the
<i>±%</i> band you set below, and shows the resulting spread of K<sub>a</sub> —
so you can see how much of your uncertainty comes from the inputs rather than
the fit itself.</p>
"""

_MODE_HELP_HTML = """
<h3>Mode</h3>
<ul>
  <li><b>Joint</b> — every selected input is jittered <i>together</i> on each
      draw. One K<sub>a</sub> histogram showing the combined effect of input
      error.</li>
  <li><b>One-at-a-time</b> — each input is jittered <i>alone</i>, the others
      held fixed. One histogram per input, so you can rank which input
      K<sub>a</sub> is most sensitive to.</li>
  <li><b>Heatmap</b> — sweep two chosen inputs on a regular grid and fit at
      every cell, giving a 2-D K<sub>a</sub> surface.</li>
</ul>
"""

_SAMPLES_HELP_HTML = """
<h3>Samples</h3>
<p>Number of random draws taken for each histogram (Joint and One-at-a-time
only). More samples give a smoother, more trustworthy distribution but take
proportionally longer. 200 is a sensible default; raise it for a publication
figure.</p>
"""

_DELTA_HELP_HTML = """
<h3>Input variation (±%)</h3>
<p>The relative uncertainty of each input. A value of <b>5</b> means that input
is drawn uniformly from <i>−5%</i> to <i>+5%</i> of its configured value on each
fit. Set an input to <b>0</b> to hold it fixed (exclude it from the analysis).</p>
<p>Use realistic numbers: pipetting error is often a few percent, a stated
literature K<sub>a,dye</sub> may be uncertain by 10–20%.</p>
"""

_HEATMAP_HELP_HTML = """
<h3>Heatmap axes</h3>
<p>Choose the two inputs to sweep. Each axis runs from <i>−</i> to <i>+</i> that
input's <i>±%</i> value (set above), split into <i>Grid steps</i> points. The fit
runs once per cell — cost grows as <i>steps²</i>, so 11×11 = 121 fits is a good
starting resolution.</p>
"""

_SEED_HELP_HTML = """
<h3>Fixed seed</h3>
<p>Tick to make the random draws reproducible: the same seed always produces the
same histogram. Leave unticked for a fresh random sample each run.</p>
"""


def _plain(html_label: str) -> str:
    """Strip HTML tags so a rich condition label reads in a plain QComboBox."""
    return re.sub(r'<[^>]+>', '', html_label)


class SensitivityPanel(InfoGroupBox):
    """Configuration panel for a :class:`SensitivityConfig` run.

    Signals
    -------
    run_requested()
        Emitted when the user clicks *Run* (panel idle).
    cancel_requested()
        Emitted when the user clicks the button while a run is in progress.
    """

    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__('Ka Sensitivity', 'Ka Sensitivity Analysis', _SECTION_HELP_HTML, parent)
        self._assay_type: AssayType = AssayType.GDA
        self._running = False
        # input key -> its ±% spinbox; rebuilt per assay type.
        self._delta_spins: dict[str, NoScrollDoubleSpinBox] = {}
        self._setup_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_assay_type(self, assay_type: AssayType) -> None:
        """Rebuild the ±% rows and heatmap axis combos for ``assay_type``.

        Disables the whole panel for ``DYE_ALONE`` (no association constant to
        analyse).
        """
        self._assay_type = assay_type
        self._rebuild_inputs(assay_type)
        self.setEnabled(assay_type is not AssayType.DYE_ALONE and bool(self._delta_spins))

    def current_config(self) -> SensitivityConfig:
        """Build a :class:`SensitivityConfig` from the current widget state."""
        mode: SensitivityMode = self._mode_combo.currentData()
        delta_pct = {key: spin.value() for key, spin in self._delta_spins.items() if spin.value() > 0}
        seed = int(self._seed_spin.value()) if self._seed_check.isChecked() else None
        return SensitivityConfig(
            mode=mode,
            n_samples=self._samples_spin.value(),
            delta_pct=delta_pct,
            seed=seed,
            x_key=self._x_combo.currentData(),
            y_key=self._y_combo.currentData(),
            n_steps=self._steps_spin.value(),
        )

    def set_running(self, running: bool) -> None:
        """Flip the Run/Cancel button and lock the inputs while a run is live."""
        self._running = running
        self._run_btn.setText('Cancel' if running else 'Run Sensitivity Analysis')
        for w in self._input_widgets():
            w.setEnabled(not running)
        if not running:
            # Seed value follows its checkbox once inputs are re-enabled.
            self._seed_spin.setEnabled(self._seed_check.isChecked())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem('Joint', SensitivityMode.JOINT)
        self._mode_combo.addItem('One-at-a-time', SensitivityMode.OAT)
        self._mode_combo.addItem('Heatmap', SensitivityMode.HEATMAP)
        self._mode_combo.currentIndexChanged.connect(self._update_mode_visibility)
        form.addRow('Mode:', self._with_info(self._mode_combo, 'Mode', _MODE_HELP_HTML))

        self._samples_spin = NoScrollSpinBox()
        self._samples_spin.setRange(10, 100_000)
        self._samples_spin.setSingleStep(10)
        self._samples_spin.setValue(200)
        self._samples_spin.setToolTip('Random draws per histogram (Joint / One-at-a-time).')
        self._samples_row = self._with_info(self._samples_spin, 'Samples', _SAMPLES_HELP_HTML, label='Samples:')
        form.addRow(self._samples_row)
        layout.addLayout(form)

        # ±% variation rows — rebuilt per assay type.
        self._delta_group = InfoGroupBox('Input variation (±%)', 'Input variation', _DELTA_HELP_HTML)
        self._delta_form = QFormLayout(self._delta_group)
        self._delta_form.setContentsMargins(8, 4, 8, 8)
        layout.addWidget(self._delta_group)

        # Heatmap axis selectors — only visible in Heatmap mode.
        self._heatmap_group = InfoGroupBox('Heatmap axes', 'Heatmap axes', _HEATMAP_HELP_HTML)
        hm_form = QFormLayout(self._heatmap_group)
        hm_form.setContentsMargins(8, 4, 8, 8)
        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        self._steps_spin = NoScrollSpinBox()
        self._steps_spin.setRange(3, 101)
        self._steps_spin.setValue(11)
        self._steps_spin.setToolTip('Grid points per axis (fits ≈ steps²).')
        hm_form.addRow('X axis:', self._x_combo)
        hm_form.addRow('Y axis:', self._y_combo)
        hm_form.addRow('Grid steps:', self._steps_spin)
        layout.addWidget(self._heatmap_group)

        # Fixed-seed row.
        self._seed_check = QCheckBox('Fixed seed')
        self._seed_check.setToolTip('Reproducible draws — the same seed gives the same result.')
        self._seed_spin = NoScrollSpinBox()
        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.setValue(0)
        self._seed_spin.setEnabled(False)
        self._seed_check.toggled.connect(self._seed_spin.setEnabled)
        seed_row = QWidget()
        seed_lay = QHBoxLayout(seed_row)
        seed_lay.setContentsMargins(0, 0, 0, 0)
        seed_lay.setSpacing(8)
        seed_lay.addWidget(self._seed_check)
        self._seed_spin.setFixedWidth(140)
        seed_lay.addWidget(self._seed_spin)
        seed_lay.addWidget(InfoButton('Fixed seed', _SEED_HELP_HTML))
        seed_lay.addStretch(1)
        layout.addWidget(seed_row)

        self._run_btn = QPushButton('Run Sensitivity Analysis')
        self._run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self._run_btn)

        self._rebuild_inputs(self._assay_type)
        self._update_mode_visibility()

    def _rebuild_inputs(self, assay_type: AssayType) -> None:
        """Rebuild the ±% rows and heatmap axis combos for ``assay_type``."""
        self._delta_spins.clear()
        while self._delta_form.rowCount():
            self._delta_form.removeRow(0)

        fields = ASSAY_CONDITIONS[assay_type][1]
        # Perturbable-input classification lives in the core layer; feed it the
        # condition Quantities so titrant/concentration/binding order is canonical.
        base = {f.key: Q_(f.default, f.unit) for f in fields if f.unit_type in ('concentration', 'binding_constant')}
        labels = {f.key: f.label for f in fields}
        labels[TITRANT] = f'{ASSAY_REGISTRY[assay_type].x_label} (titrant)'
        inputs = perturbable_inputs(assay_type, base)

        self._x_combo.clear()
        self._y_combo.clear()
        for key, _kind in inputs:
            spin = NoScrollDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix(' %')
            spin.setValue(5.0)
            spin.setToolTip('± relative uncertainty of this input (0 = hold fixed).')
            lbl = QLabel(labels[key])
            lbl.setTextFormat(Qt.TextFormat.RichText)
            self._delta_form.addRow(lbl, spin)
            self._delta_spins[key] = spin

            self._x_combo.addItem(_plain(labels[key]), key)
            self._y_combo.addItem(_plain(labels[key]), key)

        # Default heatmap axes: first two distinct perturbable inputs.
        if self._y_combo.count() > 1:
            self._y_combo.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Slots / helpers
    # ------------------------------------------------------------------

    def _update_mode_visibility(self) -> None:
        is_heatmap = self._mode_combo.currentData() is SensitivityMode.HEATMAP
        self._samples_row.setVisible(not is_heatmap)
        self._heatmap_group.setVisible(is_heatmap)

    def _on_run_clicked(self) -> None:
        if self._running:
            self.cancel_requested.emit()
        else:
            self.run_requested.emit()

    def _input_widgets(self) -> list[QWidget]:
        """Every control that must lock while a run is in progress."""
        return [
            self._mode_combo,
            self._samples_spin,
            self._seed_check,
            self._seed_spin,
            self._x_combo,
            self._y_combo,
            self._steps_spin,
            *self._delta_spins.values(),
        ]

    @staticmethod
    def _with_info(widget: QWidget, title: str, html: str, *, label: str | None = None) -> QWidget:
        """Wrap ``widget`` (optionally with a leading label) plus a trailing info button."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        if label is not None:
            lay.addWidget(QLabel(label))
        lay.addWidget(widget)
        lay.addWidget(InfoButton(title, html))
        lay.addStretch(1)
        return row
