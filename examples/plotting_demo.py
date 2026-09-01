"""End-to-end demo of the gui/plotting module.

Shows the full pipeline:
  synthetic GDA data  →  MeasurementSet  →  fit  →  PyQtGraph widgets

Window layout
-------------
┌────────────────────────────┬──────────────────────────┐
│        PlotWidget          │  FitSummaryWidget (top)  │
│  (replicas, avg, fit curve)│  PlotStyleWidget (bottom)│
└────────────────────────────┴──────────────────────────┘

Run with:
    uv run python examples/plotting_demo.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QSplitter

from core.assays.gda import GDAAssay
from core.assays.registry import ASSAY_REGISTRY, AssayType
from core.data_processing.measurement_set import MeasurementSet
from core.data_processing.plotting import prepare_plot_data
from core.models.equilibrium import gda_signal
from core.pipeline.fit_pipeline import FitConfig, fit_measurement_set, summarize_parameters
from gui.plotting import FitSummaryWidget, PlotStyleWidget, PlotWidget

# ---------------------------------------------------------------------------
# Ground-truth parameters and conditions
# ---------------------------------------------------------------------------

TRUE_PARAMS = {
    'Ka_guest': 3e6,  # M⁻¹ — fitted by the optimizer
    'I0': 50.0,  # background signal
    'I_dye_free': 4e7,  # a.u./M
    'I_dye_bound': 2e8,  # a.u./M
}

CONDITIONS = {
    'Ka_dye': 5e5,  # M⁻¹ — known from a prior DBA calibration
    'h0': 10e-6,  # 10 µM host (fixed)
    'g0': 20e-6,  # 20 µM guest — large relative to Ka_guest⁻¹ for good sensitivity
}

# ---------------------------------------------------------------------------
# 1. Generate synthetic GDA data
# ---------------------------------------------------------------------------

np.random.seed(0)

d0_values = np.linspace(0, 3e-4, 11)  # dye titration grid (M)

y_clean = gda_signal(
    I0=TRUE_PARAMS['I0'],
    Ka_guest=TRUE_PARAMS['Ka_guest'],
    I_dye_free=TRUE_PARAMS['I_dye_free'],
    I_dye_bound=TRUE_PARAMS['I_dye_bound'],
    Ka_dye=CONDITIONS['Ka_dye'],
    h0=CONDITIONS['h0'],
    d0_values=d0_values,
    g0=CONDITIONS['g0'],
)


def _noisy(y: np.ndarray, noise_frac: float = 0.1) -> np.ndarray:
    return y + np.random.normal(0, noise_frac * np.abs(y).mean(), size=y.shape)


# Three normal replicas + one obvious outlier (dropped before fitting)
replicas = {
    'A': _noisy(y_clean),
    'B': _noisy(y_clean),
    'C': _noisy(y_clean),
    'r_out': _noisy(y_clean, noise_frac=0.5),
}

rows = []
for replica_id, signal in replicas.items():
    for conc, sig in zip(d0_values, signal):
        rows.append({'replica': replica_id, 'concentration': conc, 'signal': sig})

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 2. Build MeasurementSet and drop the outlier
# ---------------------------------------------------------------------------

ms = MeasurementSet.from_dataframe(
    df,
    replica_col='replica',
    concentration_col='concentration',
    signal_col='signal',
    assay_type='GDA',
    description='Synthetic GDA demo',
)

ms.set_active('r_out', False)

# ---------------------------------------------------------------------------
# 3. Fit
# ---------------------------------------------------------------------------

config = FitConfig(
    n_trials=200,
    custom_bounds={
        'Ka_guest': (1e4, 1e9),
        'I0': (-1e3, 1e4),
        'I_dye_free': (1e7, 1e8),
        'I_dye_bound': (5e7, 5e8),
    },
)

print('Fitting GDA model … (200 trials)')
result = fit_measurement_set(ms, GDAAssay, CONDITIONS, config)

# Console summary
ka_fit = result.parameters.get('Ka_guest', float('nan'))
ka_stats = next((s.stats for s in summarize_parameters(result) if s.key == 'Ka_guest' and not s.is_log), None)
ka_range = f'({ka_stats["min"]:.2e}, {ka_stats["max"]:.2e})' if ka_stats else '(no pool)'
print(f'  Ka_guest  true : {TRUE_PARAMS["Ka_guest"]:.3e} M⁻¹')
print(f'  Ka_guest  fit  : {ka_fit:.3e} {ka_range} M⁻¹')
print(f'  R²             : {result.r_squared:.4f}')
print(f'  Fits passing   : {result.n_passing} / {result.n_total}')
print(f'  RMSE           : {result.rmse:.4e}')

# ---------------------------------------------------------------------------
# 4. Build Qt window and wire up widgets
# ---------------------------------------------------------------------------

app = QApplication.instance() or QApplication(sys.argv)

meta = ASSAY_REGISTRY[AssayType.GDA]

# Plot widget
pw = PlotWidget(
    x_label=meta.x_label,
    y_label=meta.y_label,
    title='GDA Demo — synthetic data',
)
plot_data = prepare_plot_data(ms, fit_results=[result], show_dropped=True)
pw.update_plot(plot_data)
pw.set_fit_results([result])

# Fit summary
fit_summary = FitSummaryWidget()
fit_summary.update_result(result)

# Style panel — wired so every ParameterTree change calls apply_style live
style_panel = PlotStyleWidget()
style_panel.style_changed.connect(pw.apply_style)

# Layout: plot left, (summary top / style bottom) right
right_splitter = QSplitter(Qt.Orientation.Vertical)
right_splitter.addWidget(fit_summary)
right_splitter.addWidget(style_panel)
right_splitter.setSizes([250, 350])

main_splitter = QSplitter(Qt.Orientation.Horizontal)
main_splitter.addWidget(pw)
main_splitter.addWidget(right_splitter)
main_splitter.setSizes([700, 300])
main_splitter.setWindowTitle('SupraSimFit — Plotting Demo')
main_splitter.resize(1000, 600)
main_splitter.show()

sys.exit(app.exec())
