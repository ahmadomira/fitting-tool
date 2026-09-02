<p align="center">
   <img src="assets/logo-suprasense.svg" alt="Suprasense Logo" width="80%"/>
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![Build and Release](https://github.com/ASDSE/SupraSimFit/actions/workflows/build_and_release.yml/badge.svg)](https://github.com/ASDSE/SupraSimFit/actions/workflows/build_and_release.yml)
[![Latest Release](https://img.shields.io/github/v/release/ASDSE/SupraSimFit?label=Latest%20Release)](https://github.com/ASDSE/SupraSimFit/releases/latest)

# Molecular Binding Assay Fitting Toolkit

A desktop application for fitting equilibrium binding models to fluorescence titration data. It is aimed at supramolecular and biochemical researchers who need to extract association constants (Ka) from indicator-displacement, guest-displacement, and direct-binding assays using robust forward-modelling rather than linearised transforms.

The app ships a PyQt6 GUI with interactive plots, per-replica outlier removal, and one-click import of BMG plate-reader Excel exports.

## Download

Packaged builds for each OS are published on the [Releases page](https://github.com/ASDSE/SupraSimFit/releases/latest):

- macOS — `SupraSimFit-<version>-macos.zip`
- Windows — `SupraSimFit-<version>-windows.zip`
- Linux — `SupraSimFit-<version>-linux.zip`

Unzip and launch the executable or `.app` bundle. No Python installation required.

## Quick start (from source)

Requirements: Python **3.13+** and [**uv**](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ASDSE/SupraSimFit.git
cd SupraSimFit
uv sync
uv run run_app.py
```

To try the app immediately without loading your own data, click **Demo IDA** in the toolbar — it loads a bundled IDA dataset and runs a fit with sensible defaults.

## Usage

### 1. Load data

Use **Load Data** (Ctrl+O) in the toolbar. Supported input formats:

| Format | Extension | Notes |
|---|---|---|
| Tab-separated text | `.txt` | Multi-replica blocks; see below |
| Comma-separated | `.csv` | Long-format, wide-format, or multi-replica blocks |
| BMG plate reader | `.xlsx` / `.xls` | Auto-detected; non-BMG Excel files fall back to a structured reader |

**TXT layout** — each replica is a block that starts with a header row (`var` and `signal`, separated by a tab). Lines beginning with `#` are treated as comments.

```
var     signal
0.0     506.246
2.985e-05       1064.85
...
var     signal
0.0     503.100
2.985e-05       1059.21
...
```

**CSV layout** — either long-format (recommended) or wide-format with one column per replica. Column names are matched case-insensitively; `concentration`, `conc`, `x`, `[conc]`, or `titrant` are accepted for the independent variable, and `signal`, `y`, `fluorescence`, `intensity`, or `emission` for the dependent variable.

```
concentration,signal,replica
1e-7,506.246,0
2e-7,612.300,0
1e-7,503.100,1
2e-7,610.000,1
```

Concentrations are stored internally in **Molar**. The GUI accepts input in nM/µM/mM/M and converts automatically. Example datasets with reference parameters live in [data/](data/) — see [data/Readme.md](data/Readme.md).

### 2. Configure and fit

1. Pick the assay type in the **Assay** panel.
2. Enter the known experimental conditions for that assay (e.g. `[Host]₀`, `[Dye]₀`, `Ka_dye`). These fields are unit-aware and accept any supported concentration or binding-constant unit.
3. In the **Bounds** panel, review or tighten the parameter bounds. Defaults cover a wide physically reasonable range; narrower bounds produce better-conditioned fits when prior information is available.
4. In the **Fit Configuration** panel, adjust:
   - **Number of starts** — how many multi-start trials the optimiser runs (default 100). More starts → more robust estimate, at the cost of runtime.
   - **R² / RMSE filtering** — minimum R² and RMSE tolerance factor used to reject failed fits before aggregation.
5. Optionally use the **Outlier Removal** panel to drop noisy replicas via a modified Z-score filter (default threshold 3.5, MAD-based).
6. Click **Run Fit**. Fitting runs off the main thread, and results appear in the plot and summary panel when it finishes.

See the [Scientific background](#scientific-background) below for how each of these configuration options maps onto the underlying physics.

### 3. Export results

From the **File** menu:

- **Export Fit Results (JSON)** — full result ensemble (parameters, statistics, source file). Re-importable via *Import Results*.
- **Export Results (TXT)** — human-readable report.
- **Export Raw Data** — round-trip the loaded measurements back to `.txt` or `.csv`.
- **Export Plot** — save the current plot as PNG or SVG.

## Scientific background

### Binding models

All assays are described with **association constants** (Ka) and rigorous mass-action equilibria.

#### Direct Binding Assay (DBA)

Direct measurement of host–guest or dye–host binding interactions, available in both titration modes. Monitors spectroscopic changes upon complex formation, but is limited in complex matrices due to competitive binding from naturally occurring interferents.

A host `H` binds directly to a spectroscopically active dye `D`:

```
H + D ⇌ HD        Ka_dye = [HD] / ([H][D])
```

The observed fluorescence signal is modelled as a four-parameter linear combination:

```
I = I0 + I_dye_free · [D_free] + I_dye_bound · [HD]
```

where `I0` absorbs baseline offsets and `I_dye_free`, `I_dye_bound` are the per-species molar signal coefficients. The app supports both titration modes: host titrated into dye (`DBA_HtoD`) and dye titrated into host (`DBA_DtoH`).

#### Indicator Displacement Assay (IDA)

Determines binding constants by competitive displacement of indicator dyes — enabling detection in complex biological matrices through ultra-high-affinity reporter pairs. IDA involves two coupled equilibria, where the reporter dye and the analyte guest compete for the same host binding site:

```
H + D ⇌ HD        Ka_dye    (assumed known, measured via DBA first)
H + G ⇌ HG        Ka_guest  (the quantity being fitted)
```

The guest `G` is titrated into a preformed `HD` complex. As `G` displaces `D`, the observed signal tracks the free-dye / bound-dye ratio, letting the fit recover `Ka_guest` from the same 4-parameter signal model as DBA applied to the coupled mass-balance at each titration point.

#### Guest Displacement Assay (GDA)

Quantifies guest binding affinity by monitoring displacement of a preformed host–indicator complex — particularly valuable for spectroscopically silent hosts and guests, and superior for insoluble or weakly binding guests that are difficult to measure by IDA.

Mechanistically, GDA shares the same coupled equilibria as IDA, but the *dye* is titrated into a pre-equilibrated host + guest mixture instead of the guest being titrated into `HD`. The fitted quantity is again `Ka_guest`.

#### Dye Alone

Linear calibration fitting for indicator dyes — establishes baseline fluorescence properties and helps correct for inner-filter effects in competitive binding assays. Concretely, a linear fit of dye-only fluorescence against dye concentration yields the free-dye response (`I_dye_free`) and baseline (`I0`), which can then be used as priors or fixed values for subsequent DBA/IDA/GDA fits.

### Why forward modelling

Historical fitting methods (Scatchard, Hill, double-reciprocal) rearrange the binding isotherm so it can be fit with linear regression. Those transforms:

- distort the measurement noise structure (non-uniform error weighting in the transformed space),
- lose validity in ligand-depletion regimes, and
- cannot handle coupled competitive equilibria without strong approximations.

This toolkit instead fits the raw signal directly with the full nonlinear model — so the residual is evaluated in the measurement space where the experimental error is best understood.

### Fitting strategy

The nonlinear binding likelihood is typically multi-modal, so single-start gradient methods get trapped in local minima. The toolkit uses:

- **Multi-start L-BFGS-B** — the constrained quasi-Newton optimiser is launched from many random initial parameter vectors (the `n_trials` knob), with Ka parameters sampled in log space (they span several orders of magnitude).
- **Physical bounds** — every parameter has lower/upper bounds so the search stays inside a thermodynamically reasonable region (Ka defaults: 10⁻⁸ to 10¹² M⁻¹). Tighten these in the Bounds panel when you have prior knowledge.
- **Quality filter** — after all starts complete, failed fits are rejected by `min_r_squared` and by an RMSE tolerance factor (`rmse_threshold_factor × best_RMSE`). Only survivors are aggregated.
- **Real fits, honestly summarised** — the reported estimate is the **representative fit**: an actual member of the surviving ensemble, the one with the highest R². Being a real fit, it lies on the model's degenerate manifold and reproduces the plotted curve exactly. Alongside it the app quotes the **range (min, max) across every accepted fit**, with the median ± MAD, mean ± SD and central-68% interval in the summary table. Each of these is read straight from the ensemble, so it describes a skewed or nonlinear parameter distribution as faithfully as a symmetric one.
- **Replica outlier removal** — the optional Z-score preprocessing step uses a *modified* Z-score based on the median and MAD, so it is not biased by the very outliers it is trying to flag.

### Identifiability note

In the 4-parameter signal model, the coefficients `I0`, `I_dye_free`, and `I_dye_bound` are **structurally degenerate** for DBA and IDA — only `Ka` is uniquely identifiable from a single titration. The app fits the full 4-parameter model because it reconstructs the observed signal faithfully, but only `Ka` should be reported as a physical constant across datasets. Fixing the signal coefficients with a dye-alone calibration (see [data/Readme.md](data/Readme.md)) breaks the degeneracy when stronger constraints are needed.

## Development

```bash
uv sync                # install runtime + dev dependencies
uv run pytest          # run the test suite
uv run run_app.py      # launch the GUI
```

The GUI can also be launched directly as a module: `uv run python -m gui.main_window`.

Build a standalone executable with PyInstaller:

```bash
uv run --with pyinstaller pyinstaller --clean -y \
    --distpath ./dist --workpath ./build SupraSimFit.spec
```

## Dependencies

Runtime dependencies are pinned in [pyproject.toml](pyproject.toml):

- `numpy`, `scipy` — numerics and L-BFGS-B optimisation
- `pandas`, `openpyxl` — tabular I/O and Excel / BMG plate-reader import
- `pint` — unit-aware parameter conversion
- `PyQt6`, `pyqtgraph` — GUI and interactive plots

## License

GPL 3.0 — see [LICENSE](LICENSE).

## References

1. Sinn, S., Spuling, E., Bräse, S., & Biedermann, F. (2019). Rational design and implementation of a cucurbit[8]uril-based indicator-displacement assay for application in blood serum. *Chemical Science*, 10(28), 6584-6593. <https://doi.org/10.1039/C9SC00705A>
2. Sinn, S., Krämer, J., & Biedermann, F. (2020). Teaching old indicators even more tricks: binding affinity measurements with the guest-displacement assay (GDA). *Chemical Communications*, 56(49), 6620-6623. <https://doi.org/10.1039/D0CC01841D>

---

**Contact:** contact@suprabank.org
