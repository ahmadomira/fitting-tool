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

The six binding assays use concentration **association constants** (Ka, M⁻¹) and specified mass-action equilibria. Dye Alone is a linear calibration. The [scientific reference](docs/scientific-summary.md) defines all seven models, assumptions, units, and identifiability limits, with literature citations and mathematical derivations.

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

IDA estimates guest affinity from two coupled equilibria, where dye and guest compete for the same host binding site:

```
H + D ⇌ HD        Ka_dye    (assumed known, measured via DBA first)
H + G ⇌ HG        Ka_guest  (the quantity being fitted)
```

The guest `G` is titrated into a host–dye mixture with fixed host and dye totals. The signal is the additive free- and bound-dye response above, not their ratio. Affinity estimation is conditional on the known dye affinity, totals, optical contrast, and sufficient information in the titration.

#### Guest Displacement Assay (GDA)

GDA shares the same coupled equilibria as IDA, but *dye* is titrated into a host + guest mixture at fixed host and guest totals. Dye displaces guest, and the fitted affinity is again `Ka_guest`. Suitability relative to IDA depends on affinities, concentrations, solubility, and optical contrast. Both assays assume mutually exclusive 1:1 binding and dark host/guest species.

#### Stepwise binding: HG2 and H2G

`DBA_HG2` models `H + G ⇌ HG` followed by `HG + G ⇌ HG2`. `DBA_H2G` instead uses `H + HG ⇌ H2G` as its second step. Both titrate guest into a fixed host total. Each step constant has units M⁻¹; their product is a cumulative constant in M⁻². Signal adds free host, free guest, and the two complexes with a separate response per species. Free-host response is fixed to zero by default. Excess guest favors HG2 in the first model but HG in the second. Macroscopic step constants alone do not establish microscopic cooperativity.

#### Dye Alone

Linear dye-only fluorescence versus concentration determines free-dye response and baseline. It does not determine bound-dye response or correct inner-filter effects. Transfer to binding assays requires matching conditions; calibration-derived bounds or fixed values impose external information. The application does not implement Bayesian priors.

### Why forward modelling

The toolkit evaluates total-concentration mass balances and fits signal in measurement units. This avoids assuming that total titrant equals free titrant and avoids changing the residual through a linearizing transformation. Its unweighted least-squares objective assumes equal signal-error variance for a Gaussian likelihood interpretation; fitting raw data does not establish that this assumption holds. The models use scalar fixed totals and do not automatically correct changing-volume dilution.

### Fitting strategy

The search can encounter local optima, flat directions, and poor scaling. The toolkit uses:

- **Multi-start L-BFGS-B** — many initial vectors (`n_trials`), with logarithmically sampled Ka starts. Optimization itself uses linearly scaled constants.
- **Bounds** — configured intervals constrain the search (Ka defaults: 10⁻⁸ to 10¹² M⁻¹). Bounds supply assumptions; they do not prove physical applicability or identifiability.
- **Quality filter** — candidates pass configured R² and optional relative RMSE thresholds. Optimizer success status alone does not determine acceptance.
- **Representative and spread** — the default representative is an actual accepted fit with highest R² and its predicted curve. Min/max, median/MAD, mean/SD, and central percentiles describe accepted optimized solutions. They are not bootstrap estimates or calibrated confidence intervals. Per-replica pools mix search and replica variability and weight replicas by the number of accepted solutions.
- **Replica filtering** — optional modified Z-scores use median/MAD. Zero MAD currently assigns zero scores, including majority ties with a differing value.

### Identifiability note

At fixed dye total (`DBA_HtoD` and `IDA`), three raw signal coefficients reduce to an effective offset and free/bound contrast. Individual coefficients cannot all be recovered from that curve. This exact ambiguity does not apply to varying-dye `DBA_DtoH` or `GDA`. Stepwise models have a corresponding fixed-host signal ambiguity if `I_H` is freed. Dye calibration measures only free-dye slope and baseline. Even when affinity is structurally identifiable, weak contrast, a limited concentration range, or noise can prevent precise recovery. See the [scientific reference](docs/scientific-summary.md#6-structural-and-practical-identifiability) for conditions, exceptions, and experimental remedies.

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
