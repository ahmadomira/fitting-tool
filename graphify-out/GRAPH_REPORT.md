# Graph Report - fitting_app  (2026-09-02)

## Corpus Check
- 147 files · ~117,755 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2664 nodes · 5357 edges · 133 communities (110 shown, 15 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 312 edges (avg confidence: 0.93)
- Token cost: 199,283 input · 0 output

## Community Hubs (Navigation)
- Fit Pipeline Entry Points
- Assay Base and Registry
- Fit Result Filtering
- Simulation Knob Model
- Release Build and Science Docs
- Multi-Export Dialog
- Flat Tab Bar
- MeasurementSet Writing
- Ensight Plate Reader
- Ensemble Collapse
- FitResult Serialization
- MeasurementSet Container
- Competitive Signal Models
- Per-Replica Fitting
- BMG and XLSX Readers
- Dye-Alone Linear Calibration
- Simulation Noise Controls
- Distribution Export Dialog
- MeasurementSet Tests
- Forward Simulation Core
- Fit Summary Table
- Stepwise Binding Equilibria
- Preprocessing and Plot Prep
- Fitting Session Wiring
- Simulation Parameter Control
- Assay Condition Fields
- CSV Reader
- Parameter Scaling
- DBA Assay
- Distribution Box Drawing
- Bounds Panel Rows
- Image Export Tests
- Distribution Widget
- Tab-Separated TXT I/O
- Update Download Worker
- Parameter Label Formatting
- Concentration Vector Helpers
- Z-Score Replica Filter
- Multi-File Replica Loading
- Plot Widget Layout
- GDA Assay
- Results Text and CSV Export
- Plot Style Widget
- Synthetic Test Fixtures
- IDA Assay
- Fit Metrics and Dense Curves
- Scientific Axis Ticks
- Data Panel Tests
- JASCO Reader
- Main Window Shell
- Draggable Annotation Item
- Assay Config Panel
- Replica Activation Panel
- Plot Annotation Tests
- DBA Forward Model
- Fitting Workflow Diagram
- Plot Image Export
- Plot Style Application
- Data Panel Loading
- Ensemble Statistics Tests
- I/O Reader Protocols
- Distribution Subplot Toggles
- Reader Registry Dispatch
- Update Check Wiring
- Plot Color Constants
- BaseAssay Contract Methods
- Plot Data Preparation
- Bounds Resolution Helpers
- GDA Forward Model
- JASCO Data Parsing
- Species Speciation Plot
- Numeric Input Widgets
- Parameter Kind Units
- Session UI Grouping
- Simulation Window
- Release Version Check
- IDA Species Grid
- Initial Guess Generation
- Session Export Actions
- Axis Label Composition
- Speciation Contracts
- Core and I/O Public API
- Pint Unit Architecture
- App Launch Entry Point
- Assay Type Selector
- Concentration Table Editing
- Fit Config Panel
- Registry Test Doubles
- Species Label Formatting
- Single-File Channel Loading
- BaseAssay Fail-Fast Tests
- Plot Style Templates
- Simulation Settings I/O
- Sidebar Scroll Area
- Multi-File Load Dialog
- Session Layout Guards
- Stepwise GUI Labels
- Composite Plot Layout
- Info Button Group Box
- Brent Solver Robustness
- Dye-Alone Assay Model
- H2G Stepwise Assay
- HG2 Stepwise Assay
- Reader Writer Protocol Methods
- JASCO Format Sniffing
- Background Fit Worker
- Stepwise Assay Contracts
- Toolbar Menu Buttons
- Data Panel Help UI
- Channel Combo Tests
- Demo Fit Trigger
- GUI Test Fixtures
- Assay Category Metadata
- BaseAssay Conditions
- H2G Conditions
- HG2 Conditions
- Concentration Boundary Audit
- Parameter Descriptions
- Linked Axis Plots
- MCP Server Config
- Registry Isolation Fixture
- Formats Package
- Simulation Package Init
- Pint Contexts Note
- Project Metadata

## God Nodes (most connected - your core abstractions)
1. `MeasurementSet` - 100 edges
2. `AssayType` - 83 edges
3. `FittingSession` - 73 edges
4. `FitConfig` - 66 edges
5. `FitResult` - 64 edges
6. `BaseAssay` - 60 edges
7. `PlotWidget` - 51 edges
8. `DistributionWidget` - 50 edges
9. `IDAAssay` - 44 edges
10. `GDAAssay` - 43 edges

## Surprising Connections (you probably didn't know these)
- `Signal response model (linear species combination)` --semantically_similar_to--> `4-parameter signal model`  [INFERRED] [semantically similar]
  docs/scientific-summary.md → Readme.md
- `Forward Modeling approach` --semantically_similar_to--> `Why forward modelling over linearised transforms`  [INFERRED] [semantically similar]
  docs/scientific-summary.md → Readme.md
- `Bound-constrained quasi-Newton optimization (L-BFGS-B)` --semantically_similar_to--> `Multi-start L-BFGS-B fitting strategy`  [INFERRED] [semantically similar]
  docs/scientific-summary.md → Readme.md
- `Non-negative signal and binding parameters` --semantically_similar_to--> `Physical parameter bounds`  [INFERRED] [semantically similar]
  docs/scientific-summary.md → Readme.md
- `Ensemble fitting and robust median-based aggregation` --semantically_similar_to--> `Replica outlier removal (modified Z-score)`  [INFERRED] [semantically similar]
  docs/scientific-summary.md → Readme.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Quantities-at-edges unit architecture** — docs_pint_findings_shared_registry, docs_pint_findings_au_signal_dimension, docs_pint_findings_normalization_boundary, docs_pint_findings_float_only_core, docs_pint_findings_boundary_validation, docs_pint_findings_serialization_tokens, docs_pint_findings_no_silent_fallbacks [EXTRACTED 1.00]
- **Signal-coefficient degeneracy: model, consequence, mitigation** — readme_four_parameter_signal_model, readme_identifiability_note, readme_dye_alone, docs_scientific_summary_signal_coefficient_degeneracy, docs_scientific_summary_effective_offset_contrast, docs_scientific_summary_degeneracy_inspection_steps, docs_scientific_summary_degeneracy_mitigation [EXTRACTED 1.00]
- **Tag-driven multi-OS release pipeline** — _github_workflows_build_and_release_verify_tag_matches_version, _github_workflows_build_and_release_build, _github_workflows_build_and_release_pyinstaller_build, _github_workflows_build_and_release_per_os_packaging, _github_workflows_build_and_release_publish_release [EXTRACTED 1.00]
- **Multi-start trial loop (sample, optimize, score, filter, repeat)** — docs_fitting_workflow_generate_multiple_parameter_sets, docs_fitting_workflow_optimize_parameters, docs_fitting_workflow_minimize_residuals, docs_fitting_workflow_compute_rmse_r2, docs_fitting_workflow_filter_fits_using_rmse_r2, docs_fitting_workflow_more_trials [EXTRACTED 1.00]
- **Initial parameter seeding branch** — docs_fitting_workflow_initial_guess_available, docs_fitting_workflow_load_parameters, docs_fitting_workflow_initialize_randomly, docs_fitting_workflow_generate_multiple_parameter_sets [EXTRACTED 1.00]
- **Per-replica outcome handling and replica loop** — docs_fitting_workflow_any_good_fits, docs_fitting_workflow_aggregate_good_fits, docs_fitting_workflow_compute_final_model, docs_fitting_workflow_plot_experimental_vs_simulated, docs_fitting_workflow_skip_replica_no_valid_fits, docs_fitting_workflow_more_replicas, docs_fitting_workflow_save_results [EXTRACTED 1.00]

## Communities (133 total, 15 thin omitted)

### Community 0 - "Fit Pipeline Entry Points"
Cohesion: 0.06
Nodes (54): fit_assay(), fit_measurement_set(), FitConfig, Configuration for the fitting pipeline. ``rmse_threshold_factor`` is an…, Fit an assay using multi-start optimisation. Parameters ---------- assay :…, Convenience: build an assay from a MeasurementSet and fit it. When…, assert_within_tolerance(), h2g_clean() (+46 more)

### Community 1 - "Assay Base and Registry"
Cohesion: 0.06
Nodes (39): ABC, BaseAssay, Base class for assay data containers. This module defines the abstract base…, Abstract base class for assay data containers. Subclasses must implement: -…, Validate data after initialization., Get metadata from the assay registry., Parameter names for this assay type., Number of parameters to fit. (+31 more)

### Community 2 - "Fit Result Filtering"
Cohesion: 0.06
Nodes (36): Fit linear model using simple linear regression. Returns -------…, filter_by_r_squared(), filter_by_rmse(), Filtering utilities for fit results. Filter multi-start fit attempts down to…, Filter fit attempts by RMSE threshold. Parameters ---------- results :…, Filter fit attempts by minimum R² value. Parameters ---------- results :…, Select the valid-fit pool kept for aggregation. The absolute R² floor is the…, select_valid_fits() (+28 more)

### Community 3 - "Simulation Knob Model"
Cohesion: 0.07
Nodes (47): AssayType, Enumeration of all supported assay types. Each assay type has associated…, knobs_for(), _ParamSpec, Registry-driven knob model for the simulation applet. A *knob* is one live,…, Default slider range for a *condition* knob, derived from its default.…, Return the full live-knob list for *assay_type*: conditions ∪ parameters., slider_bounds() (+39 more)

### Community 4 - "Release Build and Science Docs"
Cohesion: 0.05
Nodes (52): build job (macOS/Linux/Windows matrix), Per-OS artifact packaging, publish_release job, PyInstaller bundle build step, Version drift guard (tag vs __version__), Build and Release workflow, F1 — _scale_factor classified units by .dimensionless, F2 — au given its own [signal] dimension (+44 more)

### Community 5 - "Multi-Export Dialog"
Cohesion: 0.06
Nodes (36): ExportMultipleDialog, Exception, Path, QDialog, QWidget, Consolidated multi-artefact export dialog. Lets the user pick which artefacts…, Pick artefacts, folder, and filename base; export in one click., _Row (+28 more)

### Community 6 - "Flat Tab Bar"
Cohesion: 0.07
Nodes (29): ButtonPosition, _FlatCloseButton, FlatTabBar, FlatTabWidget, QLineEdit, QToolButton, Flat / minimal tab widgets — one reusable style for every tab in the app.…, Tab bar that keeps an optional inline "+" button after the last tab. (+21 more)

### Community 7 - "MeasurementSet Writing"
Cohesion: 0.06
Nodes (33): DataFrame, In-memory container for multi-replica measurement data. MeasurementSet holds a…, Build a MeasurementSet from a long-form DataFrame. The DataFrame must contain…, _header_comment(), _iter_points(), Path, Writers that serialise a :class:`MeasurementSet` back to disk. These produce…, Yield ``(replica_idx, concentration, signal)`` tuples for every point. Includes… (+25 more)

### Community 8 - "Ensight Plate Reader"
Cohesion: 0.06
Nodes (28): EnsightReader, format_channel_label(), Any, DataFrame, ndarray, Path, PerkinElmer EnSight plate-reader CSV export parser. EnSight (Kaleido 3.x)…, Capture the few lines above the first 'Result for ...' block. (+20 more)

### Community 9 - "Ensemble Collapse"
Cohesion: 0.07
Nodes (39): collapse(), describe(), describe_log10(), EnsembleResult, _mad(), _mean(), _median(), ndarray (+31 more)

### Community 10 - "FitResult Serialization"
Cohesion: 0.06
Nodes (30): _config_to_dict(), Any, Convert to a JSON-safe dictionary. ``Quantity`` fields are converted to…, Reconstruct a ``FitResult`` from a dictionary. Parameters ---------- d : dict…, Serialize a FitConfig to a plain dict., Re-point *result* at a different valid fit from its pool. Mutates *result* in…, select_representative(), The per-replica aggregate must carry the flat pool of every valid trial from… (+22 more)

### Community 11 - "MeasurementSet Container"
Cohesion: 0.05
Nodes (27): MeasurementSet, Any, ndarray, Quantity, Total number of replicas (active + dropped)., Number of currently active replicas., Number of concentration points per replica., Replica IDs currently marked active. (+19 more)

### Community 12 - "Competitive Signal Models"
Cohesion: 0.06
Nodes (33): competitive_signal_point(), gda_signal(), ida_signal(), Compute signal for a single point in competitive binding equilibrium. Delegates…, Compute GDA signal across dye concentration range. In GDA (Guest Displacement…, Compute IDA signal across guest concentration range. In IDA (Indicator…, Forward models for molecular binding assays., linear_signal() (+25 more)

### Community 13 - "Per-Replica Fitting"
Cohesion: 0.07
Nodes (33): fit_measurement_set_per_replica(), _model_name_for_assay(), PerReplicaFitError, Derive a model name string from an assay instance. Each assay family declares…, Raised when per-replica fitting cannot produce any surviving fit. Carries a…, Fit every active replica independently and pool passing trials. Each active…, _make_ida_data(), Generate synthetic IDA data (Guest titrated into Host+Dye). (+25 more)

### Community 14 - "BMG and XLSX Readers"
Cohesion: 0.08
Nodes (31): _extract_metadata(), _find_bmg_sheet(), _find_header_row(), is_bmg_workbook(), parse_bmg_workbook(), Any, DataFrame, BMG plate reader export parser. BMG CLARIOstar / FLUOstar / PHERAstar… (+23 more)

### Community 15 - "Dye-Alone Linear Calibration"
Cohesion: 0.08
Nodes (26): DyeAloneAssay, Any, Dye-alone calibration data container. Attributes ---------- x_data : np.ndarray…, Return experimental conditions. Returns ------- Dict[str, Any] Empty dict for…, bounds_from_dye_alone(), fit_linear_assay(), FitResult, Whether the fit was successful (at least one passing fit). (+18 more)

### Community 16 - "Simulation Noise Controls"
Cohesion: 0.07
Nodes (25): _display_factor(), _labelled(), NoiseControl, _parse_floats(), QWidget, Reusable live controls for the simulation applet. These are generic and assay-…, Titrant setup: a maximum concentration (0 → max in N points), or an explicit…, Switch to a custom vector from Molar values (used on data import). (+17 more)

### Community 17 - "Distribution Export Dialog"
Cohesion: 0.11
Nodes (13): DistributionsExportConfig, QDialog, QWidget, Append one checkbox + HTML label to a shared single-row layout., Return ``width_in`` for a preset, or ``None`` for 'custom'. Height is no longer…, DPI input is meaningless for vector SVG; grey it out., Return value when the dialog is accepted. ``height_in`` is derived from…, Pick subplots, layout, dimensions, and format for the distributions image.… (+5 more)

### Community 18 - "MeasurementSet Tests"
Cohesion: 0.08
Nodes (18): DataFrame, Tests for MeasurementSet construction and behaviour., Raw data arrays are read-only after construction., Active mask management., iter_replicas, get_replica_signal, average_signal., Build a simple long-form DataFrame with known values., Yielded signals are views (not copies) into the underlying array., Dropping a replica must exclude it from the mean. _sample_dataframe gives… (+10 more)

### Community 19 - "Forward Simulation Core"
Cohesion: 0.10
Nodes (35): _as_count(), _build_assay(), build_concentration_vector(), Any, ndarray, Forward simulation of titration data. This is the inverse of the fitting…, Construct an assay for forward evaluation over *x_vector* (M). Mirrors…, Evaluate an assay's forward model at *x_vector* from explicit parameters. Uses… (+27 more)

### Community 20 - "Fit Summary Table"
Cohesion: 0.08
Nodes (27): FitSummaryWidget, QWidget, Read-only display of ``FitResult`` statistics. Layout ------ -…, Populate the widget from a ``FitResult``. ``Estimate`` is the representative…, Pint HTML value with the trailing unit stripped (units have a column)., Merged 'central ± spread' cell (units stripped; shown in the Units column)., Merged '[min, max]' range cell., Offer Best, Median, Worst, and the sticky Selected representative. Best/Worst… (+19 more)

### Community 21 - "Stepwise Binding Equilibria"
Cohesion: 0.09
Nodes (27): dba_species(), h2g_signal(), h2g_species(), hg2_signal(), hg2_species(), ndarray, Forward models for equilibrium binding assays. This module contains pure…, Solve for free-ligand concentration in a stepwise 1:2 (core·ligand₂) system. A… (+19 more)

### Community 22 - "Preprocessing and Plot Prep"
Cohesion: 0.08
Nodes (22): Data processing: containers, preprocessing, and plot helpers. Public API…, Plot data preparation helpers. Prepares structured data from a…, apply_preprocessing(), get_step(), PreprocessingStep, Any, Protocol, Minimalist preprocessing pipeline for MeasurementSet. Provides a tiny registry… (+14 more)

### Community 23 - "Fitting Session Wiring"
Cohesion: 0.08
Nodes (12): FittingSession, One complete fitting workflow: data → preprocess → configure → fit → visualize.…, Set a manual tab name (double-click rename); empty reverts to auto., Resolve the tab title: custom name, else dataset stem, else 'Untitled'., Stem-only version of :meth:`_default_save_name` (no tag, no suffix)., Import fit results from JSON and replot., Show the consolidated multi-artefact export dialog., DataPanel announces a new plot-display unit — propagate to plot. (+4 more)

### Community 24 - "Simulation Parameter Control"
Cohesion: 0.10
Nodes (13): _clamp(), ParameterControl, Current value in the base unit (M / M⁻¹ / au / au·M⁻¹)., Current value as a pint Quantity (for assay conditions)., Serializable knob state for settings save., (Re)build the maximum-concentration knob for a new assay's titrant., Short Unicode unit label for a static (non-selectable) unit., A single live knob: slider + labelled editable bounds + exact value. Internals… (+5 more)

### Community 25 - "Assay Condition Fields"
Cohesion: 0.09
Nodes (19): ConditionField, Schema for one assay condition input field. Attributes ---------- key : str…, Quantity, Per-unit decimals via the shared policy in ``gui.widgets.numeric_inputs``., Spinbox paired with an optional unit selector. For fields with display choices…, Return value as a Quantity in the base unit (M or M⁻¹)., Set widget from a base-unit value., _UnitWidget (+11 more)

### Community 26 - "CSV Reader"
Cohesion: 0.11
Nodes (16): CsvReader, DataFrame, Path, True if every column name parses as a number (suggesting no header row)., Reader for CSV measurement files., Read a CSV measurement file. Parameters ---------- path : Path Path to input…, Series, CsvReader handles varied CSV dialects and headers. (+8 more)

### Community 27 - "Parameter Scaling"
Cohesion: 0.10
Nodes (26): ParamScaler, ndarray, Rescale raw (lower, upper) bound tuples to tilded space., Return a tilded-parameter objective wrapping a raw-parameter one.…, Compute the affine rescaling factor for a parameter unit. Each component of…, Per-parameter affine rescaler derived from assay data. Attributes ----------…, Build a scaler from a ``BaseAssay`` instance. Raises ------ ValueError If the…, Raw parameters → tilded. (+18 more)

### Community 28 - "DBA Assay"
Cohesion: 0.09
Nodes (19): create_dba_dye_to_host(), create_dba_host_to_dye(), DBAAssay, Any, ndarray, Quantity, DBA (Direct Binding Assay) data containers. DBA measures direct binding between…, Free host, free dye and host-dye complex (M) across the titration. (+11 more)

### Community 29 - "Distribution Box Drawing"
Cohesion: 0.11
Nodes (19): _box_stats(), _log10_positive(), ndarray, PlotItem, ``log10`` of a strictly-positive pool, matching ``ensemble.describe_log10``. A…, Populate one PlotItem with the box-whisker distribution for *key*. Single…, Compute box-whisker statistics for a 1-D array., Populate one subplot with the RMSE or R² distribution over the pool. The strip… (+11 more)

### Community 30 - "Bounds Panel Rows"
Cohesion: 0.09
Nodes (13): _BoundRow, BoundsPanel, _parse_sci(), Quantity, QWidget, Registry-driven parameter bounds editor. Auto-populates from…, Return custom Quantity bounds or None if all match defaults., Apply dye-alone–derived bounds to I0 and I_dye_free rows. (+5 more)

### Community 31 - "Image Export Tests"
Cohesion: 0.08
Nodes (23): QImage, annotated_plot_widget(), fitted_dist_widget(), fixture, Tests for the consolidated image-export pipeline. Verifies: * single-plot PNG…, The composite PNG's width matches the request exactly; height comes from the…, A 2x2 layout with 3 selected keys exports as a valid composite PNG., ``derive_height_in`` returns a height that preserves the live cell aspect. (+15 more)

### Community 32 - "Distribution Widget"
Cohesion: 0.09
Nodes (21): DistributionWidget, QWidget, Parameter keys of the currently displayed subplots, in order., Pixel size of a single live distribution subplot. This is the source of truth…, Height in inches that preserves live cell aspect for the chosen layout. With…, Emit representative_selected with the front-most clicked fit's index.…, Side-by-side box-whisker plots of fitted parameter distributions. One pyqtgraph…, pooled_result() (+13 more)

### Community 33 - "Tab-Separated TXT I/O"
Cohesion: 0.09
Nodes (18): DataFrame, Path, Check if a row is a header row., Writer for tab-separated fit results., Write fit results as tab-separated text. Parameters ---------- results : dict…, Reader for tab-separated measurement files with multi-replica support., Read tab-separated measurement file. Handles multi-replica files where each…, TxtReader (+10 more)

### Community 34 - "Update Download Worker"
Cohesion: 0.09
Nodes (16): DownloadWorker, Path, QThread, Background download worker — streams a URL to a local file with progress. Used…, Stream a URL to a local path, emitting progress in 64 KB chunks. Signals…, Request cooperative cancellation. Safe to call from the GUI thread. The chunk…, _os_label(), _pick_asset() (+8 more)

### Community 35 - "Parameter Label Formatting"
Cohesion: 0.11
Nodes (16): _canonical_unit(), fmt_param(), fmt_unit_html(), fmt_unit_pretty(), HTML display labels for parameter names. Used by FitSummaryWidget and plot…, Format a unit string as abbreviated HTML with negative exponents. Pint's ``~H``…, Format a unit string as abbreviated Unicode with negative exponents. Like…, Map the signal unit's display alias ``a.u.`` to its parseable token ``au``.… (+8 more)

### Community 36 - "Concentration Vector Helpers"
Cohesion: 0.12
Nodes (18): extract_concentrations_from_file(), ndarray, Path, Quantity, Helpers for saving, loading, and extracting concentration vectors.…, Save a concentration vector to a JSON file. Parameters ----------…, Read a concentration vector as a self-describing ``pint.Quantity``. Dispatches…, Load a data file via the I/O registry and extract its concentration grid. This… (+10 more)

### Community 37 - "Z-Score Replica Filter"
Cohesion: 0.11
Nodes (16): Mark replicas as inactive if any point is a z-score outlier. Uses **robust…, Apply modified z-score filtering to *ms* in-place., ZScoreReplicaFilter, _ms_few_replicas(), _ms_with_outlier(), Tests for the preprocessing pipeline and z-score replica filter., Running filter twice doesn't crash or double-drop., Identical replicas (std=0) → no drops, no errors. (+8 more)

### Community 38 - "Multi-File Replica Loading"
Cohesion: 0.11
Nodes (25): load_measurements_multi(), DataFrame, Load several measurement files and stack them as replicas. Each file is read…, DataPanel — file loading and concentration vector management., _jasco(), Tests for batch-loading several files as replicas…, Same file name in different folders → distinct replica IDs, no collision., A file that parses to a multi-channel frame is refused (single-curve only). (+17 more)

### Community 39 - "Plot Widget Layout"
Cohesion: 0.09
Nodes (17): PlotWidget, ndarray, PlotItem, QWidget, Move the annotation to the emptiest slot, unless the user moved it. Once…, Re-place the annotation once the view range or size is real., Qt widget that renders ``prepare_plot_data()`` output via PyQtGraph. Parameters…, The underlying pyqtgraph PlotItem — used to x-link a second plot to this one. (+9 more)

### Community 40 - "GDA Assay"
Cohesion: 0.14
Nodes (10): GDAAssay, Any, Return experimental conditions. Returns ------- Dict[str, Any] {'Ka_dye': ...,…, Guest Displacement Assay data container. Attributes ---------- x_data :…, Validate data and conditions., GDA assay accepts Quantity conditions with dimensional validation., TestGDAQuantityConditions, GDA constructor rejects invalid inputs. (+2 more)

### Community 41 - "Results Text and CSV Export"
Cohesion: 0.18
Nodes (25): export_results_csv(), export_results_txt(), _fixed_width_table(), Path, Export fit results as a human-readable text report. Parameters ----------…, Plain-text row label; log twins are named ``log10(<key>)``., Render *rows* as space-padded columns sized to their widest entry. The first…, Export fit results as a tidy CSV — one row per reported parameter. Complements… (+17 more)

### Community 42 - "Plot Style Widget"
Cohesion: 0.11
Nodes (18): PlotStyleWidget, QWidget, _qcolor_to_tuple(), Convert a QColor (or tuple) from ParameterTree to an (R, G, B, A) tuple., Style configuration panel backed by a PyQtGraph ParameterTree. Emits…, Set the x-axis display unit and emit ``style_changed``. The x-axis unit no…, Return a copy of the current style as a plain dict., Apply a style dict to the ParameterTree, updating all widgets. Parameters… (+10 more)

### Community 43 - "Synthetic Test Fixtures"
Cohesion: 0.09
Nodes (25): dba_clean(), dba_noisy(), dye_alone_clean(), gda_clean(), gda_noisy(), ida_clean(), ida_noisy(), _make_dba_data() (+17 more)

### Community 44 - "IDA Assay"
Cohesion: 0.12
Nodes (11): IDAAssay, Any, IDA (Indicator Displacement Assay) data container. In IDA, guest is titrated…, Return experimental conditions. Returns ------- Dict[str, Any] {'Ka_dye': ...,…, Indicator Displacement Assay data container. Attributes ---------- x_data :…, Validate data and conditions., IDA assay accepts Quantity conditions with dimensional validation., TestIDAQuantityConditions (+3 more)

### Community 45 - "Fit Metrics and Dense Curves"
Cohesion: 0.11
Nodes (18): calculate_fit_metrics(), ndarray, Calculate RMSE and R² for a fit. Parameters ---------- y_observed : np.ndarray…, _dense_fit_curve(), ndarray, Quantity, Evaluate the forward model on a dense grid spanning the data range. Same…, Wrap optimizer float params into named Quantity dict. (+10 more)

### Community 46 - "Scientific Axis Ticks"
Cohesion: 0.13
Nodes (20): _format_exponent_unicode(), Convert an integer exponent to Unicode superscript, e.g. 5 → '⁵', -3 → '⁻³'., AxisItem that formats tick labels with a shared exponent. When all tick values…, Update exponent, firing callback only on change. The callback guard (``exp !=…, ScientificAxisItem, _bottom_label(), _left_label(), Widget tests for PlotWidget — requires a QApplication. (+12 more)

### Community 47 - "Data Panel Tests"
Cohesion: 0.09
Nodes (16): loaded_panel(), _multi_channel_frame(), multi_channel_panel(), fixture, Tests for the inline DataPanel concentration controls., A DataPanel set up as ``load_file`` would leave it for a 2-channel file., A DataPanel populated with a tiny three-point dataset (face values in M)., A loaded concentration vector honours its declared unit; a unit outside the… (+8 more)

### Community 48 - "JASCO Reader"
Cohesion: 0.15
Nodes (11): JascoReader, JASCO Spectra Manager titration export reader. JASCO instruments (FP-8300…, Reader for JASCO Spectra Manager titration CSV exports., _minimal_jasco(), parametrize, Tests for the JASCO Spectra Manager CSV reader. Covers the parts of the format…, Real JASCO exports stack accessories — both must survive., Non-duplicate keys keep the plain-string value type. (+3 more)

### Community 49 - "Main Window Shell"
Cohesion: 0.15
Nodes (4): FittingMainWindow, QMainWindow, Block until any in-flight update check finishes before closing. The startup…, Main application window: tab management + toolbar + menu routing. All fitting…

### Community 50 - "Draggable Annotation Item"
Cohesion: 0.10
Nodes (14): _DraggableTextItem, Remember a user drag so later rebuilds stop re-placing the box., A ``TextItem`` that reports where the user dropped it. ``pg.TextItem`` has no…, Hand the drop position to *on_moved*, but only after a real move. A press whose…, QPointF, _draggable(), Auto-placement applies until the user moves the box; then it stays put., A click that moves nothing must leave auto-placement in charge. Reporting a… (+6 more)

### Community 51 - "Assay Config Panel"
Cohesion: 0.14
Nodes (10): AssayConfigPanel, Any, QWidget, Registry-driven assay type selector and dynamic conditions form. A two-level…, Return conditions dict (values in base units) including implicit DBA mode., Programmatically select an assay (demo loader, tests). Selects the matching…, Fill the subtype combo with the assays in ``category`` (index 0 selected).…, Adopt a newly selected assay: rebuild the form/info and announce it. (+2 more)

### Community 52 - "Replica Activation Panel"
Cohesion: 0.13
Nodes (10): _display_label(), ReplicaPanel — per-replica activation checkboxes., Rebuild the checkbox grid from current MeasurementSet state., Map a zero-based replica index to an A–Z display label. For indices 0–25…, Show one checkbox per replica and allow toggling active/inactive state. Auto-…, Populate checkboxes from MeasurementSet replica IDs., Re-read active states from the MeasurementSet (after preprocessing)., ReplicaPanel (+2 more)

### Community 53 - "Plot Annotation Tests"
Cohesion: 0.10
Nodes (22): _annotation_rect(), _binding_plot(), _ka_line(), parametrize, A shown PlotWidget with a saturating titration and one fit annotated. The curve…, The Ka parameter line of the annotation., Each parameter reads 'Estimate (min, max)' over the accepted pool., A result with no stored pool shows the estimate and says why, not a fake range. (+14 more)

### Community 54 - "DBA Forward Model"
Cohesion: 0.11
Nodes (14): dba_signal(), Compute DBA signal for host-dye equilibrium (H + D ⇌ HD). Delegates the…, Verify the forward models against equilibrium solutions computed by a different…, dba_signal (quadratic) agrees with a Brent solve of the dye balance., competitive_signal_point agrees with an fsolve of the full 3-species system…, Tests for the DBA (direct binding) forward model., HtoD with no host => all dye free; signal = I0 + I_dye_free * d0., Very high Ka_dye should drive binding to saturation. (+6 more)

### Community 55 - "Fitting Workflow Diagram"
Cohesion: 0.16
Nodes (20): Aggregate Good Fits, Any Good Fits? (decision), Compute Final Model, Compute RMSE and R-squared, Fitting Workflow Flowchart, Filter Fits using RMSE and R-squared, Generate Multiple Parameter Sets, Initial Guess Available? (decision) (+12 more)

### Community 56 - "Plot Image Export"
Cohesion: 0.15
Nodes (17): Dialog for saving the distributions plot as a composite PNG or SVG. The dialog…, Save the distributions composite as PNG or SVG. The export figure's aspect is…, export_plot_item(), export_scene(), _ext(), prepare_widget_for_offscreen_render(), Path, PlotItem (+9 more)

### Community 57 - "Plot Style Application"
Cohesion: 0.12
Nodes (14): line_style_to_qt(), Map style string to Qt.PenStyle. Accepts both internal strings (``"solid"``,…, _ErrorBarSample, Any, Rebuild the draggable fit-summary overlay, then re-place it., Legend sample that renders an error-bar glyph instead of a line. pyqtgraph's…, Clear and redraw from a ``prepare_plot_data()`` dict. Parameters ----------…, Mutate existing plot items in-place with new style settings. Wired to… (+6 more)

### Community 58 - "Data Panel Loading"
Cohesion: 0.13
Nodes (10): DataPanel, Load measurement data and edit the concentration vector inline. Signals -------…, Set the Display Unit combo without re-emitting if unchanged., Bring keyboard focus to the inline table. Public hook for the fit-time…, End-to-end load of the real EnSight fixture through ``load_file``., JASCO reader metadata must survive the load into MeasurementSet., Selecting several files in the import dialog stacks them as replicas., TestEnsightLoadIntegration (+2 more)

### Community 59 - "Ensemble Statistics Tests"
Cohesion: 0.11
Nodes (9): Unit tests for the ensemble-collapse module. Pins the single-source collapse…, On a fixed dataset the two criteria agree by construction., When R² ties (e.g. all 0 for constant y, ss_tot==0), pick lowest RMSE., All eight keys, against arithmetic done by hand rather than by numpy. [1, 2,…, One sample → no dispersion defined; report 0, never NaN., log₁₀ stats must come from log₁₀(pool). The centre commutes (median), but the…, TestCollapse, TestDescribe (+1 more)

### Community 60 - "I/O Reader Protocols"
Cohesion: 0.15
Nodes (14): MeasurementReader, Protocol, Protocol definitions for I/O readers and writers. This module defines the…, Protocol for reading measurement data files. Implementations must define: -…, Protocol for writing fit results. Implementations must define: - extensions:…, ResultWriter, CSV format reader for comma-separated measurement data. Supported formats…, TXT format reader and writer for tab-separated measurement data. File Format… (+6 more)

### Community 61 - "Distribution Subplot Toggles"
Cohesion: 0.12
Nodes (10): Redraw all subplots from a FitResult's parameter_samples. One subplot per…, Plain-text label for a subplot's toggle checkbox., Rebuild the checkbox row when the key-set changes; else reflect state., Show/hide one subplot without re-rendering (no jitter/stat recompute)., Pick the stacked page: placeholder, all-hidden notice, or the plots., Update visual style (fonts, palette) from PlotStyleWidget., Reset to empty placeholder., Remove all data items AND the legend from a PlotItem. (+2 more)

### Community 62 - "Reader Registry Dispatch"
Cohesion: 0.19
Nodes (8): get_reader(), Register a reader class for its supported extensions. Multiple readers may…, Get a reader instance for the given file path. Walks the registered candidates…, register_reader(), Tests for the content-sniffing reader registry. Covers the dispatch contract…, Default registrations (txt, csv, xlsx) survive the rewrite., TestRegistry, TestDispatch

### Community 63 - "Update Check Wiring"
Cohesion: 0.14
Nodes (6): Set the window title to ``SupraSimFit <version> [suffix]``. Called once at…, Spawn an :class:`UpdateCheckWorker`. Parameters ---------- silent : bool…, Clear the worker reference and schedule the QObject for deletion. Must run…, QThread, Query GitHub ``/releases/latest`` for the configured repo. Signals -------…, UpdateCheckWorker

### Community 64 - "Plot Color Constants"
Cohesion: 0.18
Nodes (10): Color constants and helpers for the plotting module., Return an RGBA tuple from an RGB tuple and optional alpha. Parameters…, rgba(), Box-whisker distribution plots for fitted parameters (pyqtgraph)., Style configuration widget using PyQtGraph ParameterTree., Main plot widget wrapping a PyQtGraph PlotWidget., Live speciation plot for the simulation applet. Shows the internal model…, Tests for gui.plotting.colors — no QApplication required. (+2 more)

### Community 65 - "BaseAssay Contract Methods"
Cohesion: 0.15
Nodes (9): ndarray, Quantity, Compute predicted signal from parameters. Parameters ---------- params :…, Equilibrium speciation across ``x_data`` from the same solve as the signal.…, Compute residuals (observed - predicted). Parameters ---------- params :…, Compute sum of squared residuals (SSR) for optimization. Parameters ----------…, Get default parameter bounds as a name-keyed dictionary. Returns the…, Convert parameter array to named dictionary. Parameters ---------- params :… (+1 more)

### Community 66 - "Plot Data Preparation"
Cohesion: 0.26
Nodes (8): prepare_plot_data(), Any, Gather plot-ready data from a MeasurementSet and optional fits. Parameters…, Tests for prepare_plot_data() — the MeasurementSet → plot dict bridge., _simple_fit_result(), _simple_ms(), TestPrepPlotDataFits, TestPrepPlotDataReplicas

### Community 67 - "Bounds Resolution Helpers"
Cohesion: 0.25
Nodes (8): Merge user overrides with registry defaults -> named bounds dict., Convert parameter names -> positional indices for the optimizer., _resolve_bounds(), _resolve_log_scale(), _gda_assay(), Tests for pipeline helper functions: _resolve_bounds, _resolve_log_scale., TestResolveBounds, TestResolveLogScale

### Community 68 - "GDA Forward Model"
Cohesion: 0.14
Nodes (10): ndarray, Quantity, GDA (Guest Displacement Assay) data container. In GDA, dye is titrated into a…, Free host, dye, guest and both complexes (M) across the dye titration., Compute predicted signal from parameters. Parameters ---------- params :…, gda_species(), Competitive speciation for GDA (dye titrated, guest fixed). Returns…, _noisy() (+2 more)

### Community 69 - "JASCO Data Parsing"
Cohesion: 0.19
Nodes (9): Any, DataFrame, ndarray, Path, Split file into (header dict, data lines, extended-info sections)., Parse the bracketed-INI extended-info block into nested dicts. Real JASCO…, Parse two-column numeric data and cross-check against ``NPOINTS``., Convert the x column to M using the unit token in ``XUNITS``. JASCO writes… (+1 more)

### Community 70 - "Species Speciation Plot"
Cohesion: 0.18
Nodes (9): _fmt(), QWidget, Compact concentration readout: 3 significant figures, no exponent clutter., A live, hover-readable equilibrium-speciation plot., SpeciesPlotWidget, The plot scales to the unit it is given, so it can match the signal plot's…, Dye-alone has no equilibrium, so the plot is blanked with a plain-language note., test_species_plot_honors_display_unit() (+1 more)

### Community 71 - "Numeric Input Widgets"
Cohesion: 0.16
Nodes (10): format_number(), QLineEdit, Human-readable float: plain for O(1) magnitudes, scientific for the rest. Uses…, Single-float entry shown in adaptive/scientific notation. Preferred over…, Set the shown value from code without emitting ``value_changed``., SciLineEdit, Large magnitudes read as scientific notation; O(1) values stay plain., A user can type '1e8' — the field parses and reformats it, no zero-counting. (+2 more)

### Community 72 - "Parameter Kind Units"
Cohesion: 0.17
Nodes (12): ParamKind, Semantic identity of a fitted parameter. Dimensional analysis alone cannot…, Enum, Registry unit-lint and the signal-dimension invariant. These guard the "both"…, Every registered parameter carries an explicit ParamKind., Each parameter's ParamKind is dimensionally consistent with its unit. With ``au…, A parameter is a BINDING_CONSTANT iff it is sampled in log space., ``au`` is not dimensionless, so ``au/M`` cannot be read as ``1/M``. (+4 more)

### Community 73 - "Session UI Grouping"
Cohesion: 0.22
Nodes (8): _Grouped(), _GroupedPlain, _GroupedWithInfo, QGroupBox, QWidget, Thin wrapper that places an existing widget inside a QGroupBox., Plain wrapper around :class:`InfoGroupBox` exposing ``self.widget``., Factory: return a plain or info-bearing group-box wrapper.

### Community 74 - "Simulation Window"
Cohesion: 0.19
Nodes (6): Open (or re-raise) the non-modal forward-simulation applet., QMainWindow, Interactive forward-simulation applet (non-modal)., SimulationWindow, Enabling noise puts N replicate scatter series alongside the model line., test_window_recompute_overlays_noisy_scatter()

### Community 75 - "Release Version Check"
Cohesion: 0.21
Nodes (8): is_newer(), Background check for newer SupraSimFit releases on GitHub. Queries the GitHub…, Return True if *remote_tag* represents a strictly newer version than *local*.…, parametrize, Tests for :func:`gui.update_check.is_newer`. Pure-function tests — no GitHub…, Behaviour of ``is_newer(remote_tag, local)``., Malformed remote tags must not raise — treat as 'not newer'., TestIsNewer

### Community 76 - "IDA Species Grid"
Cohesion: 0.17
Nodes (10): ndarray, Quantity, Free host, dye, guest and both complexes (M) across the guest titration., Compute predicted signal from parameters. Parameters ---------- params :…, _competitive_species_grid(), competitive_species_point(), ida_species(), Equilibrium speciation for one point of the competitive H/D/G system. Core… (+2 more)

### Community 77 - "Initial Guess Generation"
Cohesion: 0.26
Nodes (5): generate_initial_guesses(), ndarray, Generate random initial parameter guesses within bounds. Parameters ----------…, When lower bound is 0, log-scale falls back to linear (no log10(0) crash)., TestGenerateInitialGuesses

### Community 78 - "Session Export Actions"
Cohesion: 0.17
Nodes (6): Build a default save filename based on the loaded dataset's stem. Falls back to…, Export current fit results to JSON., Export current fit results as a human-readable text report., Export current fit results as a machine-readable CSV table., Save the current plot as PNG or SVG., Save the distributions plot as a composite PNG with a layout picker.

### Community 79 - "Axis Label Composition"
Cohesion: 0.20
Nodes (6): Update default axis names (and optionally the y-unit). The current overrides in…, Set an axis label, appending ×10ⁿ if *exp* is not None., Compose ``"<name> [<unit>]"`` using *override* if non-empty. The override is…, Update y-axis label reactively when the exponent changes during paint., Update x-axis label reactively when the exponent changes during paint., Set axis labels with exponent suffix if known. Exponents may be stale on the…

### Community 80 - "Speciation Contracts"
Cohesion: 0.32
Nodes (11): parametrize, Species-level contracts for the forward models. The simulation applet plots the…, _sig(), _sp(), test_dba_species_close_mass_balance_and_rebuild_signal(), test_dye_alone_species_is_the_titrant(), test_gda_species_close_mass_balance_and_rebuild_signal(), test_h2g_species_close_mass_balance_and_rebuild_signal() (+3 more)

### Community 81 - "Core and I/O Public API"
Cohesion: 0.20
Nodes (8): Core domain logic for molecular binding assay fitting. Main entry points: -…, Path, Minimal I/O module for measurement data and fit results. Public API ----------…, Save fit results to file. Parameters ---------- results : dict Fit results…, save_results(), TxtWriter correctly serializes fit results., save_results() public API works end-to-end., TestTxtWriter

### Community 82 - "Pint Unit Architecture"
Cohesion: 0.20
Nodes (11): F3 — silent uM fallback in plot x-unit scaling, F7 — custom bounds converted to canonical unit before .magnitude, Runtime unit validation at boundary crossings, Explicit .to(target) conversion, never to_base_units(), Float-only numeric core (benchmarked), No silent unit fallbacks (anti-pattern checklist), The canonical normalization boundary table, Self-describing I/O and the M unit contract (+3 more)

### Community 83 - "App Launch Entry Point"
Cohesion: 0.24
Nodes (8): _app_icon_path(), launch(), FittingMainWindow — thin shell managing tabbed fitting sessions., Block non-focused spinboxes from swallowing wheel events. Qt's event-filter…, Locate the bundled app icon for both source runs and PyInstaller bundles., Entry point — create the QApplication and launch the main window., _SpinBoxWheelRedirect, QObject

### Community 84 - "Assay Type Selector"
Cohesion: 0.29
Nodes (4): AssayTypeSelector, QWidget, Two dependent combos: assay *category* then *subtype*. Signals -------…, Programmatically select an assay, emitting ``assay_type_changed`` once.

### Community 85 - "Concentration Table Editing"
Cohesion: 0.20
Nodes (4): _fmt_cell(), Format a float for the concentration table — short, scientific when needed., Return the token in ``UNITS`` whose pint unit matches ``units``. Matched by…, Write the face-value buffer into the current MeasurementSet and announce it.

### Community 86 - "Fit Config Panel"
Cohesion: 0.25
Nodes (5): FitConfigPanel, QWidget, Editor for :class:`~core.pipeline.fit_pipeline.FitConfig` parameters. Signals…, Row: [enable checkbox] [factor spinbox] [info] for the optional trim., Wrap ``widget`` in an HBox together with a trailing info button. The spinbox…

### Community 87 - "Registry Test Doubles"
Cohesion: 0.25
Nodes (6): _Accepter, _Fallback, DataFrame, Path, No can_read → always-accepting fallback., _Rejecter

### Community 88 - "Species Label Formatting"
Cohesion: 0.20
Nodes (7): fmt_species(), Return a concentration label for a species, e.g. ``HG2`` → ``[HG₂]``.…, _m_to(), ndarray, Redraw one line per species (concentrations in M) versus the titrant, shown in…, Clear the plot and show *message* (e.g. dye-alone has no speciation)., Pint-derived M→display multiplier for a concentration unit (single source of…

### Community 89 - "Single-File Channel Loading"
Cohesion: 0.24
Nodes (4): Load a single file: parse → build the MeasurementSet → emit. No modal dialog…, Build a MeasurementSet from a single-channel frame, forwarding the…, Return the single-channel sub-frame for *channel*, attrs preserved., Rebuild the MeasurementSet for the newly selected channel in-memory. The…

### Community 90 - "BaseAssay Fail-Fast Tests"
Cohesion: 0.20
Nodes (6): Base assay properties and methods work correctly., Assay exposes correct parameter_keys from registry., params_to_dict maps array values to parameter names., Residuals = observed - predicted, hand-computed for a linear model. slope=1e8,…, SSR hand-computed: 0² + 10² + (-10)² = 200., TestBaseAssayContracts

### Community 91 - "Plot Style Templates"
Cohesion: 0.22
Nodes (7): Save current plot style settings to a JSON file., Load plot style settings from a JSON file., load_style_json(), Path, Save a style dict to a JSON file. Parameters ---------- style : dict As…, Load a style dict from a JSON file. Parameters ---------- path : str or Path…, save_style_json()

### Community 92 - "Simulation Settings I/O"
Cohesion: 0.28
Nodes (7): load_simulation_settings(), Any, Path, Save / load simulation applet settings as JSON. The settings dict (from…, Write *state* (from :meth:`SimulationPanel.state`) to a JSON file., Read a settings dict previously written by :func:`save_simulation_settings`., save_simulation_settings()

### Community 93 - "Sidebar Scroll Area"
Cohesion: 0.36
Nodes (4): Vertically-scrolling sidebar whose width follows its content. A plain…, _SidebarScrollArea, QScrollArea, QSize

### Community 94 - "Multi-File Load Dialog"
Cohesion: 0.25
Nodes (5): _build_file_filter(), Load one or more measurement files. From the dialog the user may select several…, Stack several files as replicas in one MeasurementSet. Replica labels come from…, Fill the channel combo from the loaded frame; disable if ≤1 channel., Build QFileDialog filter string from registered I/O readers.

### Community 95 - "Session Layout Guards"
Cohesion: 0.29
Nodes (7): _assert_embedded(), Regression guard: the plot widget must stay embedded after any import. A…, The sidebar must be freely widenable — no finite max-width cap. A previous…, The sidebar must advertise at least its content's width to the splitter. A…, test_plot_stays_embedded_through_ensight_load_and_switch(), test_sidebar_has_no_hard_maxwidth_cap(), test_sidebar_reserves_content_width_so_it_never_clips()

### Community 96 - "Stepwise GUI Labels"
Cohesion: 0.29
Nodes (4): parametrize, Each stepwise parameter renders a real HTML label, not the raw-key fallback., TestLabelsAndHelp, TestRegistration

### Community 97 - "Composite Plot Layout"
Cohesion: 0.29
Nodes (5): GraphicsLayoutWidget, Build a transient ``GraphicsLayoutWidget`` whose cells match the live GUI. Each…, Ensure we have exactly *n* PlotWidget instances with ScientificAxisItem., Wire *axis* to update *plot_item*'s y-label when its exponent changes., _wire_exponent_callback()

### Community 98 - "Info Button Group Box"
Cohesion: 0.33
Nodes (3): InfoGroupBox, QGroupBox, QGroupBox with an :class:`InfoButton` inline after the title text. The button…

### Community 99 - "Brent Solver Robustness"
Cohesion: 0.33
Nodes (5): parametrize, Verify the competitive Brent solver finds a physical root across several…, [H_free] must be finite and in (0, h0) regardless of Ka magnitudes., DBA quadratic must produce a non-negative free fixed-species root., TestBrentBracketRobustness

### Community 100 - "Dye-Alone Assay Model"
Cohesion: 0.33
Nodes (4): ndarray, Quantity, Compute predicted signal from parameters. Parameters ---------- params :…, Free dye equals the titrant — a linear calibration has no equilibrium. There is…

### Community 101 - "H2G Stepwise Assay"
Cohesion: 0.33
Nodes (4): ndarray, Quantity, Compute predicted signal from parameters. Parameters ---------- params :…, Free host, guest, HG and H2G complexes (M) across the guest titration.

### Community 102 - "HG2 Stepwise Assay"
Cohesion: 0.33
Nodes (4): ndarray, Quantity, Compute predicted signal from parameters. Parameters ---------- params :…, Free host, guest, HG and HG2 complexes (M) across the guest titration.

### Community 103 - "Reader Writer Protocol Methods"
Cohesion: 0.33
Nodes (4): DataFrame, Path, Read measurement data from file. Parameters ---------- path : Path Path to…, Write fit results to file. Parameters ---------- results : dict Fit results…

### Community 105 - "Background Fit Worker"
Cohesion: 0.33
Nodes (4): FitWorker, Any, QThread, Run :func:`fit_measurement_set` in a background thread. Prevents the GUI from…

### Community 106 - "Stepwise Assay Contracts"
Cohesion: 0.47
Nodes (3): parametrize, Fixed host concentration is required, positive, and unit-bearing., TestStepwiseAssayContracts

### Community 107 - "Toolbar Menu Buttons"
Cohesion: 0.50
Nodes (3): QToolButton, Create a toolbar button that pops a menu without the Qt auto-arrow. The default…, QMenu

### Community 111 - "GUI Test Fixtures"
Cohesion: 0.50
Nodes (3): fixture, qapp(), Shared GUI-test fixtures. A single session-scoped ``qapp`` fixture lives here…

### Community 116 - "Concentration Boundary Audit"
Cohesion: 0.67
Nodes (3): Concentration-input boundary — three silent 1e6-1e9 errors, Deliberate non-changes (no column-name unit guessing), F8 — self-describing TXT unit header

### Community 120 - "Registry Isolation Fixture"
Cohesion: 0.67
Nodes (3): isolated_registry(), fixture, Snapshot READERS and restore after the test.

## Knowledge Gaps
- **11 isolated node(s):** `uv`, `_ParamSpec`, `fitting-app`, `Replica outlier removal (modified Z-score)`, `Result export and re-import (JSON / TXT / raw / plot)` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1058 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MeasurementSet` connect `MeasurementSet Container` to `Fit Pipeline Entry Points`, `Assay Base and Registry`, `MeasurementSet Writing`, `Ensemble Collapse`, `Per-Replica Fitting`, `Dye-Alone Linear Calibration`, `MeasurementSet Tests`, `Forward Simulation Core`, `Preprocessing and Plot Prep`, `Fitting Session Wiring`, `Bounds Panel Rows`, `Z-Score Replica Filter`, `Multi-File Replica Loading`, `Data Panel Tests`, `Replica Activation Panel`, `Data Panel Loading`, `Plot Data Preparation`, `GDA Forward Model`, `Simulation Window`, `Core and I/O Public API`, `Single-File Channel Loading`, `Background Fit Worker`?**
  _High betweenness centrality (0.173) - this node is a cross-community bridge._
- **Why does `AssayType` connect `Simulation Knob Model` to `Assay Base and Registry`, `Ensemble Collapse`, `FitResult Serialization`, `Dye-Alone Linear Calibration`, `Forward Simulation Core`, `Fitting Session Wiring`, `Assay Condition Fields`, `DBA Assay`, `Distribution Box Drawing`, `Bounds Panel Rows`, `Distribution Widget`, `GDA Assay`, `IDA Assay`, `Assay Config Panel`, `Plot Color Constants`, `GDA Forward Model`, `Parameter Kind Units`, `Simulation Window`, `Assay Type Selector`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `FittingSession` connect `Fitting Session Wiring` to `Fit Pipeline Entry Points`, `Simulation Knob Model`, `Multi-Export Dialog`, `Flat Tab Bar`, `MeasurementSet Writing`, `Ensemble Collapse`, `MeasurementSet Container`, `Dye-Alone Linear Calibration`, `Distribution Export Dialog`, `Fit Summary Table`, `Preprocessing and Plot Prep`, `Bounds Panel Rows`, `Distribution Widget`, `Plot Widget Layout`, `Plot Style Widget`, `Main Window Shell`, `Assay Config Panel`, `Replica Activation Panel`, `Data Panel Loading`, `Session UI Grouping`, `Session Export Actions`, `App Launch Entry Point`, `Fit Config Panel`, `Plot Style Templates`, `Session Layout Guards`, `Background Fit Worker`, `Demo Fit Trigger`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Are the 30 inferred relationships involving `MeasurementSet` (e.g. with `BaseAssay` and `prepare_plot_data()`) actually correct?**
  _`MeasurementSet` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 35 inferred relationships involving `AssayType` (e.g. with `BaseAssay` and `DBAAssay`) actually correct?**
  _`AssayType` has 35 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `FittingSession` (e.g. with `AssayType` and `MeasurementSet`) actually correct?**
  _`FittingSession` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `FitConfig` (e.g. with `SessionState` and `FittingSession`) actually correct?**
  _`FitConfig` has 20 INFERRED edges - model-reasoned connections that need verification._