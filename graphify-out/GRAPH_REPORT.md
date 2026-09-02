# Graph Report - fitting_app  (2026-09-02)

## Corpus Check
- 145 files · ~117,755 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2676 nodes · 5388 edges · 150 communities (120 shown, 22 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 331 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `055d43c4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- units.py
- FitConfig
- Generate Multiple Parameter Sets
- EnsightReader
- FlatTabWidget
- test_preprocessing.py
- _sample_fit_result
- FitSummaryWidget
- SimulationWindow
- TitrantInput
- ida_signal
- test_plot_widget.py
- tests/conftest.py
- DataPanel
- test_scaling.py
- MeasurementSet
- SimulationPanel
- test_simulation_applet.py
- h2g_signal
- _UnitWidget
- FitResult
- test_optimizer.py
- CsvReader
- optimizer/__init__.py
- AssayConfigPanel
- TxtReader
- SaveDistributionsPlotDialog
- test_export.py
- read_raw_concentrations
- ._populate_subplot
- .update_plot
- test_data_panel.py
- load_measurements
- bounds_panel.py
- test_pipeline_e2e.py
- UpdateAvailableDialog
- distribution_widget.py
- io/registry.py
- DistributionWidget
- PlotStyleWidget
- SpeciesPlotWidget
- ParameterControl
- test_ensemble.py
- save_distributions_dialog.py
- ReplicaPanel
- DBAAssay
- JascoReader
- labels.py
- FittingMainWindow
- write_measurements_txt
- _binding_plot
- ExportMultipleDialog
- .from_dataframe
- generate_initial_guesses
- data_panel.py
- ._run_update_check
- _settings
- AssayType
- .get_conditions
- test_simulation.py
- is_newer
- TestConstruction
- .forward_model
- IDAAssay
- .forward_model
- .to_assay
- test_io_bmg.py
- dba_signal
- ._checkbox_label
- controls.py
- ._valid_kwargs
- .residuals
- prepare_plot_data
- test_pipeline_helpers.py
- FittingSession
- ZScoreReplicaFilter
- Molecular Binding Assay Fitting Toolkit
- session.py
- test_results_export.py
- SciLineEdit
- .get_conditions
- XlsxReader
- ScientificAxisItem
- fitting_session.py
- _sample_measurement_set
- Scientific Summary of the Molecular Binding Assay Fitting Toolkit
- linear_regression
- F2 (root cause): au defined as dimensionless and shadowing astronomical_unit
- ._refresh_plot
- main_window.py
- InfoGroupBox
- PreprocessingPanel
- Path
- Minimize Residuals
- ._default_save_name
- fit_measurement_set
- ParamKind
- Unit normalization boundary table
- multistart_minimize
- _DraggableTextItem
- UpdateCheckWorker
- _SidebarScrollArea
- test_fitting_session_layout.py
- ._best_overlay_slot
- TestCalculateFitMetrics
- ._emit_tab_title
- FitConfigPanel
- TestBrentBracketRobustness
- .forward_model
- ._load_single
- AssayTypeSelector
- .read
- parametrize
- test_assay_config_panel.py
- SimKnob
- SessionState
- ._make_menu_button
- BoundsPanel
- TestRepresentativeFitTrustworthy
- TestHG2MassBalance
- TestModelVsIndependentEquilibrium
- TestStepwiseAssayContracts
- ._on_representative_selected
- QHBoxLayout
- .load_demo_ida
- gui/conftest.py
- test_assay_categories.py
- ._decimals_for_scale
- .get_conditions
- PlotWidget
- graphify
- formats/__init__.py
- .get_conditions
- .get_conditions
- .get_conditions
- simulation/__init__.py
- Pint contexts deliberately unused
- fitting-app
- .get_conditions
- .current_conditions
- .value
- Pint ~P/~H display formatting

## God Nodes (most connected - your core abstractions)
1. `MeasurementSet` - 100 edges
2. `AssayType` - 83 edges
3. `FittingSession` - 73 edges
4. `FitConfig` - 68 edges
5. `FitResult` - 65 edges
6. `BaseAssay` - 60 edges
7. `PlotWidget` - 51 edges
8. `DistributionWidget` - 50 edges
9. `IDAAssay` - 44 edges
10. `GDAAssay` - 43 edges

## Surprising Connections (you probably didn't know these)
- `Aggregate Good Fits` --semantically_similar_to--> `ZScoreReplicaFilter`  [INFERRED] [semantically similar]
  docs/fitting_workflow.jpg → core/data_processing/preprocessing.py
- `generate_initial_guesses()` --implements--> `Initial Guess vs. Random Initialization`  [INFERRED]
  core/optimizer/multistart.py → docs/fitting_workflow.jpg
- `H1: extract_concentrations_from_file ignored the declared unit` --references--> `extract_concentrations_from_file()`  [EXTRACTED]
  docs/pint-audit.md → core/data_processing/concentration.py
- `prepare_plot_data()` --implements--> `Plot Experimental vs. Simulated`  [INFERRED]
  core/data_processing/plotting.py → docs/fitting_workflow.jpg
- `load_measurements()` --implements--> `Load Data`  [INFERRED]
  core/io/__init__.py → docs/fitting_workflow.jpg

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **au = [signal] remediation surface** — core_units, core_optimizer_scaling_scale_factor, core_assays_registry_paramkind, core_assays_base_baseassay_residuals, docs_pint_audit_f2_au_dimensionless_root_cause, docs_pint_findings_au_signal_dimension [EXTRACTED 1.00]
- **Unit normalization boundary crossings (Quantity in, float out, Quantity back)** — gui_widgets_assay_config_panel_unitwidget, core_data_processing_measurement_set_measurementset_from_dataframe, core_assays_base_baseassay_post_init, core_pipeline_fit_pipeline_fit_assay, core_pipeline_fit_pipeline_wrap_params_as_quantities, core_pipeline_fit_pipeline_fitresult_to_dict, docs_pint_findings_unit_normalization_boundaries [EXTRACTED 1.00]
- **Multi-start trial loop (generate, optimize, score, filter, repeat)** — docs_fitting_workflow_generate_multiple_parameter_sets, docs_fitting_workflow_optimize_parameters, docs_fitting_workflow_minimize_residuals, docs_fitting_workflow_compute_rmse_r2, docs_fitting_workflow_filter_fits_using_rmse_r2, docs_fitting_workflow_more_trials [INFERRED 0.85]
- **Per-replica outcome branch (aggregate, model, plot, or skip)** — docs_fitting_workflow_any_good_fits, docs_fitting_workflow_aggregate_good_fits, docs_fitting_workflow_compute_final_model, docs_fitting_workflow_plot_experimental_vs_simulated, docs_fitting_workflow_skip_replica_no_valid_fits, docs_fitting_workflow_more_replicas [INFERRED 0.85]

## Communities (150 total, 22 thin omitted)

### Community 0 - "units.py"
Cohesion: 0.05
Nodes (51): Base class for assay data containers. This module defines the abstract base…, create_dba_dye_to_host(), create_dba_host_to_dye(), ndarray, DBA (Direct Binding Assay) data containers. DBA measures direct binding between…, Create DBA assay for host-to-dye titration. Parameters ---------- host_conc :…, Create DBA assay for dye-to-host titration. Parameters ---------- dye_conc :…, Dye-alone calibration assay. This is a simple linear calibration where signal… (+43 more)

### Community 1 - "FitConfig"
Cohesion: 0.05
Nodes (56): _config_to_dict(), fit_assay(), fit_measurement_set_per_replica(), FitConfig, PerReplicaFitError, Any, Configuration for the fitting pipeline. ``rmse_threshold_factor`` is an…, Serialize a FitConfig to a plain dict. (+48 more)

### Community 2 - "Generate Multiple Parameter Sets"
Cohesion: 0.17
Nodes (16): Any Good Fits? (decision), Compute Final Model, Fitting Workflow Flowchart, Generate Multiple Parameter Sets, Initial Guess Available? (decision), Initial Guess vs. Random Initialization, Initialize Randomly, Load Data (+8 more)

### Community 3 - "EnsightReader"
Cohesion: 0.07
Nodes (25): EnsightReader, DataFrame, ndarray, Path, PerkinElmer EnSight plate-reader CSV export parser. EnSight (Kaleido 3.x)…, Capture the few lines above the first 'Result for ...' block., Return list of (block name, line index) for every 'Result for' line., Locate and parse the plate grid that follows a 'Result for' line. (+17 more)

### Community 4 - "FlatTabWidget"
Cohesion: 0.07
Nodes (29): ButtonPosition, _FlatCloseButton, FlatTabBar, FlatTabWidget, QLineEdit, QToolButton, Flat / minimal tab widgets — one reusable style for every tab in the app.…, Tab bar that keeps an optional inline "+" button after the last tab. (+21 more)

### Community 5 - "test_preprocessing.py"
Cohesion: 0.13
Nodes (19): Data processing: containers, preprocessing, and plot helpers. Public API…, apply_preprocessing(), get_step(), PreprocessingStep, Any, Protocol, Minimalist preprocessing pipeline for MeasurementSet. Provides a tiny registry…, Apply a sequence of preprocessing steps to *ms* in order. Each entry in *steps*… (+11 more)

### Community 6 - "_sample_fit_result"
Cohesion: 0.05
Nodes (29): Convert to a JSON-safe dictionary. ``Quantity`` fields are converted to…, Reconstruct a ``FitResult`` from a dictionary. Parameters ---------- d : dict…, F3: silent micromolar fallback for an unknown x-axis unit, F5: missing unit silently re-wrapped as dimensionless, F6: condition units dropped on export, Pint anti-pattern checklist, Canonical internal units: M, 1/M, au, Convert with an explicit target, never to_base_units() (+21 more)

### Community 7 - "FitSummaryWidget"
Cohesion: 0.08
Nodes (27): FitSummaryWidget, QWidget, Read-only display of ``FitResult`` statistics. Layout ------ -…, Populate the widget from a ``FitResult``. ``Estimate`` is the representative…, Pint HTML value with the trailing unit stripped (units have a column)., Merged 'central ± spread' cell (units stripped; shown in the Units column)., Merged '[min, max]' range cell., Offer Best, Median, Worst, and the sticky Selected representative. Best/Worst… (+19 more)

### Community 8 - "SimulationWindow"
Cohesion: 0.12
Nodes (13): load_simulation_settings(), Any, Path, Save / load simulation applet settings as JSON. The settings dict (from…, Write *state* (from :meth:`SimulationPanel.state`) to a JSON file., Read a settings dict previously written by :func:`save_simulation_settings`., save_simulation_settings(), ndarray (+5 more)

### Community 9 - "TitrantInput"
Cohesion: 0.11
Nodes (15): _display_factor(), _parse_floats(), Titrant setup: a maximum concentration (0 → max in N points), or an explicit…, Switch to a custom vector from Molar values (used on data import)., Pint-derived scale: magnitude of ``1 <label>`` expressed in ``base`` units.…, Parse a comma/space-separated list of floats; raise ValueError if any token is…, TitrantInput, With no custom vector, the titration is a 0 → max linear scan of N_POINTS… (+7 more)

### Community 10 - "ida_signal"
Cohesion: 0.06
Nodes (33): competitive_signal_point(), gda_signal(), ida_signal(), Compute signal for a single point in competitive binding equilibrium. Delegates…, Compute GDA signal across dye concentration range. In GDA (Guest Displacement…, Compute IDA signal across guest concentration range. In IDA (Indicator…, Forward models for molecular binding assays., linear_signal() (+25 more)

### Community 11 - "test_plot_widget.py"
Cohesion: 0.11
Nodes (24): _format_exponent_unicode(), Convert an integer exponent to Unicode superscript, e.g. 5 → '⁵', -3 → '⁻³'., _bottom_label(), _draggable(), _left_label(), Widget tests for PlotWidget — requires a QApplication., A click that moves nothing must leave auto-placement in charge. Reporting a…, Press → move → release does report, so a drag wins over auto-placement. (+16 more)

### Community 12 - "tests/conftest.py"
Cohesion: 0.06
Nodes (42): H2GAssay, Stepwise 2:1 host–guest direct-binding assay (two hosts bind one guest).…, Validate data and the fixed host concentration., dba_clean(), dba_noisy(), dye_alone_clean(), gda_clean(), gda_noisy() (+34 more)

### Community 13 - "DataPanel"
Cohesion: 0.09
Nodes (13): DataPanel, _fmt_cell(), Format a float for the concentration table — short, scientific when needed., Load measurement data and edit the concentration vector inline. Signals -------…, Set the Display Unit combo without re-emitting if unchanged., Bring keyboard focus to the inline table. Public hook for the fit-time…, Write the face-value buffer into the current MeasurementSet and announce it., End-to-end load of the real EnSight fixture through ``load_file``. (+5 more)

### Community 14 - "test_scaling.py"
Cohesion: 0.13
Nodes (23): ParamScaler, Data-driven parameter rescaling for well-conditioned fitting. Given an assay…, Rescale raw (lower, upper) bound tuples to tilded space., Compute the affine rescaling factor for a parameter unit. Each component of…, Per-parameter affine rescaler derived from assay data. Attributes ----------…, Build a scaler from a ``BaseAssay`` instance. Raises ------ ValueError If the…, _scale_factor(), ida_assay() (+15 more)

### Community 15 - "MeasurementSet"
Cohesion: 0.08
Nodes (13): MeasurementSet, Quantity, Total number of replicas (active + dropped)., Number of currently active replicas., Number of concentration points per replica., Replica IDs currently marked active., Replica IDs currently marked inactive., Mark all replicas active (undo all filtering). (+5 more)

### Community 16 - "SimulationPanel"
Cohesion: 0.11
Nodes (15): QWidget, Switch the titration input to an explicit vector (Molar). Used on data import., Full serializable settings for save., Live control stack: model/condition knobs + titration vector + noise., Rebuild the knob controls for *assay_type* (registry-driven)., SimulationPanel, The flat list is gone: knobs render inside titled section group boxes., The titrant input renders inside the Concentrations section, not a separate box. (+7 more)

### Community 17 - "test_simulation_applet.py"
Cohesion: 0.11
Nodes (23): knobs_for(), Default slider range for a *condition* knob, derived from its default.…, Return the full live-knob list for *assay_type*: conditions ∪ parameters., slider_bounds(), _knob(), GUI tests for the forward-simulation applet. These pin the applet-specific…, Enabling noise puts N replicate scatter series alongside the model line., Every knob lands in the section its physical role dictates, for all assays. (+15 more)

### Community 18 - "h2g_signal"
Cohesion: 0.10
Nodes (17): ndarray, Quantity, Compute predicted signal from parameters. Parameters ---------- params :…, Free host, guest, HG and H2G complexes (M) across the guest titration., h2g_signal(), h2g_species(), Equilibrium speciation for stepwise 2:1 host–guest binding. Two hosts bind one…, Compute the signal for stepwise 2:1 host–guest binding (HG, H₂G). Guest is… (+9 more)

### Community 19 - "_UnitWidget"
Cohesion: 0.17
Nodes (9): ConditionField, Schema for one assay condition input field. Attributes ---------- key : str…, QWidget, Spinbox paired with an optional unit selector. For fields with display choices…, Set widget from a base-unit value., _UnitWidget, parametrize, The primary user-input boundary: _UnitWidget always returns a base-unit… (+1 more)

### Community 20 - "FitResult"
Cohesion: 0.06
Nodes (41): ABC, BaseAssay, Abstract base class for assay data containers. Subclasses must implement: -…, Parameter names for this assay type., Number of parameters to fit., Number of data points., In-memory container for multi-replica measurement data. MeasurementSet holds a…, Plot data preparation helpers. Prepares structured data from a… (+33 more)

### Community 21 - "test_optimizer.py"
Cohesion: 0.12
Nodes (20): filter_by_r_squared(), filter_by_rmse(), Filtering utilities for fit results. Filter multi-start fit attempts down to…, Filter fit attempts by RMSE threshold. Parameters ---------- results :…, Filter fit attempts by minimum R² value. Parameters ---------- results :…, Select the valid-fit pool kept for aggregation. The absolute R² floor is the…, select_valid_fits(), FitAttempt (+12 more)

### Community 22 - "CsvReader"
Cohesion: 0.11
Nodes (16): CsvReader, DataFrame, Path, True if every column name parses as a number (suggesting no header row)., Reader for CSV measurement files., Read a CSV measurement file. Parameters ---------- path : Path Path to input…, Series, CsvReader handles varied CSV dialects and headers. (+8 more)

### Community 23 - "optimizer/__init__.py"
Cohesion: 0.16
Nodes (21): collapse(), describe(), describe_log10(), EnsembleResult, _mad(), _mean(), _median(), ndarray (+13 more)

### Community 24 - "AssayConfigPanel"
Cohesion: 0.24
Nodes (5): AssayConfigPanel, Registry-driven assay type selector and dynamic conditions form. A two-level…, Programmatically select an assay (demo loader, tests). Selects the matching…, Fill the subtype combo with the assays in ``category`` (index 0 selected).…, Adopt a newly selected assay: rebuild the form/info and announce it.

### Community 25 - "TxtReader"
Cohesion: 0.07
Nodes (24): Validate data after initialization., DataFrame, Path, Check if a row is a header row., Writer for tab-separated fit results., Write fit results as tab-separated text. Parameters ---------- results : dict…, Reader for tab-separated measurement files with multi-replica support., Read tab-separated measurement file. Handles multi-replica files where each… (+16 more)

### Community 26 - "SaveDistributionsPlotDialog"
Cohesion: 0.14
Nodes (10): DistributionsExportConfig, QDialog, QWidget, Return ``width_in`` for a preset, or ``None`` for 'custom'. Height is no longer…, DPI input is meaningless for vector SVG; grey it out., Return value when the dialog is accepted. ``height_in`` is derived from…, Pick subplots, layout, dimensions, and format for the distributions image.…, SaveDistributionsPlotDialog (+2 more)

### Community 27 - "test_export.py"
Cohesion: 0.08
Nodes (23): QImage, annotated_plot_widget(), fitted_dist_widget(), fixture, Tests for the consolidated image-export pipeline. Verifies: * single-plot PNG…, The composite PNG's width matches the request exactly; height comes from the…, A 2x2 layout with 3 selected keys exports as a valid composite PNG., ``derive_height_in`` returns a height that preserves the live cell aspect. (+15 more)

### Community 28 - "read_raw_concentrations"
Cohesion: 0.11
Nodes (19): extract_concentrations_from_file(), ndarray, Path, Quantity, Helpers for saving, loading, and extracting concentration vectors.…, Save a concentration vector to a JSON file. Parameters ----------…, Read a concentration vector as a self-describing ``pint.Quantity``. Dispatches…, Load a data file via the I/O registry and extract its concentration grid. This… (+11 more)

### Community 29 - "._populate_subplot"
Cohesion: 0.11
Nodes (19): _box_stats(), _log10_positive(), ndarray, PlotItem, ``log10`` of a strictly-positive pool, matching ``ensemble.describe_log10``. A…, Populate one PlotItem with the box-whisker distribution for *key*. Single…, Compute box-whisker statistics for a 1-D array., Populate one subplot with the RMSE or R² distribution over the pool. The strip… (+11 more)

### Community 30 - ".update_plot"
Cohesion: 0.11
Nodes (14): line_style_to_qt(), Map style string to Qt.PenStyle. Accepts both internal strings (``"solid"``,…, Any, Rebuild the draggable fit-summary overlay, then re-place it., Clear and redraw from a ``prepare_plot_data()`` dict. Parameters ----------…, Mutate existing plot items in-place with new style settings. Wired to…, Update default axis names (and optionally the y-unit). The current overrides in…, Store FitResult objects used to populate the annotation. Call this after… (+6 more)

### Community 31 - "test_data_panel.py"
Cohesion: 0.07
Nodes (18): loaded_panel(), _multi_channel_frame(), multi_channel_panel(), fixture, Tests for the inline DataPanel concentration controls., A DataPanel set up as ``load_file`` would leave it for a 2-channel file., The Channel combo is enabled only for multi-channel data., A DataPanel populated with a tiny three-point dataset (face values in M). (+10 more)

### Community 32 - "load_measurements"
Cohesion: 0.07
Nodes (37): load_measurements(), load_measurements_multi(), DataFrame, Path, Minimal I/O module for measurement data and fit results. Public API ----------…, Save fit results to file. Parameters ---------- results : dict Fit results…, Load measurement data from file. Unit contract ------------- The returned…, Load several measurement files and stack them as replicas. Each file is read… (+29 more)

### Community 33 - "bounds_panel.py"
Cohesion: 0.21
Nodes (5): Scientist-facing descriptions of fitted parameters and their bounds. Each entry…, _BoundRow, _parse_sci(), BoundsPanel — registry-driven parameter bounds editor with dye-alone priors., A single row: lower QLineEdit | '—' | upper QLineEdit | info ⓘ.

### Community 34 - "test_pipeline_e2e.py"
Cohesion: 0.07
Nodes (27): DyeAloneAssay, Dye-alone calibration data container. Attributes ---------- x_data : np.ndarray…, fit_linear_assay(), Fit a dye-alone assay using simple linear regression. Parameters ----------…, assert_within_tolerance(), Assert fitted value is within tolerance of true value. Parameters ----------…, _dye_alone_assay(), _gda_assay() (+19 more)

### Community 35 - "UpdateAvailableDialog"
Cohesion: 0.10
Nodes (15): DownloadWorker, Path, QThread, Background download worker — streams a URL to a local file with progress. Used…, Stream a URL to a local path, emitting progress in 64 KB chunks. Signals…, Request cooperative cancellation. Safe to call from the GUI thread. The chunk…, _os_label(), _pick_asset() (+7 more)

### Community 36 - "distribution_widget.py"
Cohesion: 0.18
Nodes (10): Color constants and helpers for the plotting module., Return an RGBA tuple from an RGB tuple and optional alpha. Parameters…, rgba(), Box-whisker distribution plots for fitted parameters (pyqtgraph)., Style configuration widget using PyQtGraph ParameterTree., Main plot widget wrapping a PyQtGraph PlotWidget., Live speciation plot for the simulation applet. Shows the internal model…, Tests for gui.plotting.colors — no QApplication required. (+2 more)

### Community 37 - "io/registry.py"
Cohesion: 0.09
Nodes (24): MeasurementReader, Protocol, Protocol definitions for I/O readers and writers. This module defines the…, Protocol for reading measurement data files. Implementations must define: -…, Protocol for writing fit results. Implementations must define: - extensions:…, ResultWriter, CSV format reader for comma-separated measurement data. Supported formats…, TXT format reader and writer for tab-separated measurement data. File Format… (+16 more)

### Community 38 - "DistributionWidget"
Cohesion: 0.06
Nodes (28): DistributionWidget, QWidget, Redraw all subplots from a FitResult's parameter_samples. One subplot per…, Rebuild the checkbox row when the key-set changes; else reflect state., Show/hide one subplot without re-rendering (no jitter/stat recompute)., Pick the stacked page: placeholder, all-hidden notice, or the plots., Update visual style (fonts, palette) from PlotStyleWidget., Reset to empty placeholder. (+20 more)

### Community 39 - "PlotStyleWidget"
Cohesion: 0.11
Nodes (18): PlotStyleWidget, QWidget, _qcolor_to_tuple(), Convert a QColor (or tuple) from ParameterTree to an (R, G, B, A) tuple., Style configuration panel backed by a PyQtGraph ParameterTree. Emits…, Set the x-axis display unit and emit ``style_changed``. The x-axis unit no…, Return a copy of the current style as a plain dict., Apply a style dict to the ParameterTree, updating all widgets. Parameters… (+10 more)

### Community 40 - "SpeciesPlotWidget"
Cohesion: 0.10
Nodes (18): fmt_species(), Return a concentration label for a species, e.g. ``HG2`` → ``[HG₂]``.…, _fmt(), _m_to(), ndarray, PlotItem, QWidget, Redraw one line per species (concentrations in M) versus the titrant, shown in… (+10 more)

### Community 41 - "ParameterControl"
Cohesion: 0.17
Nodes (6): _clamp(), ParameterControl, Current value in the base unit (M / M⁻¹ / au / au·M⁻¹)., Current value as a pint Quantity (for assay conditions)., Serializable knob state for settings save., A single live knob: slider + labelled editable bounds + exact value. Internals…

### Community 42 - "test_ensemble.py"
Cohesion: 0.11
Nodes (9): Unit tests for the ensemble-collapse module. Pins the single-source collapse…, On a fixed dataset the two criteria agree by construction., When R² ties (e.g. all 0 for constant y, ss_tot==0), pick lowest RMSE., All eight keys, against arithmetic done by hand rather than by numpy. [1, 2,…, One sample → no dispersion defined; report 0, never NaN., log₁₀ stats must come from log₁₀(pool). The centre commutes (median), but the…, TestCollapse, TestDescribe (+1 more)

### Community 43 - "save_distributions_dialog.py"
Cohesion: 0.15
Nodes (17): Dialog for saving the distributions plot as a composite PNG or SVG. The dialog…, Save the distributions composite as PNG or SVG. The export figure's aspect is…, export_plot_item(), export_scene(), _ext(), prepare_widget_for_offscreen_render(), Path, PlotItem (+9 more)

### Community 44 - "ReplicaPanel"
Cohesion: 0.13
Nodes (10): _display_label(), ReplicaPanel — per-replica activation checkboxes., Rebuild the checkbox grid from current MeasurementSet state., Map a zero-based replica index to an A–Z display label. For indices 0–25…, Show one checkbox per replica and allow toggling active/inactive state. Auto-…, Populate checkboxes from MeasurementSet replica IDs., Re-read active states from the MeasurementSet (after preprocessing)., ReplicaPanel (+2 more)

### Community 45 - "DBAAssay"
Cohesion: 0.19
Nodes (8): DBAAssay, Free host, free dye and host-dye complex (M) across the titration., Direct Binding Assay data container. This class handles both Host→Dye and…, Validate data and set assay type based on mode., Ka_dye recovered in HtoD mode (host titrated, dye fixed)., TestDBAHtoD, DBA assay accepts Quantity fixed_conc with dimensional validation., TestDBAQuantityConditions

### Community 46 - "JascoReader"
Cohesion: 0.08
Nodes (23): JascoReader, Any, DataFrame, ndarray, Path, JASCO Spectra Manager titration export reader. JASCO instruments (FP-8300…, Split file into (header dict, data lines, extended-info sections)., Parse the bracketed-INI extended-info block into nested dicts. Real JASCO… (+15 more)

### Community 47 - "labels.py"
Cohesion: 0.12
Nodes (15): _canonical_unit(), fmt_param(), fmt_unit_html(), fmt_unit_pretty(), HTML display labels for parameter names. Used by FitSummaryWidget and plot…, Format a unit string as abbreviated HTML with negative exponents. Pint's ``~H``…, Format a unit string as abbreviated Unicode with negative exponents. Like…, Map the signal unit's display alias ``a.u.`` to its parseable token ``au``.… (+7 more)

### Community 48 - "FittingMainWindow"
Cohesion: 0.12
Nodes (5): FittingMainWindow, QMainWindow, Open (or re-raise) the non-modal forward-simulation applet., Block until any in-flight update check finishes before closing. The startup…, Main application window: tab management + toolbar + menu routing. All fitting…

### Community 49 - "write_measurements_txt"
Cohesion: 0.17
Nodes (13): _header_comment(), _iter_points(), Path, Writers that serialise a :class:`MeasurementSet` back to disk. These produce…, Yield ``(replica_idx, concentration, signal)`` tuples for every point. Includes…, Write *ms* to a tab-separated ``.txt`` file with repeated headers. Output…, Write *ms* to a long-format ``.csv`` file. Output round-trips through…, write_measurements_csv() (+5 more)

### Community 50 - "_binding_plot"
Cohesion: 0.10
Nodes (22): _annotation_rect(), _binding_plot(), _ka_line(), parametrize, A shown PlotWidget with a saturating titration and one fit annotated. The curve…, The Ka parameter line of the annotation., Each parameter reads 'Estimate (min, max)' over the accepted pool., A result with no stored pool shows the estimate and says why, not a fake range. (+14 more)

### Community 51 - "ExportMultipleDialog"
Cohesion: 0.15
Nodes (7): ExportMultipleDialog, Exception, Path, QDialog, QWidget, Pick artefacts, folder, and filename base; export in one click., Show the consolidated multi-artefact export dialog.

### Community 52 - ".from_dataframe"
Cohesion: 0.13
Nodes (11): DataFrame, Build a MeasurementSet from a long-form DataFrame. The DataFrame must contain…, H1: extract_concentrations_from_file ignored the declared unit, H2: pd.concat dropped per-file unit attrs in batch import, Multi-agent re-audit after merging main, Load BMG, replace placeholder concentrations, export, reload., TestBMGRoundTripAfterConcentrationFix, A declared concentration unit survives from file to a molar MeasurementSet. (+3 more)

### Community 53 - "generate_initial_guesses"
Cohesion: 0.29
Nodes (4): generate_initial_guesses(), Generate random initial parameter guesses within bounds. Parameters ----------…, When lower bound is 0, log-scale falls back to linear (no log10(0) crash)., TestGenerateInitialGuesses

### Community 54 - "data_panel.py"
Cohesion: 0.13
Nodes (11): format_channel_label(), Any, Build a human label for an EnSight channel: name + Ex/Em hints. Used by the GUI…, _build_data_help_html(), _build_file_filter(), DataPanel — file loading and concentration vector management., Load one or more measurement files. From the dialog the user may select several…, Stack several files as replicas in one MeasurementSet. Replica labels come from… (+3 more)

### Community 55 - "._run_update_check"
Cohesion: 0.29
Nodes (3): Set the window title to ``SupraSimFit <version> [suffix]``. Called once at…, Spawn an :class:`UpdateCheckWorker`. Parameters ---------- silent : bool…, Clear the worker reference and schedule the QObject for deletion. Must run…

### Community 56 - "_settings"
Cohesion: 0.17
Nodes (16): get_bool(), Thin wrapper around :class:`QSettings` for persistent user preferences. The…, Return a QSettings bound to the currently running application. Uses…, Return a boolean preference, falling back to *default*., Persist a boolean preference., set_bool(), _settings(), QSettings (+8 more)

### Community 57 - "AssayType"
Cohesion: 0.06
Nodes (37): Get metadata from the assay registry., HG2Assay, ndarray, Quantity, Stepwise 1:2 host–guest direct-binding assay (host binds two guests).…, Validate data and the fixed host concentration., Compute predicted signal from parameters. Parameters ---------- params :…, Free host, guest, HG and HG2 complexes (M) across the guest titration. (+29 more)

### Community 59 - "test_simulation.py"
Cohesion: 0.09
Nodes (39): _as_count(), _build_assay(), build_concentration_vector(), Any, ndarray, Forward simulation of titration data. This is the inverse of the fitting…, Construct an assay for forward evaluation over *x_vector* (M). Mirrors…, Evaluate an assay's forward model at *x_vector* from explicit parameters. Uses… (+31 more)

### Community 60 - "is_newer"
Cohesion: 0.15
Nodes (11): Tag-driven release trigger, Version drift guard, is_newer(), Background check for newer SupraSimFit releases on GitHub. Queries the GitHub…, Return True if *remote_tag* represents a strictly newer version than *local*.…, parametrize, Tests for :func:`gui.update_check.is_newer`. Pure-function tests — no GitHub…, Behaviour of ``is_newer(remote_tag, local)``. (+3 more)

### Community 61 - "TestConstruction"
Cohesion: 0.12
Nodes (9): MeasurementSet construction and validation., Happy-path construction from a long-form DataFrame., Concentrations are sorted ascending after construction., Missing required column raises ValueError., Replicas with different concentration grids raise ValueError., concentrations must be 1-D., signals columns must match concentrations length., replica_ids length must match signals rows. (+1 more)

### Community 62 - ".forward_model"
Cohesion: 0.33
Nodes (4): ndarray, Quantity, Compute predicted signal from parameters. Parameters ---------- params :…, Free dye equals the titrant — a linear calibration has no equilibrium. There is…

### Community 63 - "IDAAssay"
Cohesion: 0.19
Nodes (9): IDAAssay, Indicator Displacement Assay data container. Attributes ---------- x_data :…, Validate data and conditions., IDA assay accepts Quantity conditions with dimensional validation., TestIDAQuantityConditions, _bounds_with_i0_window(), Override the I0 bounds to a finite window around the non-zero ground truth., IDA recovers Ka_guest to within 10% even with a non-zero I0 baseline. (+1 more)

### Community 64 - ".forward_model"
Cohesion: 0.33
Nodes (4): ndarray, Quantity, Free host, dye, guest and both complexes (M) across the guest titration., Compute predicted signal from parameters. Parameters ---------- params :…

### Community 65 - ".to_assay"
Cohesion: 0.14
Nodes (9): Any, ndarray, Return the row index for *replica_id*, or raise ValueError., Set the active state of a single replica. Parameters ---------- replica_id :…, Check whether *replica_id* is currently active., Iterate over replicas as ``(replica_id, signal_view)`` pairs. Parameters…, Return the signal array (read-only view) for a single replica. Parameters…, Compute the mean signal across replicas. Parameters ---------- active_only :… (+1 more)

### Community 66 - "test_io_bmg.py"
Cohesion: 0.10
Nodes (27): _extract_metadata(), _find_bmg_sheet(), _find_header_row(), is_bmg_workbook(), parse_bmg_workbook(), Any, DataFrame, BMG plate reader export parser. BMG CLARIOstar / FLUOstar / PHERAstar… (+19 more)

### Community 67 - "dba_signal"
Cohesion: 0.11
Nodes (15): Quantity, Compute predicted signal from parameters. Parameters ---------- params :…, dba_signal(), dba_species(), ndarray, Compute DBA signal for host-dye equilibrium (H + D ⇌ HD). Delegates the…, Equilibrium speciation for 1:1 host–dye binding (H + D ⇌ HD). Works for both…, Tests for the DBA (direct binding) forward model. (+7 more)

### Community 68 - "._checkbox_label"
Cohesion: 0.50
Nodes (3): Plain-text label for a subplot's toggle checkbox., fmt_param_plain(), Return a plain-text (no HTML) parameter label, fallback to raw name.

### Community 69 - "controls.py"
Cohesion: 0.19
Nodes (8): NoiseControl, Reusable live controls for the simulation applet. These are generic and assay-…, Optional Gaussian measurement noise: enable, fraction-of-range, replicas, seed., FitConfigPanel — optimizer configuration (n_trials, RMSE factor, min R²)., NoScrollDoubleSpinBox, NoScrollSpinBox, Drop-in QSpinBox / QDoubleSpinBox subclasses that always ignore mouse-wheel…, OutlierRemovalPanel — Z-Score filter configuration and execution.

### Community 70 - "._valid_kwargs"
Cohesion: 0.12
Nodes (7): DBA constructor rejects invalid inputs., GDA constructor rejects invalid inputs., g0 = 0 is legal, unlike h0 and Ka_dye. It is the no-competitor limit in which…, IDA constructor rejects invalid inputs., TestDBAFailFast, TestGDAFailFast, TestIDAFailFast

### Community 71 - ".residuals"
Cohesion: 0.15
Nodes (9): ndarray, Quantity, Compute predicted signal from parameters. Parameters ---------- params :…, Equilibrium speciation across ``x_data`` from the same solve as the signal.…, Compute residuals (observed - predicted). Parameters ---------- params :…, Compute sum of squared residuals (SSR) for optimization. Parameters ----------…, Get default parameter bounds as a name-keyed dictionary. Returns the…, Convert parameter array to named dictionary. Parameters ---------- params :… (+1 more)

### Community 72 - "prepare_plot_data"
Cohesion: 0.30
Nodes (6): prepare_plot_data(), Any, Gather plot-ready data from a MeasurementSet and optional fits. Parameters…, _simple_ms(), TestPrepPlotDataFits, TestPrepPlotDataReplicas

### Community 73 - "test_pipeline_helpers.py"
Cohesion: 0.25
Nodes (8): Merge user overrides with registry defaults -> named bounds dict., Convert parameter names -> positional indices for the optimizer., _resolve_bounds(), _resolve_log_scale(), _gda_assay(), Tests for pipeline helper functions: _resolve_bounds, _resolve_log_scale., TestResolveBounds, TestResolveLogScale

### Community 74 - "FittingSession"
Cohesion: 0.15
Nodes (6): FittingSession, One complete fitting workflow: data → preprocess → configure → fit → visualize.…, Stem-only version of :meth:`_default_save_name` (no tag, no suffix)., DataPanel announces a new plot-display unit — propagate to plot., Session tab-title logic: dataset filename, custom name, and fallbacks. The…, test_tab_title_resolution_and_emission()

### Community 75 - "ZScoreReplicaFilter"
Cohesion: 0.14
Nodes (12): Mark replicas as inactive if any point is a z-score outlier. Uses **robust…, ZScoreReplicaFilter, _ms_with_outlier(), Running filter twice doesn't crash or double-drop., Identical replicas (std=0) → no drops, no errors., Log entry contains threshold, dropped, kept., Create a MeasurementSet where one replica is a clear outlier. The outlier…, Z-score-based replica outlier detection. (+4 more)

### Community 76 - "Molecular Binding Assay Fitting Toolkit"
Cohesion: 0.10
Nodes (20): 1. Load data, 2. Configure and fit, 3. Export results, Binding models, Dependencies, Development, Direct Binding Assay (DBA), Download (+12 more)

### Community 77 - "session.py"
Cohesion: 0.12
Nodes (22): Consolidated multi-artefact export dialog. Lets the user pick which artefacts…, _Row, export_batch(), export_results(), ExportableArtefact, _fixed_width_table(), import_results(), Exception (+14 more)

### Community 78 - "test_results_export.py"
Cohesion: 0.22
Nodes (22): export_results_csv(), export_results_txt(), Export fit results as a human-readable text report. Parameters ----------…, Plain-text row label; log twins are named ``log10(<key>)``., Export fit results as a tidy CSV — one row per reported parameter. Complements…, _row_label(), _csv_rows(), Path (+14 more)

### Community 79 - "SciLineEdit"
Cohesion: 0.16
Nodes (10): format_number(), QLineEdit, Human-readable float: plain for O(1) magnitudes, scientific for the rest. Uses…, Single-float entry shown in adaptive/scientific notation. Preferred over…, Set the shown value from code without emitting ``value_changed``., SciLineEdit, Large magnitudes read as scientific notation; O(1) values stay plain., A user can type '1e8' — the field parses and reformats it, no zero-counting. (+2 more)

### Community 81 - "XlsxReader"
Cohesion: 0.31
Nodes (7): DataFrame, Path, Reader for Excel measurement files. Dispatches on workbook structure: BMG…, Read an Excel measurement file. Parameters ---------- path : Path Path to the…, Pandas-based long / wide / multi-sheet fallback., XlsxReader, ExcelFile

### Community 82 - "ScientificAxisItem"
Cohesion: 0.21
Nodes (8): GraphicsLayoutWidget, Build a transient ``GraphicsLayoutWidget`` whose cells match the live GUI. Each…, Ensure we have exactly *n* PlotWidget instances with ScientificAxisItem., Wire *axis* to update *plot_item*'s y-label when its exponent changes., _wire_exponent_callback(), AxisItem that formats tick labels with a shared exponent. When all tick values…, Update exponent, firing callback only on change. The callback guard (``exp !=…, ScientificAxisItem

### Community 83 - "fitting_session.py"
Cohesion: 0.12
Nodes (16): _Grouped(), _GroupedPlain, _GroupedWithInfo, QGroupBox, QWidget, FittingSession — one complete fitting workspace (one tab)., Save current plot style settings to a JSON file., Load plot style settings from a JSON file. (+8 more)

### Community 84 - "_sample_measurement_set"
Cohesion: 0.08
Nodes (18): DataFrame, Tests for MeasurementSet construction and behaviour., Raw data arrays are read-only after construction., Active mask management., iter_replicas, get_replica_signal, average_signal., Build a simple long-form DataFrame with known values., Yielded signals are views (not copies) into the underlying array., Dropping a replica must exclude it from the mean. _sample_dataframe gives… (+10 more)

### Community 85 - "Scientific Summary of the Molecular Binding Assay Fitting Toolkit"
Cohesion: 0.10
Nodes (19): 1. Introduction, 2. Scientific Methodology: The Forward Modeling Approach, 3.1. Law of Mass Action, 3.2. Mass Balance Equations, 3.3. Signal Response Model, 3. Mathematical Framework, 4.1. Numerical Optimization, 4.2. Root-Finding Algorithms (+11 more)

### Community 86 - "linear_regression"
Cohesion: 0.20
Nodes (7): Fit linear model using simple linear regression. Returns -------…, linear_regression(), ndarray, Linear fitting for dye-alone calibration., Perform simple linear regression. Parameters ---------- x : np.ndarray…, Edge cases for the linear regression helper., TestLinearRegression

### Community 87 - "F2 (root cause): au defined as dimensionless and shadowing astronomical_unit"
Cohesion: 0.25
Nodes (9): F1: _scale_factor classified units by .dimensionless, F2 (root cause): au defined as dimensionless and shadowing astronomical_unit, Prior-art consensus: Quantities at edges, magnitudes in the core, au = [signal] custom dimension, Validate units once per boundary crossing, ParamKind plus [signal] dimension dual classification, Quantities at the edges, floats in the core, Residual subtraction as a runtime signal-law check (+1 more)

### Community 89 - "main_window.py"
Cohesion: 0.24
Nodes (8): _app_icon_path(), launch(), FittingMainWindow — thin shell managing tabbed fitting sessions., Block non-focused spinboxes from swallowing wheel events. Qt's event-filter…, Locate the bundled app icon for both source runs and PyInstaller bundles., Entry point — create the QApplication and launch the main window., _SpinBoxWheelRedirect, QObject

### Community 90 - "InfoGroupBox"
Cohesion: 0.14
Nodes (9): Read-only display widget for FitResult statistics., _info_button_qss(), InfoButton, InfoGroupBox, QGroupBox, QToolButton, Reusable round info button that opens a dialog with rich HTML content., QGroupBox with an :class:`InfoButton` inline after the title text. The button… (+1 more)

### Community 91 - "PreprocessingPanel"
Cohesion: 0.27
Nodes (4): PreprocessingPanel, Return preprocessing step configuration list for :func:`apply_preprocessing`., Programmatically apply the current preprocessing steps. Unlike the interactive…, Configure and apply outlier removal to a MeasurementSet. Exposes the Z-Score…

### Community 92 - "Path"
Cohesion: 0.25
Nodes (6): _Accepter, _Fallback, DataFrame, Path, No can_read → always-accepting fallback., _Rejecter

### Community 93 - "Minimize Residuals"
Cohesion: 0.20
Nodes (7): ndarray, Return a tilded-parameter objective wrapping a raw-parameter one.…, Raw parameters → tilded., Tilded parameters → raw., Compute RMSE & R2, Minimize Residuals, Optimize Parameters

### Community 94 - "._default_save_name"
Cohesion: 0.17
Nodes (6): Build a default save filename based on the loaded dataset's stem. Falls back to…, Export current fit results to JSON., Export current fit results as a human-readable text report., Export current fit results as a machine-readable CSV table., Save the current plot as PNG or SVG., Save the distributions plot as a composite PNG with a layout picker.

### Community 95 - "fit_measurement_set"
Cohesion: 0.14
Nodes (13): fit_measurement_set(), Convenience: build an assay from a MeasurementSet and fit it. When…, FitWorker, Any, QThread, Run :func:`fit_measurement_set` in a background thread. Prevents the GUI from…, fit_measurement_set dispatches to fit_linear_assay for DyeAlone., fit_measurement_set dispatches to fit_assay for GDA. (+5 more)

### Community 96 - "ParamKind"
Cohesion: 0.18
Nodes (11): ParamKind, Semantic identity of a fitted parameter. Dimensional analysis alone cannot…, Registry unit-lint and the signal-dimension invariant. These guard the "both"…, Every registered parameter carries an explicit ParamKind., Each parameter's ParamKind is dimensionally consistent with its unit. With ``au…, A parameter is a BINDING_CONSTANT iff it is sampled in log space., ``au`` is not dimensionless, so ``au/M`` cannot be read as ``1/M``., test_binding_constants_are_exactly_the_log_scale_params() (+3 more)

### Community 98 - "multistart_minimize"
Cohesion: 0.18
Nodes (9): multistart_minimize(), ndarray, Run L-BFGS-B optimizer from multiple starting points. Parameters ----------…, Provided initial_guesses are used instead of random generation., compute_metrics populates rmse and r_squared fields., Objective that always raises returns empty results., Without compute_metrics, rmse = sqrt(cost) and r_squared = NaN., Minimizes f(x) = (x-3)^2 with bounds [0, 10]. (+1 more)

### Community 99 - "_DraggableTextItem"
Cohesion: 0.14
Nodes (7): _DraggableTextItem, Remember a user drag so later rebuilds stop re-placing the box., A ``TextItem`` that reports where the user dropped it. ``pg.TextItem`` has no…, Hand the drop position to *on_moved*, but only after a real move. A press whose…, QPointF, Auto-placement applies until the user moves the box; then it stays put., test_user_drag_survives_a_rebuild()

### Community 100 - "UpdateCheckWorker"
Cohesion: 0.25
Nodes (7): Cross-platform build matrix, Frozen uv dependency install, GitHub release publication job, PyInstaller standalone bundle, QThread, Query GitHub ``/releases/latest`` for the configured repo. Signals -------…, UpdateCheckWorker

### Community 101 - "_SidebarScrollArea"
Cohesion: 0.36
Nodes (4): Vertically-scrolling sidebar whose width follows its content. A plain…, _SidebarScrollArea, QScrollArea, QSize

### Community 102 - "test_fitting_session_layout.py"
Cohesion: 0.29
Nodes (7): _assert_embedded(), Regression guard: the plot widget must stay embedded after any import. A…, The sidebar must be freely widenable — no finite max-width cap. A previous…, The sidebar must advertise at least its content's width to the splitter. A…, test_plot_stays_embedded_through_ensight_load_and_switch(), test_sidebar_has_no_hard_maxwidth_cap(), test_sidebar_reserves_content_width_so_it_never_clips()

### Community 103 - "._best_overlay_slot"
Cohesion: 0.14
Nodes (9): _ErrorBarSample, ndarray, Move the annotation to the emptiest slot, unless the user moved it. Once…, Re-place the annotation once the view range or size is real., Legend sample that renders an error-bar glyph instead of a line. pyqtgraph's…, Every plotted vertex, in ViewBox-local pixel coordinates. Includes the fitted…, Emptiest slot for a *size* box: ``(anchor_fractions, rect)`` in pixels. A…, The legend's current footprint in ViewBox-local pixels, if placed. (+1 more)

### Community 104 - "TestCalculateFitMetrics"
Cohesion: 0.25
Nodes (4): When all y_observed are identical, ss_tot=0 → R²=0., Hand-computed RMSE and R² for simple data. residuals = [-0.1, 0, +0.1] → ss_res…, A NaN anywhere in y_predicted (failed model eval) must yield NaN metrics, never…, TestCalculateFitMetrics

### Community 106 - "FitConfigPanel"
Cohesion: 0.25
Nodes (5): FitConfigPanel, QWidget, Editor for :class:`~core.pipeline.fit_pipeline.FitConfig` parameters. Signals…, Row: [enable checkbox] [factor spinbox] [info] for the optional trim., Wrap ``widget`` in an HBox together with a trailing info button. The spinbox…

### Community 107 - "TestBrentBracketRobustness"
Cohesion: 0.33
Nodes (5): parametrize, Verify the competitive Brent solver finds a physical root across several…, [H_free] must be finite and in (0, h0) regardless of Ka magnitudes., DBA quadratic must produce a non-negative free fixed-species root., TestBrentBracketRobustness

### Community 108 - ".forward_model"
Cohesion: 0.33
Nodes (4): ndarray, Quantity, Free host, dye, guest and both complexes (M) across the dye titration., Compute predicted signal from parameters. Parameters ---------- params :…

### Community 109 - "._load_single"
Cohesion: 0.20
Nodes (6): H3: DataPanel reused a stale Imported Unit for out-of-set units, Load a single file: parse → build the MeasurementSet → emit. No modal dialog…, Build a MeasurementSet from a single-channel frame, forwarding the…, Return the single-channel sub-frame for *channel*, attrs preserved., Rebuild the MeasurementSet for the newly selected channel in-memory. The…, Return the token in ``UNITS`` whose pint unit matches ``units``. Matched by…

### Community 110 - "AssayTypeSelector"
Cohesion: 0.29
Nodes (4): AssayTypeSelector, QWidget, Two dependent combos: assay *category* then *subtype*. Signals -------…, Programmatically select an assay, emitting ``assay_type_changed`` once.

### Community 111 - ".read"
Cohesion: 0.33
Nodes (4): DataFrame, Path, Read measurement data from file. Parameters ---------- path : Path Path to…, Write fit results to file. Parameters ---------- results : dict Fit results…

### Community 112 - "parametrize"
Cohesion: 0.20
Nodes (6): parametrize, Each stepwise parameter renders a real HTML label, not the raw-key fallback., Selecting a stepwise assay in the panel yields conditions that build a valid…, TestConfigPanelToAssay, TestLabelsAndHelp, TestRegistration

### Community 113 - "test_assay_config_panel.py"
Cohesion: 0.25
Nodes (6): _direct_binding_types(), panel(), fixture, Two-level (category → subtype) assay selector behaviour. Verifies the dependent…, test_changing_category_repopulates_subtypes_and_emits_once(), test_default_is_gda_without_emission()

### Community 114 - "SimKnob"
Cohesion: 0.25
Nodes (5): (Re)build the maximum-concentration knob for a new assay's titrant., One live numeric control: a value plus a user-adjustable [vmin, vmax]., Molar-dimensioned knobs get a nM/µM/mM/M selector (Pint dimensionality)., Concentrations and association constants are strictly positive; signal…, SimKnob

### Community 116 - "._make_menu_button"
Cohesion: 0.50
Nodes (3): QToolButton, Create a toolbar button that pops a menu without the Qt auto-arrow. The default…, QMenu

### Community 117 - "BoundsPanel"
Cohesion: 0.21
Nodes (5): BoundsPanel, Quantity, Registry-driven parameter bounds editor. Auto-populates from…, Return custom Quantity bounds or None if all match defaults., Apply dye-alone–derived bounds to I0 and I_dye_free rows.

### Community 118 - "TestRepresentativeFitTrustworthy"
Cohesion: 0.40
Nodes (3): Regression for #31. On the degenerate IDA signal model every valid fit sits on…, A per-parameter median lands off the degenerate manifold, so it reconstructs…, TestRepresentativeFitTrustworthy

### Community 119 - "TestHG2MassBalance"
Cohesion: 0.40
Nodes (3): The 1:2 speciation must conserve both total host and total guest., Brent speciation agrees with an fsolve of the coupled equilibria. Solved in µM…, TestHG2MassBalance

### Community 120 - "TestModelVsIndependentEquilibrium"
Cohesion: 0.33
Nodes (4): Verify the forward models against equilibrium solutions computed by a different…, dba_signal (quadratic) agrees with a Brent solve of the dye balance., competitive_signal_point agrees with an fsolve of the full 3-species system…, TestModelVsIndependentEquilibrium

### Community 121 - "TestStepwiseAssayContracts"
Cohesion: 0.47
Nodes (3): parametrize, Fixed host concentration is required, positive, and unit-bearing., TestStepwiseAssayContracts

### Community 123 - "QHBoxLayout"
Cohesion: 0.16
Nodes (10): Append one checkbox + HTML label to a shared single-row layout., _labelled(), QWidget, Short Unicode unit label for a static (non-selectable) unit., Wrap a numeric field with a small muted ``min``/``max`` caption., _unit_text(), QWidget, QDoubleSpinBox (+2 more)

### Community 125 - "gui/conftest.py"
Cohesion: 0.50
Nodes (3): fixture, qapp(), Shared GUI-test fixtures. A single session-scoped ``qapp`` fixture lives here…

### Community 127 - "._decimals_for_scale"
Cohesion: 0.50
Nodes (3): Per-unit decimals via the shared policy in ``gui.widgets.numeric_inputs``., decimals_for_scale(), Spinbox decimals so a value resolvable in the finest offered unit survives…

### Community 129 - "PlotWidget"
Cohesion: 0.12
Nodes (13): PlotWidget, PlotItem, QWidget, Qt widget that renders ``prepare_plot_data()`` output via PyQtGraph. Parameters…, The underlying pyqtgraph PlotItem — used to x-link a second plot to this one., Current x-axis concentration unit (e.g. ``'µM'``) — for a linked plot to match., Export the current plot to a PNG or SVG file. Parameters ---------- path : str…, Set an axis label, appending ×10ⁿ if *exp* is not None. (+5 more)

## Knowledge Gaps
- **45 isolated node(s):** `uv`, `_ParamSpec`, `fitting-app`, `Download`, `Quick start (from source)` (+40 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1085 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MeasurementSet` connect `MeasurementSet` to `FitConfig`, `test_preprocessing.py`, `SimulationWindow`, `DataPanel`, `FitResult`, `test_data_panel.py`, `load_measurements`, `bounds_panel.py`, `test_pipeline_e2e.py`, `ReplicaPanel`, `write_measurements_txt`, `.from_dataframe`, `data_panel.py`, `AssayType`, `test_simulation.py`, `TestConstruction`, `.to_assay`, `test_io_bmg.py`, `controls.py`, `prepare_plot_data`, `FittingSession`, `ZScoreReplicaFilter`, `fitting_session.py`, `_sample_measurement_set`, `PreprocessingPanel`, `fit_measurement_set`, `._emit_tab_title`, `._load_single`, `SessionState`, `BoundsPanel`?**
  _High betweenness centrality (0.178) - this node is a cross-community bridge._
- **Why does `AssayType` connect `AssayType` to `units.py`, `_sample_fit_result`, `SimulationWindow`, `tests/conftest.py`, `SimulationPanel`, `test_simulation_applet.py`, `FitResult`, `AssayConfigPanel`, `._populate_subplot`, `bounds_panel.py`, `test_pipeline_e2e.py`, `distribution_widget.py`, `DistributionWidget`, `DBAAssay`, `test_simulation.py`, `IDAAssay`, `FittingSession`, `fitting_session.py`, `AssayTypeSelector`, `test_assay_config_panel.py`, `SessionState`, `BoundsPanel`?**
  _High betweenness centrality (0.112) - this node is a cross-community bridge._
- **Why does `FitResult` connect `FitResult` to `FitConfig`, `Generate Multiple Parameter Sets`, `_sample_fit_result`, `FitSummaryWidget`, `test_plot_widget.py`, `test_export.py`, `._populate_subplot`, `test_pipeline_e2e.py`, `distribution_widget.py`, `DistributionWidget`, `_binding_plot`, `AssayType`, `prepare_plot_data`, `FittingSession`, `session.py`, `test_results_export.py`, `fitting_session.py`, `._refresh_plot`, `InfoGroupBox`, `fit_measurement_set`, `SessionState`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Are the 30 inferred relationships involving `MeasurementSet` (e.g. with `BaseAssay` and `prepare_plot_data()`) actually correct?**
  _`MeasurementSet` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 35 inferred relationships involving `AssayType` (e.g. with `BaseAssay` and `DBAAssay`) actually correct?**
  _`AssayType` has 35 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `FittingSession` (e.g. with `AssayType` and `MeasurementSet`) actually correct?**
  _`FittingSession` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `FitConfig` (e.g. with `Any Good Fits? (decision)` and `More Trials? (decision)`) actually correct?**
  _`FitConfig` has 22 INFERRED edges - model-reasoned connections that need verification._