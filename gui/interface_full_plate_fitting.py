import os
import tkinter as tk
import traceback
from tkinter import filedialog

from core.fitting.full_plate_fit import run_full_plate_fit
from core.progress_window import ProgressWindow
from utils.bmg_to_txt import ConcentrationVectorManager


# ---------------- Tooltip Helper -----------------
class ToolTip:
    """Simple tooltip for a tkinter widget.

    Usage: ToolTip(widget, text="...")
    """

    def __init__(self, widget, text, wraplength=340):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tipwindow = None
        self._id = None
        self.widget.bind("<Enter>", self._enter)
        self.widget.bind("<Leave>", self._leave)

    def _enter(self, _event):
        self._schedule()

    def _leave(self, _event):
        self._unschedule()
        self._hide()

    def _schedule(self):
        self._unschedule()
        self._id = self.widget.after(450, self._show)

    def _unschedule(self):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None

    def _show(self):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            wraplength=self.wraplength,
            padx=6,
            pady=4,
        )
        label.pack(ipadx=1)

    def _hide(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw is not None:
            tw.destroy()


class FullPlateFittingApp:
    """UI for running full plate fitting directly from BMG Excel data.

    Features:
      - Excel file selection (raw plate export)
      - Concentration vector management (save/reuse sets) in M (user inputs µM; converted later)
      - Assay type selection with dynamic fixed parameter fields
      - Optional dye-alone results file (bounds) reuse
      - Advanced options: filtering of alternative fits, plotting of additional fits, outlier detection
      - Plot & results saving controls; custom titles/labels
    """

    # (Display Label, Internal Code)
    ASSAY_TYPES = [
        ("DBA (H → D)", "dba_HtoD"),
        ("DBA (D → H)", "dba_DtoH"),
        ("IDA", "ida"),
        ("GDA", "gda"),
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Full Plate Fitting")
        self.pad_x = 10
        self.pad_y = 4
        self.info_label = None

        # Core variables
        self.excel_path_var = tk.StringVar()
        self.use_bounds_file_var = tk.BooleanVar()
        self.bounds_file_var = tk.StringVar()
        # Store display label; map to internal code later
        self.assay_type_var = tk.StringVar(value=self.ASSAY_TYPES[0][0])

        # Fixed assay parameters (entries created dynamically)
        self.param_vars = {
            "Kd": tk.DoubleVar(),
            "h0": tk.DoubleVar(),
            "d0": tk.DoubleVar(),
            "g0": tk.DoubleVar(),
        }

        # Concentration vector management
        self.cv_manager = ConcentrationVectorManager()
        self.cv_entry_var = tk.StringVar()
        self.cv_name_var = tk.StringVar()
        self.cv_dropdown_var = tk.StringVar()

        # Fit settings
        self.fit_trials_var = tk.IntVar(value=100)

        # Filtering options
        self.enable_filter_var = tk.BooleanVar()
        self.rmse_factor_var = tk.DoubleVar(value=2.0)
        self.k_factor_var = tk.DoubleVar(value=2.0)
        self.r2_threshold_var = tk.DoubleVar(value=0.9)
        self.other_fits_var = tk.IntVar(value=0)

        # Outlier detection options
        self.enable_outlier_var = tk.BooleanVar()
        self.outlier_method_var = tk.StringVar(value="mad")
        self.outlier_threshold_var = tk.DoubleVar(value=2)
        self.outlier_max_exclude_var = tk.IntVar(value=2)

        # Plot & result options
        self.save_plots_var = tk.BooleanVar()
        self.plots_dir_var = tk.StringVar()
        self.display_plots_var = tk.BooleanVar(value=True)
        self.save_results_var = tk.BooleanVar()
        self.results_dir_var = tk.StringVar()
        self.custom_x_label_var = tk.BooleanVar()
        self.custom_x_label_text_var = tk.StringVar()
        self.custom_plot_title_var = tk.BooleanVar()
        self.custom_plot_title_text_var = tk.StringVar()

        # # for testing
        # self.excel_path_var.set(
        #     "/Users/ahmadomira/Downloads/DBA_h2d_with_outliers.xlsx"
        # )
        # self.cv_entry_var.set("0, 30, 60, 100, 150, 225, 300, 400, 500, 600, 700, 840")
        # self.param_vars["d0"].set(6e-6)
        # self.enable_filter_var.set(True)
        # self.enable_outlier_var.set(True)
        # self.save_plots_var.set(True)
        # self.plots_dir_var.set("/Users/ahmadomira/Downloads")
        # self.save_results_var.set(True)
        # self.results_dir_var.set("/Users/ahmadomira/Downloads")

        self._build_ui()

        # Bring window forward
        self.root.lift()
        self.root.focus_force()

    # ---------------- UI Construction -----------------
    def _build_ui(self):
        row = 0
        tk.Label(self.root, text="Excel Plate File (*.xlsx):").grid(
            row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        tk.Entry(
            self.root, textvariable=self.excel_path_var, width=55, justify="left"
        ).grid(row=row, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)
        tk.Button(self.root, text="Browse", command=self._browse_excel).grid(
            row=row, column=2, pady=self.pad_y, sticky=tk.W
        )
        row += 1

        dye_alone_btn = tk.Checkbutton(
            self.root,
            text="Use Dye Alone Results:",
            variable=self.use_bounds_file_var,
            command=self._update_bounds_widgets,
        )
        dye_alone_btn.grid(
            row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        self.bounds_entry = tk.Entry(
            self.root,
            textvariable=self.bounds_file_var,
            width=55,
            state=tk.DISABLED,
            justify="left",
        )

        ToolTip(
            dye_alone_btn,
            "Use dye-only experiment results to set initial guesses and bounds for I₀ and Id.",
        )

        self.bounds_entry.grid(
            row=row, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W
        )
        self.bounds_browse_btn = tk.Button(
            self.root,
            text="Browse",
            state=tk.DISABLED,
            command=lambda: self._browse_generic(self.bounds_entry),
        )
        self.bounds_browse_btn.grid(row=row, column=2, pady=self.pad_y, sticky=tk.W)
        row += 1

        # Assay type
        tk.Label(self.root, text="Assay Type:").grid(
            row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        assay_menu = tk.OptionMenu(
            self.root,
            self.assay_type_var,
            *[a[0] for a in self.ASSAY_TYPES],
            command=lambda *_: self._refresh_assay_params(),
        )
        assay_menu.grid(
            row=row, column=1, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        row += 1

        # Dynamic fixed parameter frame
        self.params_frame = tk.Frame(self.root)
        self.params_frame.grid(
            row=row,
            column=1,
            columnspan=3,
            sticky=tk.W,
            padx=self.pad_x,
            pady=(self.pad_y, self.pad_y),
        )
        row += 1
        self._refresh_assay_params()

        # Concentration vector section (wider entry & dropdown, consolidated export/import)
        tk.Label(self.root, text="Concentrations (µM):").grid(
            row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        tk.Entry(
            self.root, textvariable=self.cv_entry_var, width=50, justify="left"
        ).grid(row=row, column=1, sticky=tk.W, padx=(self.pad_x, 2), pady=self.pad_y)
        tk.Label(self.root, text="Label:").grid(
            row=row, column=1, sticky=tk.E, padx=(0, 0), pady=self.pad_y
        )
        # Name entry for saving the concentration set; Enter triggers save
        self.cv_name_entry = tk.Entry(
            self.root, textvariable=self.cv_name_var, width=9, justify="left"
        )
        self.cv_name_entry.grid(
            row=row, column=2, sticky=tk.W, padx=(2, 2), pady=self.pad_y
        )
        self.cv_name_entry.bind("<Return>", lambda _e: self._save_cv())
        self.cv_name_entry.bind("<FocusOut>", lambda _e: self._save_cv())
        self.cv_name_entry.bind("<KP_Enter>", lambda _e: self._save_cv())
        row += 1

        ToolTip(
            self.cv_name_entry,
            "Name the concentration vector (titrated compound) and press Enter. You can save multiple vectors per session and export/import them for reuse.",
        )

        tk.Label(self.root, text="Saved Sets:").grid(
            row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        self.cv_dropdown = tk.OptionMenu(
            self.root, self.cv_dropdown_var, "", command=lambda *_: self._select_cv()
        )
        self.cv_dropdown.grid(
            row=row, column=1, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        # self.cv_dropdown.config(width=10)
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=row, column=1, padx=self.pad_x, pady=self.pad_y)
        tk.Button(
            btn_frame, text="Import Conc.", command=self._import_cvs, width=7
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            btn_frame, text="Export Conc.", command=self._export_cvs, width=7
        ).pack(side=tk.LEFT, padx=(0, 4))
        row += 1

        # Fit trials
        tk.Label(self.root, text="Fit Trials:").grid(
            row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        tk.Entry(
            self.root, textvariable=self.fit_trials_var, width=10, justify="left"
        ).grid(row=row, column=1, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        row += 1

        # Outlier detection options (moved above filtering)
        outlier_chk = tk.Checkbutton(
            self.root,
            text="Outlier Detection",
            variable=self.enable_outlier_var,
            command=self._toggle_outlier_widgets,
        )
        outlier_chk.grid(
            row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        ToolTip(
            outlier_chk,
            "When enabled, replicate-level outliers are flagged using the chosen method (currently MAD) before averaging. This can improve robustness by excluding anomalous wells. When disabled, all replicates are averaged as-is.",
        )
        self.outlier_frame = tk.Frame(self.root)
        self.outlier_frame.grid(
            row=row,
            column=1,
            columnspan=2,
            sticky=tk.W,
            padx=self.pad_x,
            pady=self.pad_y,
        )
        row += 1
        self._build_outlier_widgets()

        # Filtering options (below outlier section)
        filter_button = tk.Checkbutton(
            self.root,
            text="Include Other Fits:",
            variable=self.enable_filter_var,
            command=self._toggle_filter_widgets,
        )
        filter_button.grid(
            row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        ToolTip(
            filter_button,
            "Enable to retain and analyze additional optimization trials that are close to the best fit (by RMSE, Ka window, and R²). Disabled: only the single best fit is reported and plotted.",
        )
        self.filter_frame = tk.Frame(self.root)
        self.filter_frame.grid(
            row=row,
            column=1,
            columnspan=2,
            sticky=tk.W,
            padx=self.pad_x,
            pady=self.pad_y,
        )
        row += 1
        self._build_filter_widgets()

        # Plot & results options
        tk.Checkbutton(
            self.root,
            text="Save Plots To",
            variable=self.save_plots_var,
            command=self._toggle_plot_widgets,
        ).grid(row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.plots_entry = tk.Entry(
            self.root,
            textvariable=self.plots_dir_var,
            width=42,
            state=tk.DISABLED,
            justify="left",
        )
        self.plots_entry.grid(
            row=row, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W
        )
        self.plots_browse_btn = tk.Button(
            self.root,
            text="Browse",
            state=tk.DISABLED,
            command=lambda: self._browse_directory(self.plots_entry),
        )
        self.plots_browse_btn.grid(row=row, column=2, padx=self.pad_x, pady=self.pad_y)
        row += 1

        tk.Checkbutton(
            self.root,
            text="Save Results To",
            variable=self.save_results_var,
            command=self._toggle_results_widgets,
        ).grid(row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.results_entry = tk.Entry(
            self.root,
            textvariable=self.results_dir_var,
            width=42,
            state=tk.DISABLED,
            justify="left",
        )
        self.results_entry.grid(
            row=row, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W
        )
        self.results_browse_btn = tk.Button(
            self.root,
            text="Browse",
            state=tk.DISABLED,
            command=lambda: self._browse_directory(self.results_entry),
        )
        self.results_browse_btn.grid(
            row=row, column=2, padx=self.pad_x, pady=self.pad_y
        )
        row += 1

        tk.Checkbutton(
            self.root,
            text="Custom Plot Title",
            variable=self.custom_plot_title_var,
            command=self._toggle_custom_plot_title,
        ).grid(row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.custom_plot_title_entry = tk.Entry(
            self.root,
            textvariable=self.custom_plot_title_text_var,
            width=42,
            state=tk.DISABLED,
            justify="left",
        )
        self.custom_plot_title_entry.grid(
            row=row,
            column=1,
            columnspan=2,
            sticky=tk.W,
            padx=self.pad_x,
            pady=self.pad_y,
        )
        row += 1

        tk.Checkbutton(
            self.root,
            text="Custom X Label",
            variable=self.custom_x_label_var,
            command=self._toggle_custom_x_label,
        ).grid(row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.custom_x_label_entry = tk.Entry(
            self.root,
            textvariable=self.custom_x_label_text_var,
            width=42,
            state=tk.DISABLED,
            justify="left",
        )
        self.custom_x_label_entry.grid(
            row=row,
            column=1,
            columnspan=2,
            sticky=tk.W,
            padx=self.pad_x,
            pady=self.pad_y,
        )
        row += 1

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        row += 1

        tk.Button(
            self.root, text="Run Full Plate Fitting", command=self._run_fitting
        ).grid(row=row, column=0, columnspan=3, pady=(10, 6))
        row += 1

    # ------------- Dynamic Section Builders -------------
    def _refresh_assay_params(self):
        for w in self.params_frame.winfo_children():
            w.destroy()
        display_val = self.assay_type_var.get()
        assay = next(
            (code for (label, code) in self.ASSAY_TYPES if label == display_val),
            display_val,
        )
        col = 0
        row = 0

        def add(label, var_key):
            nonlocal col
            tk.Label(self.params_frame, text=label).grid(
                row=row, column=col, sticky=tk.W, padx=(0, 6), pady=self.pad_y
            )
            col += 1
            tk.Entry(
                self.params_frame,
                textvariable=self.param_vars[var_key],
                width=12,
                justify="left",
            ).grid(row=row, column=col, padx=(0, 14))
            col += 1

        if assay == "ida":
            add("Kₐ (M⁻¹):", "Kd")
            add("H₀ (M):", "h0")
            add("D₀ (M):", "d0")
        elif assay == "dba_HtoD":
            add("D₀ (M):", "d0")
        elif assay == "dba_DtoH":
            add("H₀ (M):", "h0")
        elif assay == "gda":
            add("Kₐ (M⁻¹):", "Kd")
            add("H₀ (M):", "h0")
            add("G₀ (M):", "g0")

    def _build_filter_widgets(self):
        for w in self.filter_frame.winfo_children():
            w.destroy()
        if self.enable_filter_var.get():
            lbl_rmse = tk.Label(self.filter_frame, text="RMSE Tolerance:")
            lbl_rmse.grid(row=0, column=0, sticky=tk.W)
            tk.Entry(
                self.filter_frame,
                textvariable=self.rmse_factor_var,
                width=6,
                justify="left",
            ).grid(row=0, column=1, padx=(0, 10))
            lbl_k = tk.Label(self.filter_frame, text="Ka Tolerance:")
            lbl_k.grid(row=0, column=2, sticky=tk.W)
            tk.Entry(
                self.filter_frame,
                textvariable=self.k_factor_var,
                width=6,
                justify="left",
            ).grid(row=0, column=3, padx=(0, 10))
            lbl_r2 = tk.Label(self.filter_frame, text="R² ≥")
            lbl_r2.grid(row=0, column=4, sticky=tk.W)
            tk.Entry(
                self.filter_frame,
                textvariable=self.r2_threshold_var,
                width=6,
                justify="left",
            ).grid(row=0, column=5, padx=(0, 10))
            lbl_other = tk.Label(self.filter_frame, text="Other fits to plot:")
            lbl_other.grid(row=0, column=6, sticky=tk.W)
            tk.Entry(
                self.filter_frame,
                textvariable=self.other_fits_var,
                width=4,
                justify="left",
            ).grid(row=0, column=7)

            # Attach tooltips
            ToolTip(
                lbl_rmse,
                "Retain fits with RMSE ≤ ([provided factor] × best RMSE). Lower values tighten selection; higher values are more permissive. Default 2.0 is moderate.",
            )
            ToolTip(
                lbl_k,
                "Accept other fits with Ka within [K_best / factor, K_best × factor]. Lower factor narrows allowed K variability. Higher values allow for more Ka variability.",
            )
            ToolTip(
                lbl_r2,
                "Minimum R² (coefficient of determination). Fits below provided value are discarded. Typical 0.9–0.98.",
            )
            ToolTip(
                lbl_other,
                "Number of fits (after filtering) to overlay in gray in the plot for visual comparison.",
            )

    def _build_outlier_widgets(self):
        for w in self.outlier_frame.winfo_children():
            w.destroy()
        if self.enable_outlier_var.get():
            lbl_method = tk.Label(self.outlier_frame, text="Method:")
            lbl_method.grid(row=0, column=0, sticky=tk.W)
            method_menu = tk.OptionMenu(
                self.outlier_frame, self.outlier_method_var, "MAD"
            )
            # Display uppercase while preserving lowercase for backend usage later
            self.outlier_method_var.set("MAD")
            method_menu.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
            lbl_z = tk.Label(self.outlier_frame, text="Z Threshold:")
            lbl_z.grid(row=0, column=2, sticky=tk.W)
            tk.Entry(
                self.outlier_frame,
                textvariable=self.outlier_threshold_var,
                width=6,
                justify="left",
            ).grid(row=0, column=3, padx=(0, 10))
            lbl_max = tk.Label(self.outlier_frame, text="Max exclude:")
            lbl_max.grid(row=0, column=4, sticky=tk.W)
            tk.Entry(
                self.outlier_frame,
                textvariable=self.outlier_max_exclude_var,
                width=4,
                justify="left",
            ).grid(row=0, column=5)

            # Attach tooltips
            ToolTip(
                lbl_method,
                "Outlier detection algorithm applied to replicate RMSE values versus the median curve. Currently only MAD (Median Absolute Deviation) is implemented.",
            )
            ToolTip(
                lbl_z,
                "Modified z-score threshold applied to replicate RMSEs. Replicates with score > threshold are candidates. Typical range 2.5–3.5 (lower = more aggressive).",
            )
            ToolTip(
                lbl_max,
                "Upper limit on how many replicate outliers may be excluded (highest scoring first). Prevents removing too many replicates in noisy datasets.",
            )

    # ------------- Widget state toggles -------------
    def _toggle_filter_widgets(self):
        self._build_filter_widgets()

    def _toggle_outlier_widgets(self):
        self._build_outlier_widgets()

    def _toggle_plot_widgets(self):
        state = tk.NORMAL if self.save_plots_var.get() else tk.DISABLED
        self.plots_entry.config(state=state)
        self.plots_browse_btn.config(state=state)

    def _toggle_results_widgets(self):
        state = tk.NORMAL if self.save_results_var.get() else tk.DISABLED
        self.results_entry.config(state=state)
        self.results_browse_btn.config(state=state)

    def _toggle_custom_plot_title(self):
        state = tk.NORMAL if self.custom_plot_title_var.get() else tk.DISABLED
        self.custom_plot_title_entry.config(state=state)

    def _toggle_custom_x_label(self):
        state = tk.NORMAL if self.custom_x_label_var.get() else tk.DISABLED
        self.custom_x_label_entry.config(state=state)

    def _update_bounds_widgets(self):
        state = tk.NORMAL if self.use_bounds_file_var.get() else tk.DISABLED
        self.bounds_entry.config(state=state)
        self.bounds_browse_btn.config(state=state)

    # ------------- Concentration vector mgmt -------------
    def _save_cv(self):
        name = self.cv_name_var.get().strip()
        if not name:
            self._show_message("Name required for concentration set", True)
            return
        values_raw = self.cv_entry_var.get().replace(",", " ").split()
        try:
            values = [float(v) for v in values_raw]
        except ValueError:
            self._show_message("Invalid concentration values", True)
            return
        self.cv_manager.add_vector(name, values)
        self._refresh_cv_dropdown()
        self.cv_dropdown_var.set(name)
        self._show_message(f"Saved set '{name}'", False)

    def _refresh_cv_dropdown(self):
        menu = self.cv_dropdown["menu"]
        menu.delete(0, "end")
        for n in self.cv_manager.get_names():
            menu.add_command(label=n, command=lambda v=n: self._on_cv_select(v))

    def _on_cv_select(self, name):
        self.cv_dropdown_var.set(name)
        values = self.cv_manager.get_vector(name)
        self.cv_entry_var.set(" ".join(str(v) for v in values))

    def _select_cv(self):  # triggered by optionmenu change
        name = self.cv_dropdown_var.get()
        if name:
            self._on_cv_select(name)

    def _export_cvs(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if not path:
            return
        with open(path, "w") as f:
            for name, values in self.cv_manager.vectors.items():
                f.write(name + "," + ",".join(str(v) for v in values) + "\n")
        self._show_message("Exported concentration sets", False)

    def _import_cvs(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        try:
            self.cv_manager.vectors.clear()
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        name = parts[0]
                        try:
                            values = [float(v) for v in parts[1:] if v]
                        except ValueError:
                            continue
                        self.cv_manager.add_vector(name, values)
            self._refresh_cv_dropdown()
            self._show_message("Imported sets", False)
        except Exception as e:
            self._show_message(f"Import error: {e}", True)

    # ------------- File dialogs -------------
    def _browse_excel(self):
        path = filedialog.askopenfilename(
            filetypes=[("Excel", "*.xlsx"), ("All", "*.*")]
        )
        if path:
            self.excel_path_var.set(path)
            root_dir = os.path.dirname(path)
            if not self.plots_dir_var.get():
                self.plots_dir_var.set(root_dir)
            if not self.results_dir_var.get():
                self.results_dir_var.set(root_dir)

    def _browse_directory(self, entry):
        initial_dir = (
            os.path.dirname(self.excel_path_var.get())
            if self.excel_path_var.get()
            else os.getcwd()
        )
        dir_path = filedialog.askdirectory(initialdir=initial_dir)
        if dir_path:
            entry.delete(0, tk.END)
            entry.insert(0, dir_path)

    def _browse_generic(self, entry):
        path = filedialog.askopenfilename()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    # ------------- Run fitting -------------
    def _run_fitting(self):
        try:
            excel_file = self.excel_path_var.get().strip()
            if not excel_file:
                self._show_message("Excel file required", True)
                return
            if not os.path.exists(excel_file):
                self._show_message("Excel file not found", True)
                return

            conc_values_raw = self.cv_entry_var.get().replace(",", " ").split()
            if not conc_values_raw:
                self._show_message("Concentration vector required", True)
                return
            try:
                conc_uM = [float(v) for v in conc_values_raw]
            except ValueError:
                self._show_message("Invalid concentration values", True)
                return
            conc_M = [
                v * 1e-6 for v in conc_uM
            ]  # convert to M for run_full_plate_fit external API

            display_val = self.assay_type_var.get()
            assay_type = next(
                (code for (label, code) in self.ASSAY_TYPES if label == display_val),
                display_val,
            )

            # Validate required parameters > 0 (basic sanity checks)
            def _ensure_positive(keys):
                for k in keys:
                    if k not in assay_params or assay_params[k] <= 0:
                        raise ValueError(f"Parameter '{k}' must be > 0")

            assay_params = {}
            if assay_type == "ida":
                assay_params = {
                    "Kd": self.param_vars["Kd"].get(),
                    "h0": self.param_vars["h0"].get(),
                    "d0": self.param_vars["d0"].get(),
                }
                _ensure_positive(["Kd", "h0", "d0"])
            elif assay_type == "dba_HtoD":
                assay_params = {"d0": self.param_vars["d0"].get()}
                _ensure_positive(["d0"])
            elif assay_type == "dba_DtoH":
                assay_params = {"h0": self.param_vars["h0"].get()}
                _ensure_positive(["h0"])
            elif assay_type == "gda":
                assay_params = {
                    "Kd": self.param_vars["Kd"].get(),
                    "h0": self.param_vars["h0"].get(),
                    "g0": self.param_vars["g0"].get(),
                }
                _ensure_positive(["Kd", "h0", "g0"])

            # Filtering params
            filter_params = None
            if self.enable_filter_var.get():
                filter_params = {
                    "rmse_factor": self.rmse_factor_var.get(),
                    "k_factor": self.k_factor_var.get(),
                    "r2_threshold": self.r2_threshold_var.get(),
                }
            other_fits_to_plot = (
                self.other_fits_var.get() if self.enable_filter_var.get() else 0
            )

            # Outlier params
            outlier_params = None
            if self.enable_outlier_var.get():
                outlier_params = {
                    "method": self.outlier_method_var.get().lower(),
                    "threshold": self.outlier_threshold_var.get(),
                    "max_exclude": self.outlier_max_exclude_var.get(),
                }

            bounds_file = (
                self.bounds_file_var.get().strip()
                if self.use_bounds_file_var.get()
                else None
            )

            save_plots = self.save_plots_var.get()
            display_plots = self.display_plots_var.get()
            save_results = self.save_results_var.get()
            plots_dir = self.plots_entry.get().strip() if save_plots else None
            results_dir = self.results_entry.get().strip() if save_results else None

            custom_x_label = (
                self.custom_x_label_text_var.get().strip()
                if self.custom_x_label_var.get()
                and self.custom_x_label_text_var.get().strip()
                else None
            )
            custom_plot_title = (
                self.custom_plot_title_text_var.get().strip()
                if self.custom_plot_title_var.get()
                and self.custom_plot_title_text_var.get().strip()
                else None
            )

            with ProgressWindow(
                self.root, "Full Plate Fitting", "Processing & fitting, please wait..."
            ):
                result = run_full_plate_fit(
                    excel_file,
                    conc_M,
                    assay_type,
                    assay_params,
                    number_of_fit_trials=self.fit_trials_var.get(),
                    parameter_bounds=None,
                    results_file_path=bounds_file,
                    save_plots=save_plots,
                    display_plots=display_plots,
                    save_results=save_results,
                    results_save_dir=results_dir,
                    plots_dir=plots_dir,
                    custom_x_label=custom_x_label,
                    custom_plot_title=custom_plot_title,
                    filter_params=filter_params,
                    other_fits_to_plot=other_fits_to_plot,
                    outlier_params=outlier_params,
                )

            if not result:
                self._show_message("No result returned", True)
            else:
                # Provide quick summary for sanity checking
                best = result.get("best_params_internal")
                if best is not None and len(best) >= 2:
                    I0 = best[0]
                    K_internal = best[1]
                    K_physical = K_internal / 1e6  # internal µM^-1 -> M^-1
                    self._show_message(
                        f"Done. I0={I0:.3g} AU, K={K_physical:.3g} M^-1", False
                    )
                else:
                    self._show_message("Full plate fitting completed", False)
        except Exception as e:
            self._show_message(f"Error: {e}\n{traceback.format_exc()}", True)

    # ------------- Helpers -------------
    def _show_message(self, msg, is_error=False):
        if self.info_label:
            self.info_label.destroy()
        fg = "red" if is_error else "green"
        self.info_label = tk.Label(
            self.root, text=msg, fg=fg, justify="left", wraplength=380
        )
        self.info_label.grid(
            column=0, row=999, columnspan=3, sticky=tk.W, padx=self.pad_x, pady=(8, 4)
        )


if __name__ == "__main__":
    root = tk.Tk()
    FullPlateFittingApp(root)
    root.mainloop()
