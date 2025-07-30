#!/usr/bin/env python3

import csv
import tkinter as tk
import tkinter.colorchooser
import warnings
from pathlib import Path
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import pandas as pd

from . import plot_utils


def read_bmg_xlsx(excel_path: str):
    sheets_dict = pd.read_excel(excel_path, sheet_name=None)
    dfs = {sheet_name: sheet_data for sheet_name, sheet_data in sheets_dict.items()}
    data = list(dfs.items())[0][1]
    protocol_info = list(dfs.items())[1][1]
    data = extract_bmg_raw_data(data)
    return data, protocol_info


def extract_bmg_raw_data(path_or_df):
    if isinstance(path_or_df, (str, Path)):
        xls = pd.ExcelFile(path_or_df)
        sheet = [s for s in xls.sheet_names if "end point" in s.lower()]
        if not sheet:
            raise ValueError("No 'End point' sheet found.")
        df_raw = xls.parse(sheet[0], header=None)
    else:
        df_raw = path_or_df
    start_row = df_raw[df_raw.iloc[:, 0] == "A"].index[0]
    block = df_raw.iloc[start_row : start_row + 8, 0:13]
    block.columns = ["well_row"] + list(range(1, 13))
    block = block.set_index("well_row")
    block = block.apply(pd.to_numeric, errors="coerce")
    return block


class ConcentrationVectorManager:
    def __init__(self):
        self.vectors = {}
        self.last_used = None

    def add_vector(self, name, values):
        self.vectors[name] = values
        self.last_used = name

    def remove_vector(self, name):
        if name in self.vectors:
            del self.vectors[name]
            if self.last_used == name:
                self.last_used = next(iter(self.vectors), None)

    def export_vectors(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Name"]
                + [
                    f"Value_{i+1}"
                    for i in range(
                        max(len(v) for v in self.vectors.values())
                        if self.vectors
                        else 0
                    )
                ]
            )
            for name, values in self.vectors.items():
                writer.writerow([name] + values)

    def import_vectors(self, path):
        self.vectors = {}
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row:
                    name = row[0]
                    values = [float(x) for x in row[1:] if x]
                    self.vectors[name] = values
        self.last_used = next(iter(self.vectors), None)

    def get_names(self):
        return list(self.vectors.keys())

    def get_vector(self, name):
        return self.vectors.get(name, [])


def parse_annotations_sheet(excel_path):
    """
    Reads the 'annotations' sheet from the given Excel file and formats it as a string.
    Expects columns: Section, Parameter, Value (case-insensitive, flexible order).
    Returns formatted string or None if not found/invalid.
    """
    try:
        xls = pd.ExcelFile(excel_path)
        if "annotations" not in [s.lower() for s in xls.sheet_names]:
            return None
        # Find actual sheet name (case-insensitive)
        sheet_name = [s for s in xls.sheet_names if s.lower() == "annotations"][0]
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Normalize columns
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not {"section", "parameter", "value"}.issubset(df.columns):
            return None
        # Group by section
        annotation_lines = []
        for section, group in df.groupby("section"):
            annotation_lines.append(f"{section}:")
            for _, row in group.iterrows():
                param = str(row["parameter"]).strip()
                val = str(row["value"]).strip()
                if param:
                    annotation_lines.append(f"- {param}: {val}")
            annotation_lines.append("")  # Blank line between sections
        return "\n".join(annotation_lines).strip()
    except Exception as e:
        warnings.warn(f"Failed to parse annotations sheet: {e}")
        return None


def plot_single_file_replicas(
    file_path: str,
    concentration_vector: list,
    save_dir: str = None,
    annotation: str = None,
    plot_title: str = "Rhodamin",
):
    file_path = Path(file_path)
    # Try to parse per-file annotation sheet
    per_file_annotation = parse_annotations_sheet(file_path)
    if per_file_annotation:
        annotation = per_file_annotation
    data, info = read_bmg_xlsx(file_path)
    if len(concentration_vector) != data.shape[1]:
        raise ValueError(
            f"Concentration vector length ({len(concentration_vector)}) doesn't match data columns ({data.shape[1]})"
        )
    fig, ax = plot_utils.create_plots(plot_title=plot_title)
    try:
        fig.canvas.manager.set_window_title(plot_title)
    except Exception:
        pass
    min_vals = data.min(axis=0).values
    max_vals = data.max(axis=0).values
    avg_vals = data.mean(axis=0).values

    # Calculate STDs for steps 2-6 (5µL) and 7-12 (15µL)
    std_5ul = data.iloc[:, 1:6].values.flatten()  # steps 2-6 (0-based)
    std_15ul = data.iloc[:, 6:12].values.flatten()  # steps 7-12
    std_5ul_val = (
        float(pd.Series(std_5ul).std(ddof=1)) if std_5ul.size > 1 else float("nan")
    )
    std_15ul_val = (
        float(pd.Series(std_15ul).std(ddof=1)) if std_15ul.size > 1 else float("nan")
    )
    std_text = f"STD (steps 2-6, 5µL): {std_5ul_val:.4g}\nSTD (steps 7-12, 15µL): {std_15ul_val:.4g}\n\n"
    if annotation:
        annotation = std_text + annotation
    else:
        annotation = std_text.strip()

    lower_err = avg_vals - min_vals
    upper_err = max_vals - avg_vals
    yerr = [lower_err, upper_err]
    for i in range(data.shape[0]):
        ax.scatter(
            range(1, len(concentration_vector) + 1),
            data.iloc[i, :],
            color="gray",
            alpha=0.5,
            s=20,
        )
    ax.errorbar(
        range(1, len(concentration_vector) + 1),
        avg_vals,
        yerr=yerr,
        fmt="o-",
        color="red",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
        label="Mean ± Range",
        markersize=6,
    )
    ax.set_xlabel("Titration Steps")
    ax.set_ylabel("Absorbance [a.u.]")
    ax.legend(loc="best")
    ax.set_xticks(range(1, len(concentration_vector) + 1))
    if annotation:
        ax.annotate(
            annotation,
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            fontsize=8,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )
    if save_dir:
        save_dir = Path(save_dir)
        output_file = save_dir / f"{file_path.stem}_rhodamin_range.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        return output_file
    return None


class BMGToTxtConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("BMG Data Plotter")
        self.root.geometry("750x800")
        self.root.resizable(True, True)
        self.cv_manager = ConcentrationVectorManager()
        self.file_list = []
        self.style = ttk.Style()
        self.style.configure("Gray.TLabel", foreground="gray")
        self.style.configure("Blue.TLabel", foreground="blue")
        self.style.configure("Green.TLabel", foreground="green")
        self.style.configure("Red.TLabel", foreground="red")
        self.style.configure("Orange.TLabel", foreground="orange")
        self.mainframe = ttk.Frame(root, padding="8 8 8 8")
        self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.mainframe.columnconfigure(0, weight=1)
        pad_listbox_x = 8
        pad_listbox_y = 8
        row = 0
        ttk.Label(
            self.mainframe,
            text="BMG (*.xlsx) files:",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=row, column=0, sticky=tk.W, pady=(0, pad_listbox_y))
        row += 1
        file_frame = ttk.Frame(self.mainframe)
        file_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, pad_listbox_y))
        file_frame.columnconfigure(0, weight=1)
        file_frame.rowconfigure(0, weight=1)
        row += 1
        self.file_listbox = tk.Listbox(file_frame, selectmode=tk.EXTENDED, height=4)
        self.file_listbox.grid(
            row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(0, pad_listbox_x)
        )
        file_scrollbar = tk.Scrollbar(
            file_frame, orient="vertical", command=self.file_listbox.yview
        )
        file_scrollbar.grid(
            row=0,
            column=0,
            sticky=(tk.N, tk.S, tk.E),
            padx=(0, pad_listbox_x * 1.5),
            pady=pad_listbox_y / 2,
        )
        self.file_listbox.config(yscrollcommand=file_scrollbar.set, width=6, height=4)
        btn_frame = ttk.Frame(file_frame)
        btn_frame.grid(row=0, column=1, sticky=(tk.N))
        for i, (text, cmd) in enumerate(
            [
                ("Add", self.add_files),
                ("Remove", self.remove_files),
                ("Up", self.move_up),
                ("Down", self.move_down),
            ]
        ):
            b = ttk.Button(btn_frame, text=text, command=cmd, width=7)
            b.grid(row=i, column=0, sticky="ew", pady=1)
            btn_frame.rowconfigure(i, weight=1)
        btn_frame.columnconfigure(0, weight=1)
        cv_frame = ttk.Frame(self.mainframe)
        cv_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=pad_listbox_y)
        cv_frame.columnconfigure(1, weight=1)
        row += 1
        ttk.Label(cv_frame, text="Concentrations (µM):").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 3)
        )
        self.cv_entry = ttk.Entry(cv_frame)
        self.cv_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 6))
        self.cv_entry.insert(0, "1 2 3 4 5 6 7 8 9 10 11 12")
        ttk.Label(cv_frame, text="of").grid(row=0, column=2, sticky=tk.W, padx=(0, 3))
        self.cv_name_entry = ttk.Entry(cv_frame, width=12)
        self.cv_name_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 6))
        self.cv_name_entry.insert(0, "Steps")
        self.save_cv_btn = ttk.Button(
            cv_frame, text="Save", command=self.save_cv, width=8
        )
        self.save_cv_btn.grid(row=0, column=4, sticky=tk.W)
        ttk.Label(
            cv_frame,
            text="Values separated by commas or spaces (e.g., 1.0, 2.0, 3.0)",
            style="Gray.TLabel",
            font=("TkDefaultFont", 11),
        ).grid(row=1, column=1, sticky=(tk.W), pady=(0, pad_listbox_y), padx=(3, 0))
        ttk.Label(
            cv_frame,
            text="Compound Name",
            style="Gray.TLabel",
            font=("TkDefaultFont", 11),
        ).grid(row=1, column=3, sticky=tk.W, pady=(0, pad_listbox_y), padx=(3, 0))
        row += 1
        prev_frame = ttk.Frame(self.mainframe)
        prev_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, pad_listbox_y))
        prev_frame.columnconfigure(1, weight=1)
        row += 1
        ttk.Label(prev_frame, text="Saved Concentration Sets:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6), pady=(0, pad_listbox_y)
        )
        self.cv_var = tk.StringVar()
        self.cv_dropdown = ttk.Combobox(
            prev_frame, textvariable=self.cv_var, state="readonly"
        )
        self.cv_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 6))
        self.cv_dropdown.bind("<<ComboboxSelected>>", self.select_cv)
        ttk.Button(
            prev_frame, text="Import", command=self.import_vectors, width=5
        ).grid(
            row=0,
            column=2,
            sticky=tk.W,
            padx=(0, 4),
        )
        ttk.Button(
            prev_frame, text="Export", command=self.export_vectors, width=5
        ).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(
            prev_frame,
            text="You can export concentration sets and re-import them in later sessions",
            style="Gray.TLabel",
            font=("TkDefaultFont", 11),
        ).grid(
            row=1,
            column=1,
            columnspan=3,
            sticky=tk.W,
            pady=(0, pad_listbox_y),
            padx=(3, 0),
        )
        plot_frame = ttk.LabelFrame(self.mainframe, padding="6")
        plot_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 0))
        plot_frame.columnconfigure(1, weight=1)
        ttk.Label(
            plot_frame,
            text="Plot Raw Data",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 8))
        row += 1
        # Remove 'Plot Together' option
        ttk.Label(plot_frame, text="Plot type:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 6)
        )
        self.plot_type_var = tk.StringVar(value="separately")
        plot_type_frame = ttk.Frame(plot_frame)
        plot_type_frame.grid(
            row=1, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=(0, 6)
        )
        separately_rb = ttk.Radiobutton(
            plot_type_frame,
            text="Plot separately",
            variable=self.plot_type_var,
            value="separately",
            state="disabled",
        )
        separately_rb.grid(row=0, column=0, sticky=tk.W, padx=(0, 12))

        # Remove together_rb and related logic
        def create_tooltip(widget, text):
            def on_enter(event):
                tooltip = tk.Toplevel()
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                label = ttk.Label(
                    tooltip,
                    text=text,
                    background="lightyellow",
                    relief="solid",
                    borderwidth=1,
                    font=("TkDefaultFont", 10),
                )
                label.pack()
                widget.tooltip = tooltip

            def on_leave(event):
                if hasattr(widget, "tooltip"):
                    widget.tooltip.destroy()
                    del widget.tooltip

            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        create_tooltip(separately_rb, "Plot each selected file in a separate figure.")
        self.save_plots_var = tk.BooleanVar(value=False)
        self.save_plots_check = ttk.Checkbutton(
            plot_frame,
            text="Save plots to directory:",
            variable=self.save_plots_var,
            command=self.toggle_plot_save_controls,
        )
        self.save_plots_check.grid(
            row=3, column=0, sticky=tk.W, padx=(0, 6), pady=(8, 0)
        )
        self.plot_save_dir_entry = ttk.Entry(plot_frame)
        self.plot_save_dir_entry.insert(0, "./Plots")
        self.plot_save_dir_entry.state(["disabled"])
        self.plot_save_dir_entry.grid(
            row=3, column=1, sticky=(tk.W, tk.E), padx=(0, 6), pady=(8, 0)
        )
        self.plot_browse_btn = ttk.Button(
            plot_frame,
            text="Browse",
            command=self.select_plot_directory,
            width=7,
            state="disabled",
        )
        self.plot_browse_btn.grid(row=3, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Label(
            plot_frame,
            text="Plots will be displayed by default. Enable 'Save plots' to save as PNG files with 300 DPI.\nDefault saving location: in ./Plots (relative to the app's current app directory)",
            style="Gray.TLabel",
            font=("TkDefaultFont", 11),
        ).grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=(0, 0), padx=(3, 0))
        plot_btn_frame = ttk.Frame(plot_frame)
        plot_btn_frame.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=(8, 0))
        ttk.Button(
            plot_btn_frame,
            text="Plot Selected File(s)",
            command=self.plot_selected_file,
            width=15,
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Button(
            plot_btn_frame, text="Plot All Files", command=self.plot_all_files, width=12
        ).grid(row=0, column=1, sticky=tk.W)
        annotation_frame = ttk.Frame(plot_frame)
        annotation_frame.grid(
            row=6, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(8, 0)
        )
        ttk.Label(annotation_frame, text="Annotations:").grid(
            row=0, column=0, sticky=tk.NW
        )
        self.annotation_text = tk.Text(
            annotation_frame, width=60, height=10, wrap="word"
        )
        self.annotation_text.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(6, 0))
        annotation_frame.columnconfigure(1, weight=1)
        # Add annotation enable/disable checkbox
        self.enable_annotation_var = tk.BooleanVar(value=True)
        self.enable_annotation_check = ttk.Checkbutton(
            annotation_frame,
            text="Show Annotation",
            variable=self.enable_annotation_var,
        )
        self.enable_annotation_check.grid(row=1, column=1, sticky=tk.W, pady=(4, 0))
        # annotation_template = (
        #     "Aspirate:\n"
        #     "- rate (µL/s): \n"
        #     "- height (mm): \n"
        #     "- mixing (µL x rep.):\n"
        #     "- delay (s @ mm):\n"
        #     "- air gap (µL):\n"
        #     "- pre wet:\n\n"
        #     "Dispense:\n"
        #     "- rate (µL/s):\n"
        #     "- height (mm):\n"
        #     "- mixing (µL x rep.):\n"
        #     "- blowout (µL/s @ mm):\n"
        #     "- touch tip (@ mm):\n"
        # )
        annotation_template = ""
        self.annotation_text.insert("1.0", annotation_template)

        # Add plot title entry with toggle
        plot_title_frame = ttk.Frame(plot_frame)
        plot_title_frame.grid(
            row=7, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(8, 0)
        )
        self.use_custom_title_var = tk.BooleanVar(value=True)
        self.use_custom_title_check = ttk.Checkbutton(
            plot_title_frame,
            text="Custom Title:",
            variable=self.use_custom_title_var,
            command=self.toggle_plot_title_entry,
        )
        self.use_custom_title_check.grid(row=0, column=0, sticky=tk.W)
        self.plot_title_var = tk.StringVar(value="Rhodamin")
        self.plot_title_entry = ttk.Entry(
            plot_title_frame, textvariable=self.plot_title_var, width=30
        )
        self.plot_title_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(6, 0))
        plot_title_frame.columnconfigure(1, weight=1)

        # Add status label at the bottom
        self.status_label = ttk.Label(
            self.mainframe, text="Ready", style="Blue.TLabel", anchor="w"
        )
        self.status_label.grid(row=100, column=0, sticky=(tk.W, tk.E), pady=(12, 0))

        # Add reference plotting controls
        self.plot_reference_var = tk.BooleanVar(value=False)
        self.reference_file_var = tk.StringVar()
        reference_frame = ttk.Frame(plot_frame)
        reference_frame.grid(
            row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(4, 0)
        )
        self.plot_reference_var.set(True)
        self.plot_reference_check = ttk.Checkbutton(
            reference_frame,
            text="Plot Reference:",
            variable=self.plot_reference_var,
            command=self.toggle_reference_controls,
        )
        self.plot_reference_check.grid(row=0, column=0, sticky=tk.W)

        self.reference_file_var.set(
            "/Volumes/GRP_DHAC/SharedPhotophysics/Ahmad/Calibration SOP/calibration_tests/reference_hand_pippetting.xlsx"
        )
        self.reference_entry = ttk.Entry(
            reference_frame,
            textvariable=self.reference_file_var,
            width=40,
            state="normal",
        )
        self.reference_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(6, 0))
        self.reference_browse_btn = ttk.Button(
            reference_frame,
            text="Browse",
            command=self.select_reference_file,
            width=7,
            state="normal",
        )
        self.reference_browse_btn.grid(row=0, column=2, sticky=tk.W, padx=(6, 0))
        reference_frame.columnconfigure(1, weight=1)

        # Add reference color and opacity controls
        self.reference_color = "#09ff00"
        self.reference_opacity = tk.DoubleVar(value=0.5)
        color_opacity_frame = ttk.Frame(plot_frame)
        color_opacity_frame.grid(
            row=8, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(4, 0)
        )
        ttk.Label(color_opacity_frame, text="Reference Color:").grid(
            row=0, column=0, sticky=tk.W
        )
        self.reference_color_btn = ttk.Button(
            color_opacity_frame, text="Pick Color", command=self.pick_reference_color
        )
        self.reference_color_btn.grid(row=0, column=1, sticky=tk.W, padx=(6, 0))
        ttk.Label(color_opacity_frame, text="Opacity:").grid(
            row=0, column=2, sticky=tk.W, padx=(12, 0)
        )
        self.reference_opacity_slider = ttk.Scale(
            color_opacity_frame,
            from_=0.05,
            to=1.0,
            variable=self.reference_opacity,
            orient=tk.HORIZONTAL,
            length=100,
        )
        self.reference_opacity_slider.grid(row=0, column=3, sticky=tk.W, padx=(6, 0))

    def add_files(self):
        files = filedialog.askopenfilenames(
            parent=self.root,
            initialdir=Path(
                "/Volumes/GRP_DHAC/SharedPhotophysics/Ahmad/Calibration SOP/calibration_tests"
            ),
            title="Select BMG Files",
        )
        for f in files:
            if f not in self.file_list:
                self.file_list.append(f)
                self.file_listbox.insert(tk.END, f)

    def remove_files(self):
        selected = list(self.file_listbox.curselection())[::-1]
        for idx in selected:
            self.file_listbox.delete(idx)
            del self.file_list[idx]

    def move_up(self):
        selected = list(self.file_listbox.curselection())
        for idx in selected:
            if idx > 0:
                self.file_list[idx - 1], self.file_list[idx] = (
                    self.file_list[idx],
                    self.file_list[idx - 1],
                )
                self.file_listbox.delete(idx - 1, idx)
                self.file_listbox.insert(idx - 1, self.file_list[idx - 1])
                self.file_listbox.insert(idx, self.file_list[idx])
                self.file_listbox.selection_set(idx - 1)

    def move_down(self):
        selected = list(self.file_listbox.curselection())[::-1]
        for idx in selected:
            if idx < len(self.file_list) - 1:
                self.file_list[idx + 1], self.file_list[idx] = (
                    self.file_list[idx],
                    self.file_list[idx + 1],
                )
                self.file_listbox.delete(idx, idx + 1)
                self.file_listbox.insert(idx, self.file_list[idx])
                self.file_listbox.insert(idx + 1, self.file_list[idx + 1])
                self.file_listbox.selection_set(idx + 1)

    def update_status(self, message, color="blue"):
        style_map = {
            "blue": "Blue.TLabel",
            "green": "Green.TLabel",
            "red": "Red.TLabel",
            "orange": "Orange.TLabel",
        }
        style = style_map.get(color, "Blue.TLabel")
        self.status_label.config(text=message)
        self.status_label.configure(style=style)

    def save_cv(self):
        name = self.cv_name_entry.get().strip()
        values = self.cv_entry.get().replace(",", " ").replace("\t", " ").split()
        try:
            values = [float(v) for v in values]
        except Exception:
            self.update_status(
                "Invalid concentration values - please enter numbers only", "red"
            )
            return
        if not name:
            self.update_status(
                "Please provide a compound name for concentration values", "red"
            )
            return
        if not values:
            self.update_status("Please enter concentration values", "red")
            return
        self.cv_manager.add_vector(name, values)
        self.update_cv_dropdown()
        self.cv_var.set(name)
        self.update_status(f"Vector '{name}' saved successfully", "green")

    def update_cv_dropdown(self):
        names = self.cv_manager.get_names()
        self.cv_dropdown["values"] = names
        if self.cv_manager.last_used:
            self.cv_var.set(self.cv_manager.last_used)

    def select_cv(self, event=None):
        name = self.cv_var.get()
        values = self.cv_manager.get_vector(name)
        self.cv_entry.delete(0, tk.END)
        self.cv_entry.insert(0, " ".join(str(v) for v in values))
        self.cv_name_entry.delete(0, tk.END)
        self.cv_name_entry.insert(0, name)

    def export_vectors(self):
        if not self.cv_manager.vectors:
            self.update_status("No vectors to export", "orange")
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if path:
            try:
                self.cv_manager.export_vectors(path)
                self.update_status(f"Vectors exported to {path}", "green")
            except Exception as e:
                self.update_status(f"Error exporting vectors: {str(e)}", "red")

    def import_vectors(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if path:
            try:
                self.cv_manager.import_vectors(path)
                self.update_cv_dropdown()
                self.select_cv()
                self.update_status(f"Vectors imported from {path}", "green")
            except Exception as e:
                self.update_status(f"Error importing vectors: {str(e)}", "red")

    def select_plot_directory(self):
        current = self.plot_save_dir_entry.get().strip()
        if not current or current == "./Plots":
            initialdir = Path.cwd()
        else:
            initialdir = Path(current)
        directory = filedialog.askdirectory(parent=self.root, initialdir=initialdir)
        if directory:
            self.plot_save_dir_entry.delete(0, tk.END)
            self.plot_save_dir_entry.insert(0, directory)

    def toggle_plot_save_controls(self):
        enabled = self.save_plots_var.get()
        if enabled:
            self.plot_save_dir_entry.state(["!disabled"])
            self.plot_browse_btn.state(["!disabled"])
        else:
            self.plot_save_dir_entry.state(["disabled"])
            self.plot_browse_btn.state(["disabled"])

    def toggle_reference_controls(self):
        enabled = self.plot_reference_var.get()
        if enabled:
            self.reference_entry.state(["!disabled"])
            self.reference_browse_btn.state(["!disabled"])
        else:
            self.reference_entry.state(["disabled"])
            self.reference_browse_btn.state(["disabled"])

    def select_reference_file(self):
        file_path = filedialog.askopenfilename(
            parent=self.root,
            initialdir=Path(
                "/Volumes/GRP_DHAC/SharedPhotophysics/Ahmad/Calibration SOP/calibration_tests"
            ),
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Select Reference File",
        )
        if file_path:
            self.reference_file_var.set(file_path)

    def pick_reference_color(self):
        color = tkinter.colorchooser.askcolor(
            color=self.reference_color, title="Pick Reference Color"
        )
        if color and color[1]:
            self.reference_color = color[1]

    def get_current_concentration_vector(self):
        values_text = self.cv_entry.get().replace(",", " ").replace("\t", " ").split()
        try:
            return [float(v) for v in values_text]
        except Exception:
            self.update_status("Invalid concentration values", "red")
            return None

    def toggle_plot_title_entry(self):
        if self.use_custom_title_var.get():
            self.plot_title_entry.state(["!disabled"])
        else:
            self.plot_title_entry.state(["disabled"])

    def sanitize_title_from_filename(self, file_path):
        import os

        name = os.path.basename(file_path)
        name = os.path.splitext(name)[0]
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        return name

    def plot_selected_file(self):
        selections = self.file_listbox.curselection()
        if not selections:
            self.update_status("Please select at least one file to plot", "red")
            return
        concentration_vector = self.get_current_concentration_vector()
        if concentration_vector is None:
            return
        save_dir = None
        if self.save_plots_var.get():
            plot_save_dir = self.plot_save_dir_entry.get().strip() or "./Plots"
            Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
            save_dir = plot_save_dir
        annotation = self.annotation_text.get("1.0", tk.END).strip()
        plotted_count = 0
        plot_reference = self.plot_reference_var.get()
        reference_file = (
            self.reference_file_var.get().strip() if plot_reference else None
        )
        reference_data = None
        if plot_reference and reference_file:
            try:
                reference_data, _ = read_bmg_xlsx(reference_file)
            except Exception as e:
                self.update_status(f"Error reading reference file: {e}", "red")
                reference_data = None
        for idx in selections:
            file_path = self.file_list[idx]
            # Determine plot title
            if self.use_custom_title_var.get():
                plot_title = self.plot_title_var.get().strip() or "Rhodamin"
            else:
                plot_title = self.sanitize_title_from_filename(file_path)
            try:
                self.update_status(f"Plotting {Path(file_path).name}...", "blue")
                fig, ax = plot_utils.create_plots(plot_title=plot_title)
                try:
                    fig.canvas.manager.set_window_title(plot_title)
                except Exception:
                    pass
                # Plot reference if available (drawn first)
                if reference_data is not None:
                    min_vals = reference_data.min(axis=0).values
                    max_vals = reference_data.max(axis=0).values
                    avg_vals = reference_data.mean(axis=0).values
                    lower_err = avg_vals - min_vals
                    upper_err = max_vals - avg_vals
                    yerr = [lower_err, upper_err]
                    for i in range(reference_data.shape[0]):
                        ax.scatter(
                            range(1, len(concentration_vector) + 1),
                            reference_data.iloc[i, :],
                            color=self.reference_color,
                            alpha=self.reference_opacity.get(),
                            s=20,
                        )
                    ax.errorbar(
                        range(1, len(concentration_vector) + 1),
                        avg_vals,
                        yerr=yerr,
                        fmt="o-",
                        color=self.reference_color,
                        ecolor=self.reference_color,
                        elinewidth=1.5,
                        capsize=4,
                        label="Reference Mean ± Range",
                        markersize=6,
                        alpha=self.reference_opacity.get(),
                    )
                self.plot_main_on_axes(
                    file_path,
                    concentration_vector,
                    save_dir,
                    annotation=annotation,
                    plot_title=plot_title,
                    fig=fig,
                    ax=ax,
                    reference_data=reference_data,
                )
                plotted_count += 1
            except Exception as e:
                self.update_status(f"Error plotting {Path(file_path).name}: {e}", "red")
        plt.show()
        if save_dir and plotted_count:
            self.update_status(f"Displayed and saved {plotted_count} plot(s)", "green")
        else:
            self.update_status("Plotting complete", "green")

    def plot_all_files(self):
        if not self.file_list:
            self.update_status("Please select files to plot", "red")
            return
        concentration_vector = self.get_current_concentration_vector()
        if not concentration_vector:
            return
        save_dir = None
        if self.save_plots_var.get():
            plot_save_dir = self.plot_save_dir_entry.get().strip()
            if not plot_save_dir:
                plot_save_dir = "./Plots"
                self.plot_save_dir_entry.delete(0, tk.END)
                self.plot_save_dir_entry.insert(0, plot_save_dir)
            Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
            save_dir = plot_save_dir
        annotation = self.annotation_text.get("1.0", tk.END).strip()
        plot_reference = self.plot_reference_var.get()
        reference_file = (
            self.reference_file_var.get().strip() if plot_reference else None
        )
        reference_data = None
        if plot_reference and reference_file:
            try:
                reference_data, _ = read_bmg_xlsx(reference_file)
            except Exception as e:
                self.update_status(f"Error reading reference file: {e}", "red")
                reference_data = None
        plotted_count = 0
        for file_path in self.file_list:
            # Determine plot title
            if self.use_custom_title_var.get():
                plot_title = self.plot_title_var.get().strip() or "Rhodamin"
            else:
                plot_title = self.sanitize_title_from_filename(file_path)
            self.update_status(f"Plotting {Path(file_path).name}...", "blue")
            try:
                fig, ax = plot_utils.create_plots(plot_title=plot_title)
                try:
                    fig.canvas.manager.set_window_title(plot_title)
                except Exception:
                    pass
                if reference_data is not None:
                    min_vals = reference_data.min(axis=0).values
                    max_vals = reference_data.max(axis=0).values
                    avg_vals = reference_data.mean(axis=0).values
                    lower_err = avg_vals - min_vals
                    upper_err = max_vals - avg_vals
                    yerr = [lower_err, upper_err]
                    for i in range(reference_data.shape[0]):
                        ax.scatter(
                            range(1, len(concentration_vector) + 1),
                            reference_data.iloc[i, :],
                            color=self.reference_color,
                            alpha=self.reference_opacity.get(),
                            s=20,
                        )
                    ax.errorbar(
                        range(1, len(concentration_vector) + 1),
                        avg_vals,
                        yerr=yerr,
                        fmt="o-",
                        color=self.reference_color,
                        ecolor=self.reference_color,
                        elinewidth=1.5,
                        capsize=4,
                        label="Reference Mean ± Range",
                        markersize=6,
                        alpha=self.reference_opacity.get(),
                    )
                self.plot_main_on_axes(
                    file_path,
                    concentration_vector,
                    save_dir,
                    annotation=annotation,
                    plot_title=plot_title,
                    fig=fig,
                    ax=ax,
                    reference_data=reference_data,
                )
                plotted_count += 1
            except Exception as e:
                self.update_status(
                    f"Error plotting {Path(file_path).name}: {str(e)}", "red"
                )
                continue
        plt.show()
        if save_dir:
            self.update_status(
                f"Successfully displayed and saved {plotted_count} plots",
                "green",
            )
        else:
            self.update_status(f"Ready", "blue")

    def plot_main_on_axes(
        self,
        file_path,
        concentration_vector,
        save_dir,
        annotation,
        plot_title,
        fig,
        ax,
        reference_data=None,
    ):
        # This is a copy of plot_single_file_replicas, but draws on provided fig, ax
        file_path = Path(file_path)
        per_file_annotation = parse_annotations_sheet(file_path)
        if per_file_annotation:
            annotation = per_file_annotation
        data, info = read_bmg_xlsx(file_path)
        if len(concentration_vector) != data.shape[1]:
            raise ValueError(
                f"Concentration vector length ({len(concentration_vector)}) doesn't match data columns ({data.shape[1]})"
            )
        min_vals = data.min(axis=0).values
        max_vals = data.max(axis=0).values
        avg_vals = data.mean(axis=0).values
        std_5ul = data.iloc[:, 1:6].values.flatten()
        std_15ul = data.iloc[:, 6:12].values.flatten()
        std_5ul_val = (
            float(pd.Series(std_5ul).std(ddof=1)) if std_5ul.size > 1 else float("nan")
        )
        std_15ul_val = (
            float(pd.Series(std_15ul).std(ddof=1))
            if std_15ul.size > 1
            else float("nan")
        )
        std_text = f"STD (steps 2-6, 5µL): {std_5ul_val:.4g}\nSTD (steps 7-12, 15µL): {std_15ul_val:.4g}"
        # If reference_data is provided, compute its STDs
        ref_std_text = ""
        if reference_data is not None:
            ref_std_5ul = reference_data.iloc[:, 1:6].values.flatten()
            ref_std_15ul = reference_data.iloc[:, 6:12].values.flatten()
            ref_std_5ul_val = (
                float(pd.Series(ref_std_5ul).std(ddof=1))
                if ref_std_5ul.size > 1
                else float("nan")
            )
            ref_std_15ul_val = (
                float(pd.Series(ref_std_15ul).std(ddof=1))
                if ref_std_15ul.size > 1
                else float("nan")
            )
            ref_std_text = f"\nRef. STD (steps 2-6, 5µL): {ref_std_5ul_val:.4g}\nRef. STD (steps 7-12, 15µL): {ref_std_15ul_val:.4g}"
        stds_combined = std_text + ref_std_text
        # Only show annotation if enabled
        if not self.enable_annotation_var.get():
            annotation = stds_combined
        else:
            if annotation:
                annotation = stds_combined + "\n\n" + annotation
            else:
                annotation = stds_combined
        lower_err = avg_vals - min_vals
        upper_err = max_vals - avg_vals
        yerr = [lower_err, upper_err]
        for i in range(data.shape[0]):
            ax.scatter(
                range(1, len(concentration_vector) + 1),
                data.iloc[i, :],
                color="gray",
                alpha=0.5,
                s=20,
            )
        ax.errorbar(
            range(1, len(concentration_vector) + 1),
            avg_vals,
            yerr=yerr,
            fmt="o-",
            color="red",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            label="Mean ± Range",
            markersize=6,
        )
        ax.set_xlabel("Titration Steps")
        ax.set_ylabel("Absorbance [a.u.]")
        ax.legend(loc="best")
        ax.set_xticks(range(1, len(concentration_vector) + 1))
        if annotation:
            ax.annotate(
                annotation,
                xy=(0.03, 0.97),
                xycoords="axes fraction",
                fontsize=8,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
            )
        if save_dir:
            save_dir = Path(save_dir)
            output_file = save_dir / f"{file_path.stem}_rhodamin_range.png"
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            return output_file
        return None
