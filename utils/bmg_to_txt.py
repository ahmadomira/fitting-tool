#!/usr/bin/env python3

import csv
import re
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import pandas as pd

from . import plot_utils


def read_bmg_xlsx(excel_path: str) -> pd.DataFrame:
    """read an Excel file saved from the BMG plate reader and return the data and protocol info as dataframes"""

    sheets_dict = pd.read_excel(excel_path, sheet_name=None)
    dfs = {sheet_name: sheet_data for sheet_name, sheet_data in sheets_dict.items()}
    data = list(dfs.items())[0][1]
    protocol_info = list(dfs.items())[1][1]

    data = extract_bmg_raw_data(data)

    return data, protocol_info


def extract_bmg_raw_data(path_or_df) -> pd.DataFrame:
    """Extract wellplate data block (A-H, 1-12) from a BMG xlsx file or DataFrame, wide format."""
    if isinstance(path_or_df, (str, Path)):
        xls = pd.ExcelFile(path_or_df)
        # Try to find the correct sheet
        sheet = [s for s in xls.sheet_names if "end point" in s.lower()]
        if not sheet:
            raise ValueError("No 'End point' sheet found.")
        df_raw = xls.parse(sheet[0], header=None)
    else:
        df_raw = path_or_df

    # Find the first row with 'A' in the first column
    start_row = df_raw[df_raw.iloc[:, 0] == "A"].index[0]
    block = df_raw.iloc[start_row : start_row + 8, 0:13]
    block.columns = ["well_row"] + list(range(1, 13))
    block = block.set_index("well_row")
    block = block.apply(pd.to_numeric, errors="coerce")
    return block


def extract_concentration_vector(robot_file: str) -> list:
    """this is for a simple robot file with one analyte. Two sheets should Analyte"""

    sheet_names = ["Analyte (20)", "Analyte (300)"]

    try:
        concent_1 = (
            pd.read_excel(robot_file, sheet_name=sheet_names[0], header=None)
            .iloc[2, 1:13]
            .values
        )
        concent_2 = (
            pd.read_excel(robot_file, sheet_name=sheet_names[1], header=None)
            .iloc[2, 1:13]
            .values
        )
        return concent_1 + concent_2

    except ValueError as e:
        raise ValueError(
            f"Error reading concentration vector: The app expects two sheets in the robot file named 'Analyte (20)' and 'Analyte (300)' for reading the concentration values. Make sure the robot file is formatted accordingly. The full error is:\n\n {e}"
        ) from e


class ConcentrationVectorManager:
    def __init__(self):
        self.vectors = {}  # name: list of floats
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
        """Export vectors to CSV format"""
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
        """Import vectors from CSV format"""
        self.vectors = {}
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if row:  # Skip empty rows
                    name = row[0]
                    values = [
                        float(x) for x in row[1:] if x
                    ]  # Convert non-empty values to float
                    self.vectors[name] = values
        self.last_used = next(iter(self.vectors), None)

    def get_names(self):
        return list(self.vectors.keys())

    def get_vector(self, name):
        return self.vectors.get(name, [])


# Plotting functions (adapted from plot_replica.py)
def group_analytes(files: list, group_by="analyte") -> dict:
    """Group files by analyte or pH based on filename pattern."""
    if group_by == "analyte":
        pattern = re.compile(r"\d+_(.*?)_pH\d+\.xlsx", re.IGNORECASE)
    elif group_by == "pH":
        pattern = re.compile(r".*?_.*?_(pH\d+)\.xlsx", re.IGNORECASE)
    else:
        raise ValueError("group_by should be either 'analyte' or 'pH'")

    grouped_analytes = defaultdict(list)

    for file in files:
        match = re.search(pattern, str(file))
        if match:
            middle_string = match.group(1)
            grouped_analytes[middle_string].append(file)

    return dict(grouped_analytes)


def plot_single_file_replicas(
    file_path: str, concentration_vector: list, save_dir: str = None
):
    """Plot all replicas for a single file."""
    file_path = Path(file_path)

    plot_title = " ".join(file_path.stem.split("_")[1:])
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
    ax.plot(
        concentration_vector,
        data.values.T,
        label=[f"Replica {i+1}" for i in range(data.shape[0])],
    )
    ax.legend(loc="best")

    # Display the plot
    # plt.show()

    # Optionally save plot
    if save_dir:
        save_dir = Path(save_dir)
        output_file = save_dir / f"{file_path.stem}_all_replicas.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        return output_file

    return None


def plot_file_average(file_path: str, concentration_vector: list, save_dir: str = None):
    """Plot average of replicas for a single file."""
    file_path = Path(file_path)

    plot_title = " ".join(file_path.stem.split("_")[1:])
    data, info = read_bmg_xlsx(file_path)

    if len(concentration_vector) != data.shape[1]:
        raise ValueError(
            f"Concentration vector length ({len(concentration_vector)}) doesn't match data columns ({data.shape[1]})"
        )

    data_avg = data.mean(axis=0)

    fig, ax = plot_utils.create_plots(plot_title=f"{plot_title} (Average)")
    try:
        fig.canvas.manager.set_window_title(f"{plot_title} (Average)")
    except Exception:
        pass
    ax.plot(concentration_vector, data_avg.values, "o-", linewidth=2, markersize=6)

    # Display the plot
    # plt.show()

    # Optionally save plot
    if save_dir:
        save_dir = Path(save_dir)
        output_file = save_dir / f"{file_path.stem}_average.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        return output_file

    return None


def plot_grouped_files(
    files: list, concentration_vector: list, save_dir: str = None, group_by="analyte"
):
    """Plot grouped files (by analyte or pH)."""
    grouped = group_analytes(files, group_by=group_by)
    plot_files = []
    figures = []  # Collect figures to display all at once

    for group_name, group_files in grouped.items():
        plot_title = " ".join(group_name.split("_"))

        fig, ax = plot_utils.create_plots(plot_title=plot_title)
        try:
            fig.canvas.manager.set_window_title(plot_title)
        except Exception:
            pass

        for file_path in group_files:
            data, info = read_bmg_xlsx(file_path)
            if len(concentration_vector) != data.shape[1]:
                continue  # Skip files with mismatched concentration vectors

            data_avg = data.mean(axis=0)
            file_label = (
                Path(file_path).stem.split("_")[-1]
                if group_by == "analyte"
                else " ".join(Path(file_path).stem.split("_")[1:-1])
            )
            ax.plot(
                concentration_vector,
                data_avg.values,
                "o-",
                label=file_label,
                linewidth=2,
                markersize=4,
            )

        ax.legend()
        figures.append(fig)  # Collect figure instead of showing immediately

        # Optionally save plot
        if save_dir:
            save_dir = Path(save_dir)
            suffix = "over_pH" if group_by == "analyte" else "grouped_by_analyte"
            output_file = save_dir / f"{'_'.join(plot_title.split(' '))}_{suffix}.png"
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            plot_files.append(output_file)

    # Display all figures at once
    plt.show()

    return plot_files


class BMGToTxtConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("BMG Data Converter & Plotter")
        self.root.geometry("750x800")
        self.root.resizable(True, True)
        self.cv_manager = ConcentrationVectorManager()
        self.file_list = []

        # Configure custom ttk styles for colored labels
        self.style = ttk.Style()
        self.style.configure("Gray.TLabel", foreground="gray")
        self.style.configure("Blue.TLabel", foreground="blue")
        self.style.configure("Green.TLabel", foreground="green")
        self.style.configure("Red.TLabel", foreground="red")
        self.style.configure("Orange.TLabel", foreground="orange")

        self.mainframe = ttk.Frame(
            root, padding="8 8 8 8"
        )  # left, top, right, bottom padding
        self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.mainframe.columnconfigure(0, weight=1)
        # Remove the weight from file listbox to prevent excess expansion

        # File listbox and controls
        pad_listbox_x = 8
        pad_listbox_y = 8
        row = 0
        ttk.Label(
            self.mainframe,
            text="BMG (*.xlsx) files:",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=row, column=0, sticky=tk.W, pady=(0, pad_listbox_y))
        row += 1

        # Create a frame for file list and buttons
        file_frame = ttk.Frame(self.mainframe)
        file_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, pad_listbox_y))
        file_frame.columnconfigure(0, weight=1)
        file_frame.rowconfigure(0, weight=1)
        row += 1

        # Listbox with vertical scrollbar
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

        # Button stack using grid for vertical stretching
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
        ttk.Label(cv_frame, text="of").grid(row=0, column=2, sticky=tk.W, padx=(0, 3))
        self.cv_name_entry = ttk.Entry(cv_frame, width=12)
        self.cv_name_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 6))
        self.save_cv_btn = ttk.Button(
            cv_frame, text="Save", command=self.save_cv, width=8
        )
        self.save_cv_btn.grid(row=0, column=4, sticky=tk.W)

        # Format hint - more compact
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

        # Previously used vectors - all in one row
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
        # Add a gray hint below the dropdown
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

        # Plotting section
        plot_frame = ttk.LabelFrame(self.mainframe, padding="6")
        plot_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 0))
        plot_frame.columnconfigure(1, weight=1)
        # Add a bold, large label as the section title
        ttk.Label(
            plot_frame,
            text="Plot Raw Data",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 8))
        row += 1

        # Plot type selection with tooltips
        ttk.Label(plot_frame, text="Plot type:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 6)
        )
        self.plot_type_var = tk.StringVar(value="individual")
        plot_type_frame = ttk.Frame(plot_frame)
        plot_type_frame.grid(row=1, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=(0, 6))

        # Create radiobuttons with tooltips
        individual_rb = ttk.Radiobutton(
            plot_type_frame,
            text="Individual Replicas",
            variable=self.plot_type_var,
            value="individual",
        )
        individual_rb.grid(row=0, column=0, sticky=tk.W, padx=(0, 12))

        average_rb = ttk.Radiobutton(
            plot_type_frame,
            text="Average Replicas",
            variable=self.plot_type_var,
            value="average",
        )
        average_rb.grid(row=0, column=1, sticky=tk.W, padx=(0, 12))

        group_analyte_rb = ttk.Radiobutton(
            plot_type_frame,
            text="Group by Analyte",
            variable=self.plot_type_var,
            value="group_analyte",
        )
        group_analyte_rb.grid(row=0, column=2, sticky=tk.W, padx=(0, 12))

        group_ph_rb = ttk.Radiobutton(
            plot_type_frame,
            text="Group by pH",
            variable=self.plot_type_var,
            value="group_ph",
        )
        group_ph_rb.grid(row=0, column=3, sticky=tk.W)

        # Add tooltips for all radiobuttons
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

        create_tooltip(
            individual_rb,
            "Shows all replicas separately for each file\n(good for seeing measurement variability)",
        )
        create_tooltip(
            average_rb,
            "Shows the average of all replicas for each file\n(cleaner view for comparison)",
        )
        create_tooltip(
            group_analyte_rb,
            "Groups files by analyte name and shows pH comparison\n(requires naming: XX_CHEMOSENSOR_DYE_ANALYTE_pHY.xlsx)",
        )
        create_tooltip(
            group_ph_rb,
            "Groups files by pH and shows analyte comparison\n(requires naming: XX_CHEMOSENSOR_DYE_ANALYTE_pHY.xlsx)",
        )

        # File naming hint
        ttk.Label(
            plot_frame,
            text="For grouping: files must follow naming convention XX_CHEMOSENSOR_DYE_ANALYTE_pH\n(e.g., 01_ZY6_MDAP_Dopamine_pH7.xlsx)",
            style="Gray.TLabel",
            font=("TkDefaultFont", 11),
        ).grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=(4, 0), padx=(3, 0))

        # Save plots option and plot save directory
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

        # Hint for plotting
        ttk.Label(
            plot_frame,
            text="Plots will be displayed by default. Enable 'Save plots' to save as PNG files with 300 DPI.\nDefault saving location: in ./Plots (relative to the app's current app directory)",
            style="Gray.TLabel",
            font=("TkDefaultFont", 11),
        ).grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=(0, 0), padx=(3, 0))
        
        # Plot buttons
        plot_btn_frame = ttk.Frame(plot_frame)  # Use ttk.Frame for consistency
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

        # Convert Files section (moved below plotting)
        convert_frame = ttk.LabelFrame(
            self.mainframe, padding="6"
        )
        
        ttk.Label(
            convert_frame,
            text="Convert BMG Files to Text",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 8))
        
        convert_frame.grid(
            row=row, column=0, sticky=(tk.W, tk.E), pady=(0, pad_listbox_y)
        )
        convert_frame.columnconfigure(1, weight=1)
        row += 1

        # Save directory (moved into convert section)
        ttk.Label(convert_frame, text="Save directory:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 6), pady=(0, 0)
        )
        self.save_dir_entry = ttk.Entry(convert_frame)
        self.save_dir_entry.grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 6), pady=(0, 0)
        )
        self.save_dir_entry.insert(0, "./Converted Data")
        ttk.Button(
            convert_frame, text="Browse", command=self.select_directory, width=7
        ).grid(row=1, column=2, sticky=tk.W, pady=(0, 0))

        # Add a gray hint below the save directory entry about the default directory
        ttk.Label(
            convert_frame,
            text="Default: ./Converted Data (relative to the app's current directory)",
            style="Gray.TLabel",
            font=("TkDefaultFont", 11),
        ).grid(
            row=2,
            column=1,
            columnspan=2,
            sticky=tk.W,
            pady=(0, 0),
            padx=(3, 0),
        )

        # Export replicas option and Convert button
        self.export_replicas_var = tk.BooleanVar(value=False)
        self.export_replicas_check = ttk.Checkbutton(
            convert_frame,
            text="Export replicas in separate files as well",
            variable=self.export_replicas_var,
        )
        self.export_replicas_check.grid(row=3, column=1, sticky=tk.W, padx=(0, 0))

        ttk.Button(
            convert_frame,
            text="Convert Files",
            command=self.process_and_display_status,
        ).grid(row=4, column=1)

        # Status label
        self.status_label = ttk.Label(self.mainframe, text="Ready", style="Blue.TLabel")
        self.status_label.grid(row=row, column=0, pady=(5, 0))
        row += 1

        # Legacy: Extract from Robot File option
        self.use_robot_file_var = tk.BooleanVar(value=False)
        self.robot_file_path_var = tk.StringVar()

        def set_robot_row_state(enabled):
            if enabled:
                self.robot_file_entry.state(["!disabled"])
                self.robot_browse_btn.state(["!disabled"])
            else:
                self.robot_file_entry.state(["disabled"])
                self.robot_browse_btn.state(["disabled"])

        def browse_robot_file():
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Select Robot File",
                filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
            )
            if path:
                try:
                    values = extract_concentration_vector(path)
                    self.cv_entry.delete(0, tk.END)
                    self.cv_entry.insert(0, " ".join(str(v) for v in values))
                    self.cv_entry.state(["disabled"])
                    self.robot_file_path_var.set(path)
                    self.robot_file_entry.state(["!disabled"])
                    self.robot_file_entry.delete(0, tk.END)
                    self.robot_file_entry.insert(0, path)
                    self.update_status(
                        f"Concentrations loaded from robot file.", "green"
                    )
                except Exception as e:
                    self.update_status(f"Error: {str(e)}", "red")
                    self.use_robot_file_var.set(False)
                    self.cv_entry.state(["!disabled"])
                    self.robot_file_path_var.set("")
                    self.robot_file_entry.state(["!disabled"])
                    self.robot_file_entry.delete(0, tk.END)
            else:
                if not self.robot_file_path_var.get():
                    self.use_robot_file_var.set(False)

        def on_robot_file_toggle():
            enabled = self.use_robot_file_var.get()
            set_robot_row_state(enabled)
            if enabled:
                browse_robot_file()
            else:
                self.cv_entry.state(["!disabled"])
                self.robot_file_entry.delete(0, tk.END)
                # self.robot_file_path_var.set("")
                self.robot_file_entry.state(["disabled"])

        # Place robot file row at same level as Save button
        robot_check = ttk.Checkbutton(
            cv_frame,
            text="From Robot File:",
            variable=self.use_robot_file_var,
            command=on_robot_file_toggle,
        )
        robot_check.grid(row=2, column=0, sticky=tk.W, pady=(0, 0), padx=(0, 3))

        # ttk.Entry for file path
        self.robot_file_entry = ttk.Entry(
            cv_frame, textvariable=self.robot_file_path_var
        )
        self.robot_file_entry.grid(
            row=2,
            column=1,
            columnspan=3,
            sticky=(tk.W, tk.E),
            padx=(0, 6),
            pady=(0, 0),
        )

        # ttk.Button for browsing
        self.robot_browse_btn = ttk.Button(
            cv_frame, text="Browse", command=browse_robot_file, width=8
        )
        self.robot_browse_btn.grid(
            row=2, column=4, sticky=tk.W, padx=(0, 6), pady=(0, 0)
        )

        set_robot_row_state(False)

        # Add a gray hint below the robot file entry about required sheet names
        ttk.Label(
            cv_frame,
            text="Robot file must contain sheets named 'Analyte (20)' and 'Analyte (300)'",
            style="Gray.TLabel",
            font=("TkDefaultFont", 11),
        ).grid(
            row=3,
            column=1,
            columnspan=4,
            sticky=tk.W,
            pady=(0, pad_listbox_y),
            padx=(3, 0),
        )

    # --- File list management ---
    def add_files(self):
        files = filedialog.askopenfilenames(parent=self.root)
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
        """Helper method to update status with color"""
        style_map = {
            "blue": "Blue.TLabel",
            "green": "Green.TLabel",
            "red": "Red.TLabel",
            "orange": "Orange.TLabel",
        }
        style = style_map.get(color, "Blue.TLabel")
        self.status_label.config(text=message)
        self.status_label.configure(style=style)

    # --- Concentration vector management ---
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

    def select_directory(self):
        # Use cwd as initialdir if entry is default or empty, else use last used
        current = self.save_dir_entry.get().strip()

        if not current or current == "./Converted Data":
            initialdir = Path.cwd()
        else:
            initialdir = Path(current)
        directory = filedialog.askdirectory(parent=self.root, initialdir=initialdir)
        if directory:
            self.save_dir_entry.delete(0, tk.END)
            self.save_dir_entry.insert(0, directory)

    def select_plot_directory(self):
        # Use cwd as initialdir if entry is default or empty, else use last used
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
        """Enable/disable plot save directory controls based on checkbox."""
        enabled = self.save_plots_var.get()
        if enabled:
            self.plot_save_dir_entry.state(["!disabled"])
            self.plot_browse_btn.state(["!disabled"])
        else:
            self.plot_save_dir_entry.state(["disabled"])
            self.plot_browse_btn.state(["disabled"])

    def get_current_concentration_vector(self):
        """Get the current concentration vector from UI or robot file."""
        if self.use_robot_file_var.get():
            robot_file = self.robot_file_path_var.get().strip()
            if robot_file:
                try:
                    return extract_concentration_vector(robot_file)
                except Exception as e:
                    self.update_status(f"Error reading robot file: {str(e)}", "red")
                    return None
            else:
                self.update_status("Robot file not selected", "red")
                return None
        else:
            # Get from manual entry
            values_text = (
                self.cv_entry.get().replace(",", " ").replace("\t", " ").split()
            )
            try:
                return [float(v) for v in values_text]
            except Exception:
                self.update_status("Invalid concentration values", "red")
                return None

    def plot_selected_file(self):
        """Plot all currently selected files in the listbox."""
        selections = self.file_listbox.curselection()
        if not selections:
            self.update_status("Please select at least one file to plot", "red")
            return

        concentration_vector = self.get_current_concentration_vector()
        if not concentration_vector:
            return

        save_dir = None
        if self.save_plots_var.get():
            plot_save_dir = self.plot_save_dir_entry.get().strip() or "./Plots"
            Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
            save_dir = plot_save_dir

        plot_type = self.plot_type_var.get()
        if plot_type in ("group_analyte", "group_ph"):
            if len(selections) < 2:
                self.update_status(
                    "Grouping requires at least two files; select multiple files to group.",
                    "orange"
                )
                return
            files = [self.file_list[idx] for idx in selections]
            group_by = "analyte" if plot_type == "group_analyte" else "pH"
            plot_grouped_files(files, concentration_vector, save_dir, group_by)
            return

        plotted_count = 0
        for idx in selections:
            file_path = self.file_list[idx]
            try:
                self.update_status(f"Plotting {Path(file_path).name}...", "blue")
                if self.plot_type_var.get() == "individual":
                    output = plot_single_file_replicas(file_path, concentration_vector, save_dir)
                elif self.plot_type_var.get() == "average":
                    output = plot_file_average(file_path, concentration_vector, save_dir)
                plotted_count += 1
            except Exception as e:
                self.update_status(f"Error plotting {Path(file_path).name}: {e}", "red")

        # Show all generated figures after plotting all selections
        plt.show()

        if save_dir and plotted_count:
            self.update_status(f"Displayed and saved {plotted_count} plot(s)", "green")
        else:
            self.update_status("Plotting complete", "green")

    def plot_all_files(self):
        """Plot all files in the list according to the selected plot type."""
        if not self.file_list:
            self.update_status("Please select files to plot", "red")
            return

        concentration_vector = self.get_current_concentration_vector()
        if not concentration_vector:
            return

        # Determine save directory if saving is enabled
        save_dir = None
        if self.save_plots_var.get():
            plot_save_dir = self.plot_save_dir_entry.get().strip()
            if not plot_save_dir:
                plot_save_dir = "./Plots"
                self.plot_save_dir_entry.delete(0, tk.END)
                self.plot_save_dir_entry.insert(0, plot_save_dir)
            # Create plot directory if it doesn't exist
            Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
            save_dir = plot_save_dir

        plot_type = self.plot_type_var.get()
        plotted_count = 0

        try:
            if plot_type in ["individual", "average"]:
                # Plot each file individually
                for file_path in self.file_list:
                    self.update_status(f"Plotting {Path(file_path).name}...", "blue")
                    try:
                        if plot_type == "individual":
                            output_file = plot_single_file_replicas(
                                file_path, concentration_vector, save_dir
                            )
                        else:  # average
                            output_file = plot_file_average(
                                file_path, concentration_vector, save_dir
                            )
                        plotted_count += 1
                    except Exception as e:
                        self.update_status(
                            f"Error plotting {Path(file_path).name}: {str(e)}", "red"
                        )
                        continue

                if save_dir:
                    self.update_status(
                        f"Successfully displayed and saved {plotted_count} plots",
                        "green",
                    )
                else:
                    self.update_status(
                        f"Ready", "blue"
                    )

            elif plot_type in ["group_analyte", "group_ph"]:
                # Group files and plot
                group_by = "analyte" if plot_type == "group_analyte" else "pH"
                self.update_status(f"Plotting files grouped by {group_by}...", "blue")

                output_files = plot_grouped_files(
                    self.file_list, concentration_vector, save_dir, group_by
                )

                if save_dir:
                    self.update_status(
                        f"Successfully displayed and saved {len(output_files)} grouped plots",
                        "green",
                    )
                else:
                    self.update_status(
                        f"Ready", "blue"
                        "green",
                    )

        except Exception as e:
            self.update_status(f"Error during plotting: {str(e)}", "red")

    def process_and_display_status(self):
        try:
            if not self.file_list:
                self.update_status("Please select BMG files to process", "red")
                return
            name = self.cv_var.get()
            values = self.cv_manager.get_vector(name)
            if not values:
                self.update_status(
                    "Please select or create a concentration vector", "red"
                )
                return

            save_dir = self.save_dir_entry.get().strip()
            if not save_dir:
                save_dir = "./Converted Data"
                self.save_dir_entry.delete(0, tk.END)
                self.save_dir_entry.insert(0, save_dir)

            # Create directory if it doesn't exist
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            export_replicas = self.export_replicas_var.get()
            processed_count = 0

            for bmg_file in self.file_list:
                self.process_bmg_data(bmg_file, values, save_dir, export_replicas)
                processed_count += 1
                self.update_status(
                    f"Processing file {processed_count}/{len(self.file_list)}...",
                    "blue",
                )
                self.root.update()  # Update GUI during processing

            self.update_status(
                f"Processing complete! {processed_count} file(s) processed successfully.",
                "green",
            )
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")

    def process_bmg_data(
        self, bmg_file, concentration_vector, save_dir=None, export_replicas=False
    ):
        bmg_file = Path(bmg_file).resolve()
        output_dir = Path(save_dir) if save_dir else bmg_file.parent
        data, info = read_bmg_xlsx(bmg_file)
        if len(concentration_vector) != data.shape[1]:
            raise ValueError(
                f"Concentration vector length ({len(concentration_vector)}) does not match data columns ({data.shape[1]})"
            )
        data.columns = concentration_vector
        if export_replicas:
            self.save_replica_to_multiple_txt(data, output_dir, bmg_file.stem)
        self.merge_replica_to_single_txt(data, output_dir, bmg_file.stem)

    def save_replica_to_multiple_txt(
        self, data: pd.DataFrame, directory: Path, dir_name="for_mathematica"
    ):
        # save replica to multiple txt files

        directory = directory / dir_name
        directory.mkdir(parents=False, exist_ok=True)

        for i, index in enumerate(data.index):
            data.loc[[index]].T.to_csv(
                directory / f"Replica_{i}.txt",
                sep="\t",
                header=False,
                index=True,
                mode="w",
            )

    def merge_replica_to_single_txt(
        self, data: pd.DataFrame, directory: Path, file_name="merged_data.txt"
    ):
        # merge replica to a single txt file (in µM)
        data.rename(
            columns=lambda x: f"{x * 1e-6 :.2e}", inplace=True
        )  # convert to µM and apply sci. notation to concent. values
        output_file = (directory / file_name).with_suffix(".txt")
        with open(output_file, "w") as f:
            for index in data.index:
                data.loc[[index]].T.to_csv(
                    output_file,
                    sep="\t",
                    header=["signal"],
                    index_label=["var"],
                    mode="a",
                    lineterminator="\n",
                )


if __name__ == "__main__":
    root = tk.Tk()
    app = BMGToTxtConverter(root)
    root.mainloop()
