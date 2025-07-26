#!/usr/bin/env python3

import re
import tkinter as tk
import traceback
from collections import defaultdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

from . import plot_utils
from .bmg_to_txt import extract_concentration_vector, read_bmg_xlsx


def place_annotation_safely(
    ax, text, data_x, data_y, line2d=None, marker_size=12, margin=8, **annotate_kwargs
):
    """
    Place annotation box in a location that avoids overlap with legend, data points, and optionally a line.
    - data_x, data_y: arrays of data points (for scatter/markers)
    - line2d: optional, a matplotlib Line2D object to check for overlap with the line itself
    - marker_size: pixel size of marker to avoid (default 12)
    - margin: extra pixels to add around annotation box (default 8)
    """
    renderer = ax.figure.canvas.get_renderer()
    corners = [
        (0.01, 0.99, {"xycoords": "axes fraction", "va": "top", "ha": "left"}),
        (0.99, 0.99, {"xycoords": "axes fraction", "va": "top", "ha": "right"}),
        (0.01, 0.01, {"xycoords": "axes fraction", "va": "bottom", "ha": "left"}),
        (0.99, 0.01, {"xycoords": "axes fraction", "va": "bottom", "ha": "right"}),
    ]
    legend = ax.get_legend()
    legend_bbox = legend.get_window_extent(renderer) if legend else None
    # Data points as bboxes (expanded for marker size)
    data_disp = ax.transData.transform(np.column_stack([data_x, data_y]))
    data_bboxes = [
        Bbox.from_bounds(
            x - marker_size, y - marker_size, 2 * marker_size, 2 * marker_size
        )
        for x, y in data_disp
    ]
    # If a line is provided, rasterize it to points and add bboxes
    line_bboxes = []
    if line2d is not None:
        line_x, line_y = line2d.get_data()
        line_disp = ax.transData.transform(np.column_stack([line_x, line_y]))
        for x, y in line_disp:
            line_bboxes.append(Bbox.from_bounds(x - 2, y - 2, 4, 4))
    best = None
    min_overlap = float("inf")
    for x, y, opts in corners:
        ann = ax.annotate(
            text,
            (x, y),
            bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.8),
            annotation_clip=False,
            **opts,
            **annotate_kwargs,
        )
        ax.figure.canvas.draw()
        ann_bbox = ann.get_window_extent(renderer).expanded(1.05, 1.1).padded(margin)
        overlap = 0
        if legend_bbox and ann_bbox.overlaps(legend_bbox):
            overlap += 1e6
        for db in data_bboxes:
            if ann_bbox.overlaps(db):
                overlap += 1
        for lb in line_bboxes:
            if ann_bbox.overlaps(lb):
                overlap += 1
        if overlap == 0:
            return ann
        if overlap < min_overlap:
            min_overlap = overlap
            best = ann
        ann.remove()
    return best  # fallback: least overlap


def place_annotation_opposite_legend(ax, text, offset_frac=0.07, **annotate_kwargs):
    """
    Place annotation box in the corner diagonally opposite to the legend, with a margin from axes.
    This version determines the legend's actual position using its bounding box in axes fraction coordinates.
    """
    legend = ax.get_legend()
    if legend is not None:
        renderer = ax.figure.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer)
        inv = ax.transAxes.inverted()
        bbox_axes = inv.transform(bbox)
        # bbox_axes: [[x0, y0], [x1, y1]] in axes fraction
        x_c = (bbox_axes[0, 0] + bbox_axes[1, 0]) / 2
        y_c = (bbox_axes[0, 1] + bbox_axes[1, 1]) / 2
        # Determine which corner legend is closest to
        corners = {
            "upper right": (1, 1),
            "upper left": (0, 1),
            "lower left": (0, 0),
            "lower right": (1, 0),
        }
        dists = {
            k: (x_c - x0) ** 2 + (y_c - y0) ** 2 for k, (x0, y0) in corners.items()
        }
        legend_corner = min(dists, key=dists.get)
    else:
        legend_corner = "upper right"
    # Map to opposite corner
    opposite = {
        "upper right": (
            offset_frac,
            offset_frac,
            {"xycoords": "axes fraction", "va": "bottom", "ha": "left"},
        ),
        "upper left": (
            1 - offset_frac,
            offset_frac,
            {"xycoords": "axes fraction", "va": "bottom", "ha": "right"},
        ),
        "lower left": (
            1 - offset_frac,
            1 - offset_frac,
            {"xycoords": "axes fraction", "va": "top", "ha": "right"},
        ),
        "lower right": (
            offset_frac,
            1 - offset_frac,
            {"xycoords": "axes fraction", "va": "top", "ha": "left"},
        ),
    }
    x, y, opts = opposite.get(legend_corner, opposite["upper right"])
    return ax.annotate(
        text,
        (x, y),
        bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.8),
        annotation_clip=False,
        **opts,
        **annotate_kwargs,
    )


def place_annotation_best_corner(
    ax, text, data_x=None, data_y=None, offset_frac=0.07, **annotate_kwargs
):
    """
    Place annotation box in the best available corner (least overlap with legend and data).
    Tries all four corners, picks the one with least overlap.
    """
    renderer = ax.figure.canvas.get_renderer()
    corners = [
        (offset_frac, 1 - offset_frac, "top", "left"),  # upper left
        (1 - offset_frac, 1 - offset_frac, "top", "right"),  # upper right
        (offset_frac, offset_frac, "bottom", "left"),  # lower left
        (1 - offset_frac, offset_frac, "bottom", "right"),  # lower right
    ]
    legend = ax.get_legend()
    legend_bbox = legend.get_window_extent(renderer) if legend else None
    data_bboxes = []
    if data_x is not None and data_y is not None:
        data_disp = ax.transData.transform(np.column_stack([data_x, data_y]))
        data_bboxes = [Bbox.from_bounds(x - 8, y - 8, 16, 16) for x, y in data_disp]
    best = None
    min_overlap = float("inf")
    for x, y, va, ha in corners:
        ann = ax.annotate(
            text,
            (x, y),
            bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.8),
            annotation_clip=False,
            xycoords="axes fraction",
            va=va,
            ha=ha,
            **annotate_kwargs,
        )
        ax.figure.canvas.draw()
        ann_bbox = ann.get_window_extent(renderer)
        overlap = 0
        if legend_bbox and ann_bbox.overlaps(legend_bbox):
            overlap += 1e6
        for db in data_bboxes:
            if ann_bbox.overlaps(db):
                overlap += 1
        if overlap == 0:
            return ann
        if overlap < min_overlap:
            min_overlap = overlap
            best = ann
        ann.remove()
    return best  # fallback: least overlap


def plot_all_replica(raw_data_path: str, robot_file_path: str, save_dir: str):
    """this is for ONE excel file"""
    raw_data_path, robot_file_path, save_dir = map(
        Path, (raw_data_path, robot_file_path, save_dir)
    )

    plot_title = " ".join(raw_data_path.stem.split("_")[1:])

    concentration_vector = extract_concentration_vector(robot_file_path)
    data, info = read_bmg_xlsx(raw_data_path)

    fig, ax = plot_utils.create_plots(plot_title=plot_title)
    ax.plot(
        concentration_vector,
        data.values.T,
        label=[f"Replica {i+1}" for i in range(data.shape[0])],
    )
    ax.legend(loc="best")

    # Save plot directly to the selected folder (no subfolders)
    output_file = save_dir / f"{raw_data_path.stem}_all_replicas.png"
    fig.savefig(output_file, bbox_inches="tight")
    plt.close("all")


def group_analytes(files: list, group_by="analyte") -> dict:
    """this is for multiple excel files of multiple analytes.
    It groups the files by the middle part of the file name, and
    returns a dictionary of Path objects"""

    if group_by == "analyte":
        pattern = re.compile(r"\d+_(.*?)_pH\d+\.xlsx", re.IGNORECASE)
    elif group_by == "pH":
        pattern = re.compile(r".*?_.*?_(pH\d+)\.xlsx", re.IGNORECASE)
    else:
        raise ValueError("group_by should be either 'analyte' or 'pH'")

    grouped_analytes = defaultdict(list)

    for file in files:
        match = re.search(pattern, file)
        if match:
            middle_string = match.group(1)
            grouped_analytes[middle_string].append(file)

    return dict(grouped_analytes)


def plot_avg_over_replicas(
    files: list, robot_file_path: str, save_dir: str, plot_title: str
):
    files = list(map(Path, files))
    robot_file_path, save_dir = Path(robot_file_path), Path(save_dir)

    plot_title = " ".join(plot_title.split("_"))

    concentration_vector = extract_concentration_vector(robot_file_path)
    fig, ax = plot_utils.create_plots(plot_title=plot_title)

    for file_ph in files:
        data, info = read_bmg_xlsx(file_ph)
        data = data.mean(axis=0)
        ax.plot(concentration_vector, data.values, label=file_ph.stem.split("_")[-1])

    ax.legend()
    # Save plot directly to the selected folder (no subfolders)
    output_file = save_dir / f"{'_'.join(plot_title.split(' '))}_grouped_by_pH.png"
    fig.savefig(output_file, bbox_inches="tight")
    plt.close("all")


def plot_analytes_over_ph(
    files: list, robot_file_path: str, save_dir: str, plot_title: str
):
    files = list(map(Path, files))
    robot_file_path, save_dir = Path(robot_file_path), Path(save_dir)

    concentration_vector = extract_concentration_vector(robot_file_path)
    fig, ax = plot_utils.create_plots(plot_title=plot_title)

    for file_ph in files:
        data, info = read_bmg_xlsx(file_ph)
        data = data.mean(axis=0)
        ax.plot(
            concentration_vector,
            data.values,
            label=" ".join(file_ph.stem.split("_")[1:-1]),
        )

    ax.legend()
    if files:
        output_file = save_dir / f"{files[0].stem}_avg_over_pH.png"
        fig.savefig(output_file, bbox_inches="tight")
    plt.close("all")


class PlotReplica:
    def __init__(self, root):
        self.root = root
        self.files = []
        self.robot_file = ""
        self.save_dir = ""

        self.root.title("Plot Data")
        self.root.geometry("800x600")  # Make window wider and taller

        self.mainframe = ttk.Frame(root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        # Configure the column to expand and center content
        self.mainframe.columnconfigure(0, weight=1)

        # Stylish round info button with 'i' icon
        info_canvas = tk.Canvas(
            self.mainframe,
            width=32,
            height=32,
            highlightthickness=0,
            bg=self.root.cget("bg"),
        )
        # Draw a blue circle
        info_canvas.create_oval(4, 4, 28, 28, fill="#3498db", outline="")
        # Draw the 'i' in white, bold
        info_canvas.create_text(
            16, 16, text="i", fill="white", font=("TkDefaultFont", 16, "bold")
        )
        info_canvas.grid(row=0, column=0, sticky=(tk.W), padx=(0, 0), pady=(0, 0))
        info_canvas.bind("<Button-1>", lambda e: self.show_info())
        info_canvas.bind("<Enter>", lambda e: info_canvas.config(cursor="hand2"))
        info_canvas.bind("<Leave>", lambda e: info_canvas.config(cursor=""))

        self.files_button = tk.Button(
            self.mainframe, text="1. Select Excel Files", command=self.catch_files
        )
        self.robot_file_button = tk.Button(
            self.mainframe, text="2. Select Robot File", command=self.catch_robot_file
        )
        self.save_dir_button = tk.Button(
            self.mainframe, text="3. Save to Folder", command=self.catch_save_dir
        )

        self.files_button.grid(row=1, column=0, sticky=(tk.E, tk.W))
        self.robot_file_button.grid(row=2, column=0, sticky=(tk.E, tk.W))
        self.save_dir_button.grid(row=3, column=0, sticky=(tk.E, tk.W))

        # Show selected files/paths with color and bold headers
        self.selected_info = tk.Text(
            self.mainframe,
            height=4,
            width=80,  # Wider
            relief=tk.FLAT,
            bg=self.root.cget("bg"),
            borderwidth=0,
            highlightthickness=0,
        )
        self.selected_info.grid(row=4, column=0, sticky=(tk.W, tk.E))
        self.selected_info.tag_configure(
            "header", foreground="blue", font=("TkDefaultFont", 10, "bold")
        )
        self.selected_info.tag_configure("value", font=("TkDefaultFont", 10, "normal"))
        self.selected_info.config(state=tk.DISABLED)

        # Create a text widget for status messages with scrollbar
        self.status_frame = ttk.Frame(self.mainframe)
        self.status_frame.grid(
            row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        self.status_scrollbar = ttk.Scrollbar(self.status_frame)
        self.status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_text = tk.Text(
            self.status_frame,
            height=16,  # Make logging part longer
            width=80,  # Wider
            yscrollcommand=self.status_scrollbar.set,
        )
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.status_scrollbar.config(command=self.status_text.yview)

        self.status_text.insert(tk.END, "Ready to process files\n")
        self.status_text.config(state=tk.DISABLED)

        process_button = tk.Button(
            self.mainframe, text="Process", command=self.process_and_display_status
        )
        process_button.grid(row=6, column=0, columnspan=3, sticky=(tk.E, tk.W))

        # Remove extra expansion below the process button
        self.mainframe.rowconfigure(7, weight=0)
        self.mainframe.rowconfigure(8, weight=0)
        # Only expand the status frame (row 5) and mainframe (row 0)
        self.mainframe.rowconfigure(5, weight=1)
        self.mainframe.rowconfigure(6, weight=0)
        # Remove extra padding if any
        for child in self.mainframe.winfo_children():
            child.grid_configure(padx=10, pady=5)

        # Make the main window expandable only for main content
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

    def update_selected_label(self):
        self.selected_info.config(state=tk.NORMAL)
        self.selected_info.delete("1.0", tk.END)
        # Excel files
        self.selected_info.insert(tk.END, "Excel files: ", "header")
        if self.files:
            self.selected_info.insert(
                tk.END, f"{', '.join([Path(f).name for f in self.files])}\n", "value"
            )
        else:
            self.selected_info.insert(tk.END, "None\n", "value")
        # Robot file
        self.selected_info.insert(tk.END, "Robot file: ", "header")
        self.selected_info.insert(
            tk.END,
            f"{Path(self.robot_file).name if self.robot_file else 'None'}\n",
            "value",
        )
        # Save dir
        self.selected_info.insert(tk.END, "Save dir: ", "header")
        self.selected_info.insert(
            tk.END, f"{self.save_dir if self.save_dir else 'None'}\n", "value"
        )
        self.selected_info.config(state=tk.DISABLED)

    def catch_files(self):
        files = list(filedialog.askopenfilenames())
        if files:
            self.files = files
            self.update_status(f"Selected {len(files)} Excel files.", "blue")
            self.update_selected_label()
            # Try to set default save dir if not set
            if not self.save_dir:
                self.save_dir = str(Path(self.files[0]).parent)
                self.update_status(
                    f"Default save directory set to {self.save_dir}", "blue"
                )
                self.update_selected_label()
            # Try to set default robot file if not set
            if not self.robot_file:
                parent = Path(self.files[0]).parent
                robot_candidates = list(parent.glob("*robot*.xlsx")) + list(
                    parent.glob("*robot*.txt")
                )
                if robot_candidates:
                    self.robot_file = str(robot_candidates[0])
                    self.update_status(
                        f"Default robot file set to {self.robot_file}", "blue"
                    )
                    self.update_selected_label()
        else:
            self.update_status("No Excel files selected.", "red")
        # Optionally, auto-open robot file dialog if not set
        # if not self.robot_file:
        #     self.catch_robot_file()

    def catch_robot_file(self):
        robot_file = filedialog.askopenfilename()
        if robot_file:
            self.robot_file = robot_file
            self.update_status(f"Selected robot file: {Path(robot_file).name}", "blue")
            self.update_selected_label()
        else:
            self.update_status("No robot file selected.", "red")
        # Optionally, auto-open save dir dialog if not set
        # if not self.save_dir:
        #     self.catch_save_dir()

    def catch_save_dir(self):
        save_dir = filedialog.askdirectory()
        if save_dir:
            self.save_dir = save_dir
            self.update_status(f"Selected save directory: {save_dir}", "blue")
            self.update_selected_label()
        else:
            self.update_status("No save directory selected.", "red")

    def update_status(self, message, color="black"):
        """Update the status text widget with a new message"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        # Apply tag for color
        start = self.status_text.index("end-1c linestart")
        end = self.status_text.index("end-1c")
        self.status_text.tag_add(color, start, end)
        self.status_text.tag_config(color, foreground=color)
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        self.root.update()

    def process_and_display_status(self):
        """Process the files and display status messages"""
        if not self.files:
            self.update_status(
                "No Excel files selected. Please select files first.", "red"
            )
            return
        if not self.robot_file:
            self.update_status(
                "No robot file selected. Please select a robot file.", "red"
            )
            return
        if not self.save_dir:
            self.update_status(
                "No save directory selected. Please select a save directory.", "red"
            )
            return

        self.update_status("Processing started...", "blue")

        try:
            success, message = self.main_process(
                self.files, self.robot_file, self.save_dir
            )
            if success:
                self.update_status(f"Processing completed successfully!", "green")
                self.update_status(
                    f"Plots have been saved to: {self.save_dir}", "green"
                )
            else:
                self.update_status(f"Error encountered: {message}", "red")
        except Exception as e:
            self.update_status(f"Unexpected error: {str(e)}", "red")
            self.update_status(traceback.format_exc(), "red")

    def main_process(self, files, robot_file, save_dir):
        """Process all files and return success status and message"""
        # Process individual replica plots
        for file in files:
            try:
                self.update_status(f"Plotting replicas from {Path(file).name}...")
                plot_all_replica(file, robot_file, save_dir)
            except Exception as e:
                error_message = (
                    f"Error when plotting replica from {Path(file).name}: {str(e)}"
                )
                return False, error_message

        # Process grouped by analyte
        self.update_status("Grouping by analyte and plotting averages...")
        grouped_analytes = group_analytes(files, group_by="analyte")
        for experiment, files_analyte in grouped_analytes.items():
            try:
                self.update_status(f"Processing analyte: {experiment}")
                plot_avg_over_replicas(files_analyte, robot_file, save_dir, experiment)
            except Exception as e:
                error_message = f"Error when plotting average over replicas for {experiment}: {str(e)}"
                return False, error_message

        # Process grouped by pH
        self.update_status("Grouping by pH and plotting analytes...")
        grouped_ph = group_analytes(files, group_by="pH")
        for experiment, files_ph in grouped_ph.items():
            try:
                self.update_status(f"Processing pH: {experiment}")
                plot_analytes_over_ph(files_ph, robot_file, save_dir, experiment)
            except Exception as e:
                error_message = (
                    f"Error when plotting analytes over pH for {experiment}: {str(e)}"
                )
                return False, error_message

        return True, "All plots generated successfully"

    def show_info(self):
        info_text = (
            "This interface is for plotting data from multiple Excel files and saving them to a folder:\n\n"
            "1. Select your Excel files using the 'Select Files' button. \n"
            "2. Select the robot file using the 'Select Robot File' button. \n"
            "3. Use the 'Save to Folder' button to choose where to save your plots. \n"
            "6. Press 'Process Data'. \n \n"
            "Important Note: the Excel files should follow this naming convention: "
            "01_ZY6_MDAP_Norepinephrine_pH4.xlsx \n"
        )
        messagebox.showinfo("How to Use", info_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = PlotReplica(root)
    root.mainloop()
