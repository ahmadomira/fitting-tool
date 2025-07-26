#!/usr/bin/env python3

import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import pandas as pd


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


class BMGToTxtConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Convert BMG Data to Text Files")
        self.root.geometry("850x520")
        self.root.resizable(True, True)
        self.cv_manager = ConcentrationVectorManager()
        self.file_list = []

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
        tk.Label(
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

        tk.Label(cv_frame, text="Concentrations (µM):").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 3)
        )
        self.cv_entry = tk.Entry(cv_frame)
        self.cv_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 6))
        tk.Label(cv_frame, text="of").grid(row=0, column=2, sticky=tk.W, padx=(0, 3))
        self.cv_name_entry = tk.Entry(cv_frame, width=12)
        self.cv_name_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 6))
        self.save_cv_btn = tk.Button(
            cv_frame, text="Save", command=self.save_cv, width=8
        )
        self.save_cv_btn.grid(row=0, column=4, sticky=tk.W)

        # Format hint - more compact
        tk.Label(
            cv_frame,
            text="Values separated by commas or spaces (e.g., 1.0, 2.0, 3.0)",
            fg="gray",
            font=("TkDefaultFont", 11),
        ).grid(row=1, column=1, sticky=(tk.W), pady=(0, pad_listbox_y), padx=(3, 0))

        tk.Label(
            cv_frame,
            text="Compound Name",
            fg="gray",
            font=("TkDefaultFont", 11),
        ).grid(row=1, column=3, sticky=tk.W, pady=(0, pad_listbox_y), padx=(3, 0))

        row += 1

        # Previously used vectors - all in one row
        prev_frame = ttk.Frame(self.mainframe)
        prev_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, pad_listbox_y))
        prev_frame.columnconfigure(1, weight=1)
        row += 1

        tk.Label(prev_frame, text="Saved Concentration Sets:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6), pady=(0, pad_listbox_y)
        )
        self.cv_var = tk.StringVar()
        self.cv_dropdown = ttk.Combobox(
            prev_frame, textvariable=self.cv_var, state="readonly"
        )
        self.cv_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 6))
        self.cv_dropdown.bind("<<ComboboxSelected>>", self.select_cv)
        tk.Button(prev_frame, text="Import", command=self.import_vectors, width=5).grid(
            row=0,
            column=2,
            sticky=tk.W,
            padx=(0, 4),
        )
        tk.Button(prev_frame, text="Export", command=self.export_vectors, width=5).grid(
            row=0, column=3, sticky=tk.W
        )
        # Add a gray hint below the dropdown
        tk.Label(
            prev_frame,
            text="You can export concentration sets and re-import them in later sessions",
            fg="gray",
            font=("TkDefaultFont", 11),
        ).grid(
            row=1,
            column=1,
            columnspan=3,
            sticky=tk.W,
            pady=(0, pad_listbox_y),
            padx=(3, 0),
        )

        # Save directory with default
        save_frame = ttk.Frame(self.mainframe)
        save_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, pad_listbox_y))
        save_frame.columnconfigure(1, weight=1)
        row += 1

        tk.Label(save_frame, text="Save directory:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6), pady=(0, 0)
        )
        self.save_dir_entry = ttk.Entry(save_frame)
        self.save_dir_entry.grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 6), pady=(0, 0)
        )
        self.save_dir_entry.insert(0, "./Converted Data")
        ttk.Button(
            save_frame, text="Browse", command=self.select_directory, width=7
        ).grid(row=0, column=2, sticky=tk.W, pady=(0, 0))

        # Add a gray hint below the save directory entry about the default directory
        tk.Label(
            save_frame,
            text="Default: ./Converted Data (relative to the app's current directory)",
            fg="gray",
            font=("TkDefaultFont", 11),
        ).grid(
            row=1,
            column=1,
            columnspan=2,
            sticky=tk.W,
            pady=(0, pad_listbox_y),
            padx=(3, 0),
        )

        # Export replicas option
        self.export_replicas_var = tk.BooleanVar(value=False)
        self.export_replicas_check = ttk.Checkbutton(
            save_frame,
            text="Export replicas in separate files as well",
            variable=self.export_replicas_var,
        )
        self.export_replicas_check.grid(
            row=2, column=1, sticky=tk.W, pady=(0, pad_listbox_y)
        )
        row += 1

        # Status and process button in a frame
        bottom_frame = ttk.Frame(self.mainframe)
        bottom_frame.grid(row=row, column=0, pady=(5, 0))
        row += 1

        self.status_label = ttk.Label(bottom_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=0, column=0, pady=(0, 5))

        process_button = ttk.Button(
            bottom_frame,
            text="Process Files",
            command=self.process_and_display_status,
        )
        process_button.grid(row=1, column=0)

        # Legacy: Extract from Robot File option
        self.use_robot_file_var = tk.BooleanVar(value=False)
        self.robot_file_path_var = tk.StringVar()

        def set_robot_row_state(enabled):
            if enabled:
                self.robot_file_entry.config(state="normal")
                self.robot_browse_btn.config(state="normal")
            else:
                self.robot_file_entry.config(state="disabled")
                self.robot_browse_btn.config(state="disabled")

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
                    self.cv_entry.config(state="disabled")
                    self.robot_file_path_var.set(path)
                    self.robot_file_entry.config(state="normal")
                    self.robot_file_entry.delete(0, tk.END)
                    self.robot_file_entry.insert(0, path)
                    self.update_status(
                        f"Concentrations loaded from robot file.", "green"
                    )
                except Exception as e:
                    self.update_status(f"Error: {str(e)}", "red")
                    self.use_robot_file_var.set(False)
                    self.cv_entry.config(state="normal")
                    self.robot_file_path_var.set("")
                    self.robot_file_entry.config(state="normal")
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
                self.cv_entry.config(state="normal")
                self.robot_file_entry.delete(0, tk.END)
                # self.robot_file_path_var.set("")
                self.robot_file_entry.config(state="disabled")

        # Place robot file row at same level as Save button
        robot_check = tk.Checkbutton(
            cv_frame,
            text="From Robot File:",
            variable=self.use_robot_file_var,
            command=on_robot_file_toggle,
        )
        robot_check.grid(row=2, column=0, sticky=tk.W, pady=(0, 0), padx=(0, 3))

        # classic tk.Entry for file path
        self.robot_file_entry = tk.Entry(
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

        # classic tk.Button for browsing
        self.robot_browse_btn = tk.Button(
            cv_frame, text="Browse", command=browse_robot_file, width=8
        )
        self.robot_browse_btn.grid(
            row=2, column=4, sticky=tk.W, padx=(0, 6), pady=(0, 0)
        )

        set_robot_row_state(False)

        # Add a gray hint below the robot file entry about required sheet names
        tk.Label(
            cv_frame,
            text="Robot file must contain sheets named 'Analyte (20)' and 'Analyte (300)'",
            fg="gray",
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
        self.status_label.config(text=message, foreground=color)

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
