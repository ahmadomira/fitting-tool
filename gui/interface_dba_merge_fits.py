import os
import tkinter as tk
from tkinter import filedialog

from core.fitting.dba_merge import run_dba_merge_fits
from core.progress_window import ProgressWindow


class DBAMergeFitsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DBA Merge Fits Interface")
        self.info_label = None

        # Variables
        self.results_dir_var = tk.StringVar()
        self.assay_type_var = tk.StringVar()  # New variable for assay type
        self.outlier_threshold_var = tk.DoubleVar()
        self.rmse_threshold_factor_var = tk.DoubleVar()
        self.kd_threshold_factor_var = tk.DoubleVar()
        self.save_plots_var = tk.BooleanVar()
        self.save_plots_entry_var = tk.StringVar()
        self.display_plots_var = tk.BooleanVar()
        self.save_results_var = tk.BooleanVar()
        self.save_results_entry_var = tk.StringVar()
        self.plot_title_var = tk.StringVar()
        self.custom_plot_title_var = tk.BooleanVar()
        self.custom_plot_title_text_var = tk.StringVar()
        self.custom_x_label_var = tk.BooleanVar()
        self.custom_x_label_text_var = tk.StringVar()

        # Set default values
        self.assay_type_var.set("dba_HtoD")  # Default to Host-to-Dye
        self.outlier_threshold_var.set(0.25)
        self.rmse_threshold_factor_var.set(3)
        self.kd_threshold_factor_var.set(3)
        self.save_plots_var.set(False)
        self.display_plots_var.set(True)
        self.save_results_var.set(False)

        # # for testing
        # self.results_dir_var.set("/Users/ahmadomira/git/App Test/dba-h2d-test")
        # self.plot_title_var.set("DBA H2D Merged Fits")

        # self.save_plots_var.set(True)
        # self.display_plots_var.set(True)
        # self.save_results_var.set(True)

        # self.save_results_entry_var.set(self.results_dir_var.get())
        # self.save_plots_entry_var.set(self.results_dir_var.get())

        # Padding
        pad_x = 10
        pad_y = 5

        # Row counter for easier layout management
        row = 0

        # Widgets
        tk.Label(self.root, text="Results Directory:").grid(
            row=row, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.results_dir_entry = tk.Entry(
            self.root, textvariable=self.results_dir_var, width=40, justify="left"
        )
        self.results_dir_entry.grid(
            row=row, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.results_dir_entry),
        ).grid(row=row, column=2, padx=pad_x, pady=pad_y)
        row += 1

        # Assay Type Selection
        tk.Label(self.root, text="Assay Type:").grid(
            row=row, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )

        # Create a frame to hold the radio buttons horizontally
        assay_type_frame = tk.Frame(self.root)
        assay_type_frame.grid(row=row, column=1, sticky=tk.W, padx=pad_x, pady=pad_y)

        self.host_to_dye_radio = tk.Radiobutton(
            assay_type_frame,
            text="Host to Dye",
            variable=self.assay_type_var,
            value="dba_HtoD",
        )
        self.host_to_dye_radio.pack(side=tk.LEFT)

        self.dye_to_host_radio = tk.Radiobutton(
            assay_type_frame,
            text="Dye to Host",
            variable=self.assay_type_var,
            value="dba_DtoH",
        )

        self.dye_to_host_radio.pack(side=tk.LEFT, padx=(20, 0))
        row += 1

        tk.Label(self.root, text="Outlier Relative Threshold:").grid(
            row=row, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.outlier_threshold_entry = tk.Entry(
            self.root, textvariable=self.outlier_threshold_var, justify="left"
        )
        self.outlier_threshold_entry.grid(
            row=row, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        row += 1

        tk.Label(self.root, text="RMSE Threshold Factor:").grid(
            row=row, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.rmse_threshold_factor_entry = tk.Entry(
            self.root, textvariable=self.rmse_threshold_factor_var, justify="left"
        )
        self.rmse_threshold_factor_entry.grid(
            row=row, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        row += 1

        tk.Label(self.root, text="Kd Threshold Factor:").grid(
            row=row, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.kd_threshold_factor_entry = tk.Entry(
            self.root, textvariable=self.kd_threshold_factor_var, justify="left"
        )
        self.kd_threshold_factor_entry.grid(
            row=row, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        row += 1

        tk.Checkbutton(
            self.root,
            text="Save Plots To",
            variable=self.save_plots_var,
            command=self.update_save_plot_widgets,
        ).grid(row=row, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.plots_dir_entry = tk.Entry(
            self.root,
            textvariable=self.save_plots_entry_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.plots_dir_entry.grid(
            row=row, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.plots_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.plots_dir_entry),
            state=tk.DISABLED,
        )
        self.plots_dir_button.grid(row=row, column=2, padx=pad_x, pady=pad_y)

        self.save_plots_var.trace_add(
            "write", lambda *args: self.update_save_plot_widgets()
        )
        row += 1

        tk.Checkbutton(
            self.root,
            text="Save Results To",
            variable=self.save_results_var,
            command=self.update_save_results_widgets,
        ).grid(row=row, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.save_results_dir_entry = tk.Entry(
            self.root,
            textvariable=self.save_results_entry_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.save_results_dir_entry.grid(
            row=row, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.save_results_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.save_results_dir_entry),
            state=tk.DISABLED,
        )
        self.save_results_dir_button.grid(row=row, column=2, padx=pad_x, pady=pad_y)

        self.save_results_var.trace_add(
            "write", lambda *args: self.update_save_results_widgets()
        )
        row += 1

        # Custom plot title checkbox and text field
        self.custom_plot_title_checkbox = tk.Checkbutton(
            self.root,
            text="Custom Plot Title:",
            variable=self.custom_plot_title_var,
            command=self.update_custom_plot_title_widgets,
        )
        self.custom_plot_title_checkbox.grid(
            row=row, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.custom_plot_title_text_entry = tk.Entry(
            self.root,
            textvariable=self.custom_plot_title_text_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.custom_plot_title_text_entry.grid(
            row=row, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        row += 1

        tk.Checkbutton(
            self.root,
            text="Custom X-axis Label:",
            variable=self.custom_x_label_var,
            command=self.update_custom_x_label_widgets,
        ).grid(row=row, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.custom_x_label_entry = tk.Entry(
            self.root,
            textvariable=self.custom_x_label_text_var,
            width=30,
            state=tk.DISABLED,
            justify="left",
        )
        self.custom_x_label_entry.grid(
            row=row, column=1, columnspan=2, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.custom_x_label_var.trace_add(
            "write", lambda *args: self.update_custom_x_label_widgets()
        )
        row += 1

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
        row += 1

        tk.Button(self.root, text="Run Merge Fits", command=self.run_merge_fits).grid(
            row=row, column=0, columnspan=3, pady=10, padx=pad_x
        )
        row += 1

        # Store row counter for use in show_message
        self.next_row = row

    def browse_directory(self, entry):
        initial_dir = (
            os.path.dirname(self.results_dir_var.get())
            if self.results_dir_var.get()
            else os.getcwd()
        )
        directory_path = filedialog.askdirectory(
            initialdir=initial_dir, title="Select Directory"
        )
        if directory_path:
            entry.delete(0, tk.END)
            entry.insert(0, directory_path)

    def update_save_plot_widgets(self):
        state = tk.NORMAL if self.save_plots_var.get() else tk.DISABLED
        self.plots_dir_entry.config(state=state)
        self.plots_dir_button.config(state=state)
        self.save_plots_entry_var.set(self.results_dir_var.get())

    def update_save_results_widgets(self):
        state = tk.NORMAL if self.save_results_var.get() else tk.DISABLED
        self.save_results_dir_entry.config(state=state)
        self.save_results_dir_button.config(state=state)
        self.save_results_entry_var.set(self.results_dir_var.get())

    def update_custom_x_label_widgets(self):
        state = tk.NORMAL if self.custom_x_label_var.get() else tk.DISABLED
        self.custom_x_label_entry.config(state=state)

    def update_custom_plot_title_widgets(self):
        state = tk.NORMAL if self.custom_plot_title_var.get() else tk.DISABLED
        self.custom_plot_title_text_entry.config(state=state)

    def show_message(self, message, is_error=False):
        if self.info_label:
            self.info_label.destroy()
        fg_color = "red" if is_error else "green"
        self.info_label = tk.Label(self.root, text=message, fg=fg_color)
        self.info_label.grid(row=self.next_row, column=0, columnspan=3, pady=10)

    def run_merge_fits(self):
        try:
            results_dir = self.results_dir_var.get()
            assay_type = self.assay_type_var.get()
            outlier_relative_threshold = self.outlier_threshold_var.get()
            rmse_threshold_factor = self.rmse_threshold_factor_var.get()
            kd_threshold_factor = self.kd_threshold_factor_var.get()
            save_plots = self.save_plots_var.get()
            display_plots = self.display_plots_var.get()
            save_results = self.save_results_var.get()
            results_save_dir = self.results_dir_entry.get()
            custom_plot_title = (
                self.custom_plot_title_text_var.get()
                if self.custom_plot_title_var.get()
                else None
            )
            custom_x_label = (
                self.custom_x_label_text_var.get()
                if self.custom_x_label_var.get()
                else None
            )

            # Show a progress indicator
            with ProgressWindow(
                self.root,
                "Merging Fits in Progress",
                "Merging fits in progress, please wait...",
            ) as progress_window:
                # Call the function to merge fits with assay_type parameter
                run_dba_merge_fits(
                    results_dir,
                    outlier_relative_threshold,
                    rmse_threshold_factor,
                    kd_threshold_factor,
                    save_plots,
                    display_plots,
                    save_results,
                    results_save_dir,
                    custom_plot_title,
                    assay_type,
                    custom_x_label,
                )
            self.show_message("Merging fits completed!", is_error=False)
        except Exception as e:
            self.show_message(f"Error: {str(e)}", is_error=True)


if __name__ == "__main__":
    root = tk.Tk()
    DBAMergeFitsApp(root)
    root.mainloop()
