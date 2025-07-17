import os
import tkinter as tk
from tkinter import filedialog

from core.fitting.gda_merge import run_gda_merge_fits
from core.progress_window import ProgressWindow


class GDAMergeFitsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GDA Merge Fits Interface")
        self.info_label = None

        # Variables
        self.results_dir_var = tk.StringVar()
        self.outlier_threshold_var = tk.DoubleVar()
        self.rmse_threshold_factor_var = tk.DoubleVar()
        self.kg_threshold_factor_var = tk.DoubleVar()
        self.save_plots_var = tk.BooleanVar()
        self.save_plots_entry_var = tk.StringVar()
        self.display_plots_var = tk.BooleanVar()
        self.save_results_var = tk.BooleanVar()
        self.save_results_entry_var = tk.StringVar()
        self.plot_title_var = tk.StringVar()

        # Set default values
        self.outlier_threshold_var.set(0.25)
        self.rmse_threshold_factor_var.set(3)
        self.kg_threshold_factor_var.set(3)
        self.save_plots_var.set(False)
        self.display_plots_var.set(True)
        self.save_results_var.set(False)

        # Padding
        pad_x = 10
        pad_y = 5

        # Widgets
        tk.Label(self.root, text="Results Directory:").grid(
            row=0, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.results_dir_entry = tk.Entry(
            self.root, textvariable=self.results_dir_var, width=40, justify="left"
        )
        self.results_dir_entry.grid(
            row=0, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.results_dir_entry),
        ).grid(row=0, column=2, padx=pad_x, pady=pad_y)

        tk.Label(self.root, text="Plot Title:").grid(
            row=1, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.plot_title_entry = tk.Entry(
            self.root, textvariable=self.plot_title_var, width=40, justify="left"
        )
        self.plot_title_entry.grid(row=1, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="Outlier Relative Threshold:").grid(
            row=2, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.outlier_threshold_entry = tk.Entry(
            self.root, textvariable=self.outlier_threshold_var, justify="left"
        )
        self.outlier_threshold_entry.grid(
            row=2, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )

        tk.Label(self.root, text="RMSE Threshold Factor:").grid(
            row=3, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.rmse_threshold_factor_entry = tk.Entry(
            self.root, textvariable=self.rmse_threshold_factor_var, justify="left"
        )
        self.rmse_threshold_factor_entry.grid(
            row=3, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )

        tk.Label(self.root, text="Kg Threshold Factor:").grid(
            row=4, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.kg_threshold_factor_entry = tk.Entry(
            self.root, textvariable=self.kg_threshold_factor_var, justify="left"
        )
        self.kg_threshold_factor_entry.grid(
            row=4, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )

        tk.Checkbutton(
            self.root,
            text="Save Plots To",
            variable=self.save_plots_var,
            command=self.update_save_plot_widgets,
        ).grid(row=5, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.plots_dir_entry = tk.Entry(
            self.root,
            textvariable=self.save_plots_entry_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.plots_dir_entry.grid(row=5, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.plots_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.plots_dir_entry),
            state=tk.DISABLED,
        )
        self.plots_dir_button.grid(row=5, column=2, padx=pad_x, pady=pad_y)

        self.save_plots_var.trace_add(
            "write", lambda *args: self.update_save_plot_widgets()
        )

        tk.Checkbutton(
            self.root,
            text="Save Results To",
            variable=self.save_results_var,
            command=self.update_save_results_widgets,
        ).grid(row=6, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.save_results_dir_entry = tk.Entry(
            self.root,
            textvariable=self.save_results_entry_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.save_results_dir_entry.grid(
            row=6, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.save_results_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.save_results_dir_entry),
            state=tk.DISABLED,
        )
        self.save_results_dir_button.grid(row=6, column=2, padx=pad_x, pady=pad_y)

        self.save_results_var.trace_add(
            "write", lambda *args: self.update_save_results_widgets()
        )

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
        tk.Button(self.root, text="Run Merge Fits", command=self.run_merge_fits).grid(
            row=8, column=0, columnspan=3, pady=10, padx=pad_x
        )

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

    def show_message(self, message, is_error=False):
        if self.info_label:
            self.info_label.destroy()
        fg_color = "red" if is_error else "green"
        self.info_label = tk.Label(self.root, text=message, fg=fg_color)
        self.info_label.grid(row=9, column=0, columnspan=3, pady=10)

    def run_merge_fits(self):
        try:
            results_dir = self.results_dir_var.get()
            outlier_relative_threshold = self.outlier_threshold_var.get()
            rmse_threshold_factor = self.rmse_threshold_factor_var.get()
            kg_threshold_factor = self.kg_threshold_factor_var.get()
            save_plots = self.save_plots_var.get()
            display_plots = self.display_plots_var.get()
            save_results = self.save_results_var.get()
            results_save_dir = self.save_results_entry_var.get()
            plot_title = self.plot_title_var.get()

            # Show a progress indicator
            with ProgressWindow(
                self.root,
                "Merging Fits in Progress",
                "Merging fits in progress, please wait...",
            ) as progress_window:
                # Run the GDA merge fits process
                run_gda_merge_fits(
                    results_dir,
                    outlier_relative_threshold,
                    rmse_threshold_factor,
                    kg_threshold_factor,
                    save_plots,
                    display_plots,
                    save_results,
                    results_save_dir,
                    plot_title,
                )
            self.show_message("Merging fits completed successfully.")

        except Exception as e:
            # Show error message
            self.show_message(f"Error: {e}", is_error=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = GDAMergeFitsApp(root)
    root.mainloop()
