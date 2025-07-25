import os
import tkinter as tk
from tkinter import filedialog

from core.fitting.gda import run_gda_fitting
from core.progress_window import ProgressWindow


class GDAFittingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GDA Fitting Interface")
        self.info_label = None
        self.file_path_var = tk.StringVar()
        self.use_dye_alone_results = tk.BooleanVar()
        self.dye_alone_results_var = tk.StringVar()
        self.Kd_var = tk.DoubleVar()
        self.h0_var = tk.DoubleVar()
        self.g0_var = tk.DoubleVar()
        self.fit_trials_var = tk.IntVar()
        self.rmse_threshold_var = tk.DoubleVar()
        self.r2_threshold_var = tk.DoubleVar()
        self.save_plots_var = tk.BooleanVar()
        self.plots_dir_var = tk.StringVar()
        self.display_plots_var = tk.BooleanVar()
        self.save_results_var = tk.BooleanVar()
        self.results_dir_var = tk.StringVar()
        self.custom_x_label_var = tk.BooleanVar()
        self.custom_x_label_text_var = tk.StringVar()
        self.custom_plot_title_var = tk.BooleanVar()
        self.custom_plot_title_text_var = tk.StringVar()

        self.fit_trials_var.set(200)
        self.rmse_threshold_var.set(2)
        self.r2_threshold_var.set(0.9)
        self.display_plots_var.set(True)

        # # for testing
        # self.Kd_var.set(1.68e7)
        # self.h0_var.set(4.3e-6)
        # self.g0_var.set(6e-6)
        # self.fit_trials_var.set(10)
        # self.file_path_var.set("/Users/ahmadomira/git/App Test/gda-test/GDA_system.txt")
        # self.use_dye_alone_results.set(True)
        # self.save_plots_var.set(True)
        # self.save_results_var.set(True)

        # self.dye_alone_results_var.set(
        #     "/Users/ahmadomira/git/App Test/dye-alone-test/dye_alone_results.txt"
        # )
        # self.results_dir_var.set("/Users/ahmadomira/git/App Test/gda-test/")
        # self.plots_dir_var.set("/Users/ahmadomira/git/App Test/gda-test/")

        pad_x = 10
        pad_y = 5
        tk.Label(self.root, text="Input File Path:").grid(
            row=0, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.file_path_entry = tk.Entry(
            self.root, textvariable=self.file_path_var, width=40, justify="left"
        )
        self.file_path_entry.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        tk.Button(self.root, text="Browse", command=self.browse_input_file).grid(
            row=0, column=2, padx=pad_x, pady=pad_y
        )
        tk.Checkbutton(
            self.root,
            text="Read Boundaries from File: ",
            variable=self.use_dye_alone_results,
            command=self.update_use_results_widgets,
        ).grid(row=1, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.dye_alone_results_entry = tk.Entry(
            self.root,
            textvariable=self.dye_alone_results_var,
            width=40,
            justify="left",
            state=tk.DISABLED,
        )
        self.dye_alone_results_entry.grid(
            row=1, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.dye_alone_browse_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_file(self.dye_alone_results_entry),
            state=tk.DISABLED,
        )
        self.dye_alone_browse_button.grid(row=1, column=2, padx=pad_x, pady=pad_y)
        self.use_dye_alone_results.trace_add(
            "write", lambda *args: self.update_use_results_widgets()
        )
        tk.Label(self.root, text=r"Kₐ (M⁻¹):").grid(
            row=3, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.Kd_entry = tk.Entry(self.root, textvariable=self.Kd_var, justify="left")
        self.Kd_entry.grid(row=3, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        tk.Label(self.root, text="H₀ (M):").grid(
            row=4, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.h0_entry = tk.Entry(self.root, textvariable=self.h0_var, justify="left")
        self.h0_entry.grid(row=4, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        tk.Label(self.root, text="G₀ (M):").grid(
            row=5, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.g0_entry = tk.Entry(self.root, textvariable=self.g0_var, justify="left")
        self.g0_entry.grid(row=5, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        tk.Label(self.root, text="Number of Fit Trials:").grid(
            row=6, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.fit_trials_entry = tk.Entry(
            self.root, textvariable=self.fit_trials_var, justify="left"
        )
        self.fit_trials_entry.grid(row=6, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        tk.Label(self.root, text="RMSE Threshold Factor:").grid(
            row=7, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.rmse_threshold_entry = tk.Entry(
            self.root, textvariable=self.rmse_threshold_var, justify="left"
        )
        self.rmse_threshold_entry.grid(
            row=7, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        tk.Label(self.root, text="R² Threshold:").grid(
            row=8, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.r2_threshold_entry = tk.Entry(
            self.root, textvariable=self.r2_threshold_var, justify="left"
        )
        self.r2_threshold_entry.grid(
            row=8, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        tk.Checkbutton(
            self.root,
            text="Save Plots To",
            variable=self.save_plots_var,
            command=self.update_save_plot_widgets,
        ).grid(row=9, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.plots_dir_entry = tk.Entry(
            self.root,
            textvariable=self.plots_dir_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.plots_dir_entry.grid(row=9, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.plots_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.plots_dir_entry),
            state=tk.DISABLED,
        )
        self.plots_dir_button.grid(row=9, column=2, padx=pad_x, pady=pad_y)
        self.save_plots_var.trace_add(
            "write", lambda *args: self.update_save_plot_widgets()
        )
        tk.Checkbutton(
            self.root,
            text="Save Results To",
            variable=self.save_results_var,
            command=self.update_save_results_widgets,
        ).grid(row=10, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.results_save_dir_entry = tk.Entry(
            self.root,
            textvariable=self.results_dir_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.results_save_dir_entry.grid(
            row=10, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.results_save_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.results_save_dir_entry),
            state=tk.DISABLED,
        )
        self.results_save_dir_button.grid(row=10, column=2, padx=pad_x, pady=pad_y)
        self.save_results_var.trace_add(
            "write", lambda *args: self.update_save_results_widgets()
        )

        tk.Checkbutton(
            self.root,
            text="Custom Plot Title:",
            variable=self.custom_plot_title_var,
            command=self.update_custom_plot_title_widgets,
        ).grid(row=11, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.custom_plot_title_entry = tk.Entry(
            self.root,
            textvariable=self.custom_plot_title_text_var,
            width=30,
            state=tk.DISABLED,
            justify="left",
        )
        self.custom_plot_title_entry.grid(
            row=11, column=1, columnspan=2, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.custom_plot_title_var.trace_add(
            "write", lambda *args: self.update_custom_plot_title_widgets()
        )

        tk.Checkbutton(
            self.root,
            text="Custom X-axis Label:",
            variable=self.custom_x_label_var,
            command=self.update_custom_x_label_widgets,
        ).grid(row=12, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.custom_x_label_entry = tk.Entry(
            self.root,
            textvariable=self.custom_x_label_text_var,
            width=30,
            state=tk.DISABLED,
            justify="left",
        )
        self.custom_x_label_entry.grid(
            row=12, column=1, columnspan=2, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.custom_x_label_var.trace_add(
            "write", lambda *args: self.update_custom_x_label_widgets()
        )

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=13, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(
            row=14, column=0, columnspan=3, pady=10, padx=pad_x
        )
        self.root.lift()
        self.root.focus_force()

    def browse_input_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_var.set(file_path)
            root_dir = os.path.dirname(file_path)
            self.plots_dir_var.set(root_dir)
            self.results_dir_var.set(root_dir)

    def browse_file(self, entry):
        initial_dir = (
            os.path.dirname(self.file_path_var.get())
            if self.file_path_var.get()
            else os.getcwd()
        )
        file_path = filedialog.askopenfilename(initialdir=initial_dir)
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def browse_directory(self, entry):
        initial_dir = (
            os.path.dirname(self.file_path_var.get())
            if self.file_path_var.get()
            else os.getcwd()
        )
        directory_path = filedialog.askdirectory(initialdir=initial_dir)
        if directory_path:
            entry.delete(0, tk.END)
            entry.insert(0, directory_path)

    def update_use_results_widgets(self):
        state = tk.NORMAL if self.use_dye_alone_results.get() else tk.DISABLED
        self.dye_alone_results_entry.config(state=state)
        self.dye_alone_browse_button.config(state=state)

    def update_save_plot_widgets(self):
        state = tk.NORMAL if self.save_plots_var.get() else tk.DISABLED
        self.plots_dir_entry.config(state=state)
        self.plots_dir_button.config(state=state)

    def update_save_results_widgets(self):
        state = tk.NORMAL if self.save_results_var.get() else tk.DISABLED
        self.results_save_dir_entry.config(state=state)
        self.results_save_dir_button.config(state=state)

    def update_custom_x_label_widgets(self):
        state = tk.NORMAL if self.custom_x_label_var.get() else tk.DISABLED
        self.custom_x_label_entry.config(state=state)

    def update_custom_plot_title_widgets(self):
        state = tk.NORMAL if self.custom_plot_title_var.get() else tk.DISABLED
        self.custom_plot_title_entry.config(state=state)

    def show_message(self, message, is_error=False):
        if self.info_label:
            self.info_label.destroy()
        fg_color = "red" if is_error else "green"
        self.info_label = tk.Label(self.root, text=message, fg=fg_color)
        self.info_label.grid(row=15, column=0, columnspan=3, pady=10)

    def run_fitting(self):
        try:
            file_path = self.file_path_entry.get()
            dye_alone_results = (
                self.dye_alone_results_entry.get()
                if self.use_dye_alone_results.get()
                else None
            )
            Kd_in_M = self.Kd_var.get()
            h0_in_M = self.h0_var.get()
            g0_in_M = self.g0_var.get()
            number_of_fit_trials = self.fit_trials_var.get()
            rmse_threshold_factor = self.rmse_threshold_var.get()
            r2_threshold = self.r2_threshold_var.get()
            save_plots = self.save_plots_var.get()
            display_plots = self.display_plots_var.get()
            plots_dir = self.plots_dir_entry.get()
            save_results = self.save_results_var.get()
            results_save_dir = self.results_save_dir_entry.get()

            # Get custom x-label if enabled, otherwise None for automatic selection
            custom_x_label = (
                self.custom_x_label_text_var.get().strip()
                if self.custom_x_label_var.get()
                and self.custom_x_label_text_var.get().strip()
                else None
            )

            # Get custom plot title if enabled, otherwise None for automatic selection
            custom_plot_title = (
                self.custom_plot_title_text_var.get().strip()
                if self.custom_plot_title_var.get()
                and self.custom_plot_title_text_var.get().strip()
                else None
            )

            with ProgressWindow(
                self.root,
                "Fitting in Progress",
                "GDA fitting in progress, please wait...",
            ) as progress_window:
                run_gda_fitting(
                    file_path,
                    dye_alone_results,
                    Kd_in_M,
                    h0_in_M,
                    g0_in_M,
                    number_of_fit_trials,
                    rmse_threshold_factor,
                    r2_threshold,
                    save_plots,
                    display_plots,
                    plots_dir,
                    save_results,
                    results_save_dir,
                    custom_x_label=custom_x_label,
                    custom_plot_title=custom_plot_title,
                )
            self.show_message(f"Fitting completed!", is_error=False)
        except Exception as e:
            self.show_message(f"Error: {str(e)}", is_error=True)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Automation Project")
    GDAFittingApp(root)
    root.mainloop()
