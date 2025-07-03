import tkinter as tk

from core.fitting.dba_dye_to_host import run_dba_dye_to_host_fitting
from gui.base_gui import BaseAppGUI


class DBAFittingAppDtoH(BaseAppGUI):
    def __init__(self, root):
        super().__init__(root, title="DBA Dye-to-Host Fitting Interface")
        self.file_path_var = self.add_string_var("file_path", "")
        self.use_results_file_var = self.add_bool_var("use_results_file", False)
        self.results_file_path_var = self.add_string_var("results_file_path", "")
        self.h0_var = self.add_double_var("h0", 6e-6)
        self.fit_trials_var = self.add_int_var("fit_trials", 200)
        self.rmse_threshold_var = self.add_double_var("rmse_threshold", 2)
        self.r2_threshold_var = self.add_double_var("r2_threshold", 0.9)
        self.save_plots_var = self.add_bool_var("save_plots", False)
        self.results_dir_var = self.add_string_var("results_dir", "")
        self.display_plots_var = self.add_bool_var("display_plots", True)
        self.save_results_var = self.add_bool_var("save_results", False)
        self.results_save_dir_var = self.add_string_var("results_save_dir", "")

        pad_x = self.pad_x
        pad_y = self.pad_y

        tk.Label(self.root, text="Input File Path:").grid(
            row=0, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.file_path_entry = tk.Entry(
            self.root, textvariable=self.file_path_var, width=40, justify="left"
        )
        self.file_path_entry.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_file(self.file_path_var),
        ).grid(row=0, column=2, padx=pad_x, pady=pad_y)

        tk.Checkbutton(
            self.root,
            text="Read Boundaries from File: ",
            variable=self.use_results_file_var,
            command=self.update_use_results_widgets,
        ).grid(row=1, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.results_file_path_entry = tk.Entry(
            self.root,
            textvariable=self.results_file_path_var,
            width=40,
            justify="left",
            state=tk.DISABLED,
        )
        self.results_file_path_entry.grid(
            row=1, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.results_file_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_file(self.results_file_path_var),
            state=tk.DISABLED,
        )
        self.results_file_button.grid(row=1, column=2, padx=pad_x, pady=pad_y)
        self.use_results_file_var.trace_add(
            "write", lambda *args: self.update_use_results_widgets()
        )

        tk.Label(self.root, text="H₀ (M):").grid(
            row=3, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.h0_entry = tk.Entry(self.root, textvariable=self.h0_var, justify="left")
        self.h0_entry.grid(row=3, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="Number of Fit Trials:").grid(
            row=4, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.fit_trials_entry = tk.Entry(
            self.root, textvariable=self.fit_trials_var, justify="left"
        )
        self.fit_trials_entry.grid(row=4, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="RMSE Threshold Factor:").grid(
            row=5, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.rmse_threshold_entry = tk.Entry(
            self.root, textvariable=self.rmse_threshold_var, justify="left"
        )
        self.rmse_threshold_entry.grid(
            row=5, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )

        tk.Label(self.root, text="R² Threshold:").grid(
            row=6, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.r2_threshold_entry = tk.Entry(
            self.root, textvariable=self.r2_threshold_var, justify="left"
        )
        self.r2_threshold_entry.grid(
            row=6, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )

        tk.Checkbutton(
            self.root,
            text="Save Plots To",
            variable=self.save_plots_var,
            command=self.update_save_plot_widgets,
        ).grid(row=7, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.results_dir_entry = tk.Entry(
            self.root,
            textvariable=self.results_dir_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.results_dir_entry.grid(
            row=7, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.results_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.results_dir_var),
            state=tk.DISABLED,
        )
        self.results_dir_button.grid(row=7, column=2, padx=pad_x, pady=pad_y)
        self.save_plots_var.trace_add(
            "write", lambda *args: self.update_save_plot_widgets()
        )

        tk.Checkbutton(
            self.root,
            text="Save Results To",
            variable=self.save_results_var,
            command=self.update_save_results_widgets,
        ).grid(row=8, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.results_save_dir_entry = tk.Entry(
            self.root,
            textvariable=self.results_save_dir_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.results_save_dir_entry.grid(
            row=8, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.results_save_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.results_save_dir_var),
            state=tk.DISABLED,
        )
        self.results_save_dir_button.grid(row=8, column=2, padx=pad_x, pady=pad_y)
        self.save_results_var.trace_add(
            "write", lambda *args: self.update_save_results_widgets()
        )

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=9, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(
            row=10, column=0, columnspan=3, pady=10, padx=pad_x
        )
        self.lift_and_focus()

    def update_use_results_widgets(self):
        state = tk.NORMAL if self.use_results_file_var.get() else tk.DISABLED
        self.results_file_path_entry.config(state=state)
        self.results_file_button.config(state=state)

    def update_save_plot_widgets(self):
        state = tk.NORMAL if self.save_plots_var.get() else tk.DISABLED
        self.results_dir_entry.config(state=state)
        self.results_dir_button.config(state=state)

    def update_save_results_widgets(self):
        state = tk.NORMAL if self.save_results_var.get() else tk.DISABLED
        self.results_save_dir_entry.config(state=state)
        self.results_save_dir_button.config(state=state)

    def run_fitting(self):
        try:
            file_path = self.file_path_var.get()
            results_file_path = (
                self.results_file_path_var.get()
                if self.use_results_file_var.get()
                else None
            )
            h0_in_M = self.h0_var.get()
            rmse_threshold_factor = self.rmse_threshold_var.get()
            r2_threshold = self.r2_threshold_var.get()
            save_plots = self.save_plots_var.get()
            display_plots = self.display_plots_var.get()
            plots_dir = self.results_dir_var.get()
            save_results = self.save_results_var.get()
            results_save_dir = self.results_save_dir_var.get()
            number_of_fit_trials = self.fit_trials_var.get()
            run_dba_dye_to_host_fitting(
                file_path=file_path,
                results_file_path=results_file_path,
                h0_in_M=h0_in_M,
                rmse_threshold_factor=rmse_threshold_factor,
                r2_threshold=r2_threshold,
                save_plots=save_plots,
                display_plots=display_plots,
                plots_dir=plots_dir,
                save_results_bool=save_results,
                results_save_dir=results_save_dir,
                number_of_fit_trials=number_of_fit_trials,
            )
            self.show_message("Fitting complete!", is_error=False)
        except Exception as e:
            self.show_message(f"Error: {str(e)}", is_error=True)


if __name__ == "__main__":
    root = tk.Tk()
    DBAFittingAppDtoH(root)
    root.mainloop()
