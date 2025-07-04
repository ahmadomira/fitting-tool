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

        self.file_path_entry, self.file_path_browse = self.add_file_selector(
            row=0, label_text="Input File Path:", var=self.file_path_var
        )

        self.results_file_path_entry, self.results_file_button = (
            self.add_toggleable_file_selector(
                row=1,
                label_text="Read Boundaries from File: ",
                bool_var=self.use_results_file_var,
                file_var=self.results_file_path_var,
            )
        )

        self.h0_entry = self.add_labeled_entry(
            row=3, label_text="H₀ (M):", var=self.h0_var
        )
        self.fit_trials_entry = self.add_labeled_entry(
            row=4, label_text="Number of Fit Trials:", var=self.fit_trials_var
        )
        self.rmse_threshold_entry = self.add_labeled_entry(
            row=5, label_text="RMSE Threshold Factor:", var=self.rmse_threshold_var
        )
        self.r2_threshold_entry = self.add_labeled_entry(
            row=6, label_text="R² Threshold:", var=self.r2_threshold_var
        )

        self.results_dir_entry, self.results_dir_button = (
            self.add_toggleable_dir_selector(
                row=7,
                label_text="Save Plots To",
                bool_var=self.save_plots_var,
                dir_var=self.results_dir_var,
                input_file_var=self.file_path_var,
            )
        )
        self.results_save_dir_entry, self.results_save_dir_button = (
            self.add_toggleable_dir_selector(
                row=8,
                label_text="Save Results To",
                bool_var=self.save_results_var,
                dir_var=self.results_save_dir_var,
                input_file_var=self.file_path_var,
            )
        )

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=9, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(
            row=10, column=0, columnspan=3, pady=10, padx=pad_x
        )
        self.lift_and_focus()

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
