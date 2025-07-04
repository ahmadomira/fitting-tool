import tkinter as tk
import traceback

from core.fitting.ida import run_ida_fitting
from gui.base_gui import BaseAppGUI


class IDAFittingApp(BaseAppGUI):
    def __init__(self, root):
        super().__init__(root, title="IDA Fitting Interface")
        # Variables
        self.file_path_var = self.add_string_var(
            "file_path", "/Users/ahmadomira/Downloads/interface_test/IDA_system.txt"
        )
        self.use_results_file_var = self.add_bool_var("use_results_file", False)
        self.results_file_path_var = self.add_string_var("results_file_path", "")
        self.Kd_var = self.add_double_var("Kd", 1.68e7)
        self.h0_var = self.add_double_var("h0", 4.3e-6)
        self.g0_var = self.add_double_var("g0", 6e-6)
        self.fit_trials_var = self.add_int_var("fit_trials", 10)
        self.rmse_threshold_var = self.add_double_var("rmse_threshold", 2)
        self.r2_threshold_var = self.add_double_var("r2_threshold", 0.9)
        self.save_plots_var = self.add_bool_var("save_plots", False)
        self.plots_dir_var = self.add_string_var(
            "plots_dir", "/Users/ahmadomira/Downloads/interface_test/untitled folder"
        )
        self.display_plots_var = self.add_bool_var("display_plots", True)
        self.save_results_var = self.add_bool_var("save_results", False)
        self.results_dir_var = self.add_string_var(
            "results_dir", "/Users/ahmadomira/Downloads/interface_test/untitled folder"
        )

        pad_x = self.pad_x
        pad_y = self.pad_y

        # Widgets
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

        self.Kd_entry = self.add_labeled_entry(
            row=3, label_text="Kₐ (M⁻¹):", var=self.Kd_var
        )
        self.h0_entry = self.add_labeled_entry(
            row=4, label_text="H₀ (M):", var=self.h0_var
        )
        self.g0_entry = self.add_labeled_entry(
            row=5, label_text="G₀ (M):", var=self.g0_var
        )
        self.fit_trials_entry = self.add_labeled_entry(
            row=6, label_text="Number of Fit Trials:", var=self.fit_trials_var
        )
        self.rmse_threshold_entry = self.add_labeled_entry(
            row=7, label_text="RMSE Threshold Factor:", var=self.rmse_threshold_var
        )
        self.r2_threshold_entry = self.add_labeled_entry(
            row=8, label_text="R² Threshold:", var=self.r2_threshold_var
        )

        self.results_dir_entry, self.results_dir_button = (
            self.add_toggleable_dir_selector(
                row=9,
                label_text="Save Plots To",
                bool_var=self.save_plots_var,
                dir_var=self.plots_dir_var,
                input_file_var=self.file_path_var,
            )
        )
        self.results_save_dir_entry, self.results_save_dir_button = (
            self.add_toggleable_dir_selector(
                row=10,
                label_text="Save Results To",
                bool_var=self.save_results_var,
                dir_var=self.results_dir_var,
                input_file_var=self.file_path_var,
            )
        )

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=11, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)

        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(
            row=12, column=0, columnspan=3, pady=10, padx=pad_x
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
            Kd_in_M = self.Kd_var.get()
            h0_in_M = self.h0_var.get()
            g0_in_M = self.g0_var.get()
            number_of_fit_trials = self.fit_trials_var.get()
            rmse_threshold_factor = self.rmse_threshold_var.get()
            r2_threshold = self.r2_threshold_var.get()
            save_plots = self.save_plots_var.get()
            display_plots = self.display_plots_var.get()
            plots_dir = self.plots_dir_var.get()
            save_results_bool = self.save_results_var.get()
            results_save_dir = self.results_dir_var.get()

            progress_window = tk.Toplevel(self.root)
            progress_window.title("Fitting in Progress")
            progress_label = tk.Label(
                progress_window, text="Fitting in progress, please wait..."
            )
            progress_label.pack(padx=20, pady=20)
            self.root.update_idletasks()

            run_ida_fitting(
                file_path,
                results_file_path,
                Kd_in_M,
                h0_in_M,
                g0_in_M,
                number_of_fit_trials,
                rmse_threshold_factor,
                r2_threshold,
                save_plots,
                display_plots,
                plots_dir,
                save_results_bool,
                results_save_dir,
            )

            progress_window.destroy()
            self.show_message(f"Fitting completed!", is_error=False, row=13)
        except Exception as e:
            error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.show_message(error_message, is_error=True, row=13)
            print(error_message)


if __name__ == "__main__":
    root = tk.Tk()
    IDAFittingApp(root)
    root.mainloop()
