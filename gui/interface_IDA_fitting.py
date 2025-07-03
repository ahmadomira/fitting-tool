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
        self.results_dir_entry = tk.Entry(
            self.root,
            textvariable=self.plots_dir_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.results_dir_entry.grid(
            row=9, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.results_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.plots_dir_var),
            state=tk.DISABLED,
        )
        self.results_dir_button.grid(row=9, column=2, padx=pad_x, pady=pad_y)
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
            command=lambda: self.browse_directory(self.results_dir_var),
            state=tk.DISABLED,
        )
        self.results_save_dir_button.grid(row=10, column=2, padx=pad_x, pady=pad_y)
        self.save_results_var.trace_add(
            "write", lambda *args: self.update_save_results_widgets()
        )

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=11, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(
            row=12, column=0, columnspan=3, pady=10, padx=pad_x
        )

        self.lift_and_focus()

    def update_use_results_widgets(self, *args):
        state = tk.NORMAL if self.use_results_file_var.get() else tk.DISABLED
        self.results_file_path_entry.config(state=state)
        self.results_file_button.config(state=state)

    def update_save_plot_widgets(self, *args):
        state = tk.NORMAL if self.save_plots_var.get() else tk.DISABLED
        self.results_dir_entry.config(state=state)
        self.results_dir_button.config(state=state)

    def update_save_results_widgets(self, *args):
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
