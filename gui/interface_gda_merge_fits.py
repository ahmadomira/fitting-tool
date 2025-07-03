import tkinter as tk

from core.fitting.gda_merge import run_gda_merge_fits
from gui.base_gui import BaseAppGUI


class GDAMergeFitsApp(BaseAppGUI):
    def __init__(self, root):
        super().__init__(root, title="GDA Merge Fits Interface")
        self.results_dir_var = self.add_string_var("results_dir", "")
        self.outlier_threshold_var = self.add_double_var("outlier_threshold", 0.25)
        self.rmse_threshold_factor_var = self.add_double_var("rmse_threshold_factor", 3)
        self.kg_threshold_factor_var = self.add_double_var("kg_threshold_factor", 3)
        self.save_plots_var = self.add_bool_var("save_plots", False)
        self.save_plots_entry_var = self.add_string_var("save_plots_entry", "")
        self.display_plots_var = self.add_bool_var("display_plots", True)
        self.save_results_var = self.add_bool_var("save_results", False)
        self.save_results_entry_var = self.add_string_var("save_results_entry", "")
        self.plot_title_var = self.add_string_var("plot_title", "")

        pad_x = self.pad_x
        pad_y = self.pad_y

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
            command=lambda: self.browse_directory(self.results_dir_var),
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
        ).grid(row=5, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.save_plots_entry = tk.Entry(
            self.root,
            textvariable=self.save_plots_entry_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.save_plots_entry.grid(row=5, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.save_plots_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.save_plots_entry_var),
            state=tk.DISABLED,
        )
        self.save_plots_button.grid(row=5, column=2, padx=pad_x, pady=pad_y)
        self.save_plots_var.trace_add(
            "write", lambda *args: self.update_save_plot_widgets()
        )

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)

        tk.Checkbutton(
            self.root,
            text="Save Results To",
            variable=self.save_results_var,
            command=self.update_save_results_widgets,
        ).grid(row=7, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.save_results_entry = tk.Entry(
            self.root,
            textvariable=self.save_results_entry_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.save_results_entry.grid(
            row=7, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.save_results_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.save_results_entry_var),
            state=tk.DISABLED,
        )
        self.save_results_button.grid(row=7, column=2, padx=pad_x, pady=pad_y)
        self.save_results_var.trace_add(
            "write", lambda *args: self.update_save_results_widgets()
        )

        tk.Button(self.root, text="Run Merge Fits", command=self.run_merge_fits).grid(
            row=8, column=0, columnspan=3, pady=10, padx=pad_x
        )
        self.lift_and_focus()

    def update_save_plot_widgets(self):
        state = tk.NORMAL if self.save_plots_var.get() else tk.DISABLED
        self.save_plots_entry.config(state=state)
        self.save_plots_button.config(state=state)

    def update_save_results_widgets(self):
        state = tk.NORMAL if self.save_results_var.get() else tk.DISABLED
        self.save_results_entry.config(state=state)
        self.save_results_button.config(state=state)

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
            self.show_message("Merging fits completed!", is_error=False)
        except Exception as e:
            self.show_message(f"Error: {str(e)}", is_error=True)


if __name__ == "__main__":
    root = tk.Tk()
    GDAMergeFitsApp(root)
    root.mainloop()
