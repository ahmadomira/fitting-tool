import tkinter as tk

from core.fitting.dye_alone import DyeAloneFittingAlgorithm
from gui.base_gui import BaseAppGUI


class DyeAloneFittingApp(BaseAppGUI):
    def __init__(self, root):
        super().__init__(root, title="Dye Alone Fitting")
        self.file_path_var = self.add_string_var("file_path", "")
        self.save_path_var = self.add_string_var("save_path", "")
        self.save_plots_var = self.add_bool_var("save_plots", False)
        self.display_plots_var = self.add_bool_var("display_plots", True)
        self.plots_dir_var = self.add_string_var("plots_dir", "")

        pad_x = self.pad_x
        pad_y = self.pad_y

        tk.Label(self.root, text="Input File:").grid(
            row=0, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.file_path_entry = tk.Entry(
            self.root, textvariable=self.file_path_var, width=50
        )
        self.file_path_entry.grid(row=0, column=1, padx=pad_x, pady=pad_y)
        tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_file(self.file_path_var),
        ).grid(row=0, column=2, padx=pad_x, pady=pad_y)

        tk.Label(self.root, text="Save Result To:").grid(
            row=1, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.save_path_entry = tk.Entry(
            self.root, textvariable=self.save_path_var, width=50
        )
        self.save_path_entry.grid(row=1, column=1, padx=pad_x, pady=pad_y)
        tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_file(self.save_path_var),
        ).grid(row=1, column=2, padx=pad_x, pady=pad_y)

        tk.Checkbutton(
            self.root,
            text="Save Plots",
            variable=self.save_plots_var,
            command=self.update_save_plot_widgets,
        ).grid(row=2, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.plots_dir_entry = tk.Entry(
            self.root, textvariable=self.plots_dir_var, width=50, state=tk.DISABLED
        )
        self.plots_dir_entry.grid(row=2, column=1, padx=pad_x, pady=pad_y)
        self.plots_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.plots_dir_var),
            state=tk.DISABLED,
        )
        self.plots_dir_button.grid(row=2, column=2, padx=pad_x, pady=pad_y)
        self.save_plots_var.trace_add(
            "write", lambda *args: self.update_save_plot_widgets()
        )

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)

        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(
            row=4, column=1, pady=10
        )
        self.lift_and_focus()

    def update_save_plot_widgets(self):
        state = tk.NORMAL if self.save_plots_var.get() else tk.DISABLED
        self.plots_dir_entry.config(state=state)
        self.plots_dir_button.config(state=state)

    def run_fitting(self):
        input_path = self.file_path_var.get()
        output_path = self.save_path_var.get()
        save_plots = self.save_plots_var.get()
        display_plots = self.display_plots_var.get()
        plots_dir = self.plots_dir_var.get()
        if not input_path or not output_path:
            self.show_message("Error: Please set all parameters.", is_error=True)
            return
        try:
            algorithm = DyeAloneFittingAlgorithm()
            algorithm.perform_fitting(
                input_path, output_path, save_plots, display_plots, plots_dir
            )
            self.show_message(f"Results saved to: {output_path}")
        except Exception as e:
            self.show_message(f"Error: {str(e)}", is_error=True)


if __name__ == "__main__":
    root = tk.Tk()
    DyeAloneFittingApp(root)
    root.mainloop()
