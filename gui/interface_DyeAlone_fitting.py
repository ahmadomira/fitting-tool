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

        self.file_path_entry, self.file_path_browse = self.add_file_selector(
            row=0, label_text="Input File Path:", var=self.file_path_var
        )

        self.save_path_entry = self.add_labeled_entry(
            row=1, label_text="Save Result To:", var=self.save_path_var, width=50
        )
        self.save_path_browse = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_save_file(
                self.save_path_var, input_var=self.file_path_var
            ),
        )
        self.save_path_browse.grid(row=1, column=2, padx=pad_x, pady=pad_y)

        self.plots_dir_entry, self.plots_dir_button = self.add_toggleable_dir_selector(
            row=2,
            label_text="Save Plots",
            bool_var=self.save_plots_var,
            dir_var=self.plots_dir_var,
            input_file_var=self.file_path_var,
            width=50,
        )

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)

        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(
            row=4, column=1, pady=10
        )
        self.lift_and_focus()

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
