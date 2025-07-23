import os
import tkinter as tk
from tkinter import filedialog

from core.fitting.dye_alone import DyeAloneFittingAlgorithm
from core.progress_window import ProgressWindow


class DyeAloneFittingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dye Alone Fitting")

        # Variables
        self.file_path_var = tk.StringVar()

        self.save_path_var = tk.StringVar()
        self.save_plots_var = tk.BooleanVar()
        self.display_plots_var = tk.BooleanVar()
        self.plots_dir_var = tk.StringVar()
        self.custom_x_label_var = tk.BooleanVar()
        self.custom_x_label_text_var = tk.StringVar()
        self.custom_plot_title_var = tk.BooleanVar()
        self.custom_plot_title_text_var = tk.StringVar()

        # Set default values
        # # For testing
        # self.file_path_var.set(
        #     "/Users/ahmadomira/git/App Test/dye-alone-test/Dye_alone.txt"
        # )
        # self.save_path_var.set(
        #     "/Users/ahmadomira/git/App Test/dye-alone-test/dye_alone_results.txt"
        # )
        # self.plots_dir_var.set("/Users/ahmadomira/git/App Test/dye-alone-test/")
        # self.save_plots_var.set(True)
        self.display_plots_var.set(True)

        tk.Label(root, text="Input File:").grid(row=0, column=0, sticky=tk.W)
        self.file_path_entry = tk.Entry(root, textvariable=self.file_path_var, width=50)
        self.file_path_entry.grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_input_file).grid(
            row=0, column=2
        )

        tk.Label(root, text="Save Result To:").grid(row=1, column=0, sticky=tk.W)
        self.save_path_entry = tk.Entry(root, textvariable=self.save_path_var, width=50)
        self.save_path_entry.grid(row=1, column=1)
        tk.Button(root, text="Browse", command=self.browse_save_path).grid(
            row=1, column=2
        )

        tk.Checkbutton(
            root,
            text="Save Plots",
            variable=self.save_plots_var,
            command=self.update_save_plot_widgets,
        ).grid(row=2, column=0, sticky=tk.W)
        self.plots_dir_entry = tk.Entry(
            root, textvariable=self.plots_dir_var, width=50, state=tk.DISABLED
        )
        self.plots_dir_entry.grid(row=2, column=1)
        self.plots_dir_button = tk.Button(
            root, text="Browse", command=self.browse_plots_dir, state=tk.DISABLED
        )
        self.plots_dir_button.grid(row=2, column=2)

        tk.Checkbutton(
            root,
            text="Custom Plot Title:",
            variable=self.custom_plot_title_var,
            command=self.update_custom_plot_title_widgets,
        ).grid(row=3, column=0, sticky=tk.W)
        self.custom_plot_title_entry = tk.Entry(
            root,
            textvariable=self.custom_plot_title_text_var,
            width=50,
            state=tk.DISABLED,
        )
        self.custom_plot_title_entry.grid(row=3, column=1, columnspan=2, sticky=tk.W)

        tk.Checkbutton(
            root,
            text="Custom X-axis Label:",
            variable=self.custom_x_label_var,
            command=self.update_custom_label_widgets,
        ).grid(row=4, column=0, sticky=tk.W)
        self.custom_x_label_entry = tk.Entry(
            root, textvariable=self.custom_x_label_text_var, width=50, state=tk.DISABLED
        )
        self.custom_x_label_entry.grid(row=4, column=1, columnspan=2, sticky=tk.W)

        tk.Checkbutton(
            root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=5, column=0, columnspan=3, sticky=tk.W)

        tk.Button(root, text="Run Fitting", command=self.run_fitting).grid(
            row=6, column=1, pady=10
        )
        self.info_label = None

        self.save_plots_var.trace_add(
            "write", lambda *args: self.update_save_plot_widgets()
        )
        self.custom_x_label_var.trace_add(
            "write", lambda *args: self.update_custom_label_widgets()
        )
        self.custom_plot_title_var.trace_add(
            "write", lambda *args: self.update_custom_plot_title_widgets()
        )

    def show_message(self, message, is_error=False):
        if self.info_label:
            self.info_label.destroy()
        fg_color = "red" if is_error else "green"
        self.info_label = tk.Label(self.root, text=message, fg=fg_color)
        self.info_label.grid(row=7, column=0, columnspan=3, pady=10)

    def browse_input_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_var.set(file_path)
            root_dir = os.path.dirname(file_path)
            self.save_path_var.set(os.path.join(root_dir, f"dye_alone_results.txt"))
            self.plots_dir_var.set(os.path.join(root_dir))

    def browse_save_path(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if file_path:
            self.save_path_var.set(file_path)

    def browse_plots_dir(self):
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.plots_dir_var.set(directory_path)

    def update_save_plot_widgets(self):
        state = tk.NORMAL if self.save_plots_var.get() else tk.DISABLED
        self.plots_dir_entry.config(state=state)
        self.plots_dir_button.config(state=state)

    def update_custom_label_widgets(self):
        state = tk.NORMAL if self.custom_x_label_var.get() else tk.DISABLED
        self.custom_x_label_entry.config(state=state)

    def update_custom_plot_title_widgets(self):
        state = tk.NORMAL if self.custom_plot_title_var.get() else tk.DISABLED
        self.custom_plot_title_entry.config(state=state)

    def run_fitting(self):
        input_path = self.file_path_var.get()
        output_path = self.save_path_var.get()
        save_plots = self.save_plots_var.get()
        display_plots = self.display_plots_var.get()
        plots_dir = self.plots_dir_var.get()

        # Get custom x-axis label
        custom_x_label = None
        if self.custom_x_label_var.get() and self.custom_x_label_text_var.get().strip():
            custom_x_label = self.custom_x_label_text_var.get().strip()

        # Get custom plot title
        custom_plot_title = None
        if (
            self.custom_plot_title_var.get()
            and self.custom_plot_title_text_var.get().strip()
        ):
            custom_plot_title = self.custom_plot_title_text_var.get().strip()

        if not input_path or not output_path:
            self.show_message("Error: Please set all parameters.", is_error=True)
            return

        try:
            with ProgressWindow(
                self.root,
                "Fitting in Progress",
                "Dye-alone fitting in progress, please wait...",
            ) as progress_window:
                algorithm = DyeAloneFittingAlgorithm()
                algorithm.perform_fitting(
                    input_path,
                    output_path,
                    save_plots,
                    display_plots,
                    plots_dir,
                    custom_x_label,
                    custom_plot_title,
                )
            self.show_message(f"Results saved to: {output_path}")
        except Exception as e:
            self.show_message(f"Error: {str(e)}", is_error=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = DyeAloneFittingApp(root)
    root.mainloop()
