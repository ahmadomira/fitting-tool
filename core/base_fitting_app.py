import tkinter as tk
from tkinter import filedialog
import os
   
class ExpandedFittingApp:
    def __init__(self, root, title):
        self.root = root
        self.root.title(title)
        self.info_label = None
        
        # Variables
        self.file_path_var = tk.StringVar()
        self.use_results_file_var = tk.BooleanVar()
        self.results_file_path_var = tk.StringVar()
        self.fit_trials_var = tk.IntVar()
        self.rmse_threshold_var = tk.DoubleVar()
        self.r2_threshold_var = tk.DoubleVar()
        self.save_plots_var = tk.BooleanVar()
        self.results_dir_var = tk.StringVar()
        self.display_plots_var = tk.BooleanVar()
        self.save_results_var = tk.BooleanVar()
        self.results_save_dir_var = tk.StringVar()
        self.Kd_var = tk.DoubleVar()
        self.h0_var = tk.DoubleVar()
        self.g0_var = tk.DoubleVar()
        self.d0_var = tk.DoubleVar()

        # Padding
        self.pad_x = 10
        self.pad_y = 5

        # Set default values
        self.set_default_values()

        # Widgets
        self.create_widgets()

    def set_default_values(self):
        self.file_path_var.set("/path/to/default/file.txt")
        self.results_dir_var.set("/path/to/default/dir")
        self.results_save_dir_var.set("/path/to/default/dir")
        self.fit_trials_var.set(10)
        self.rmse_threshold_var.set(2)
        self.r2_threshold_var.set(0.9)
        self.display_plots_var.set(True)

    def create_widgets(self):
        tk.Label(self.root, text="Input File Path:").grid(row=0, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.file_path_entry = tk.Entry(self.root, textvariable=self.file_path_var, width=40, justify='left')
        self.file_path_entry.grid(row=0, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)
        tk.Button(self.root, text="Browse", command=self.browse_input_file).grid(row=0, column=2, padx=self.pad_x, pady=self.pad_y)

        tk.Checkbutton(self.root, text="Read Boundaries from File: ", variable=self.use_results_file_var, command=self.update_use_results_widgets).grid(row=1, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.results_file_path_entry = tk.Entry(self.root, textvariable=self.results_file_path_var, width=40, justify='left', state=tk.DISABLED)
        self.results_file_path_entry.grid(row=1, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)
        self.results_file_button = tk.Button(self.root, text="Browse", command=lambda: self.browse_file(self.results_file_path_entry), state=tk.DISABLED)
        self.results_file_button.grid(row=1, column=2, padx=self.pad_x, pady=self.pad_y)
        self.use_results_file_var.trace_add('write', lambda *args: self.update_use_results_widgets())

        tk.Label(self.root, text=r"Kₐ (M⁻¹):").grid(row=2, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.Kd_entry = tk.Entry(self.root, textvariable=self.Kd_var, justify='left')
        self.Kd_entry.grid(row=2, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)

        tk.Label(self.root, text="H₀ (M):").grid(row=3, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.h0_entry = tk.Entry(self.root, textvariable=self.h0_var, justify='left')
        self.h0_entry.grid(row=3, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)

        tk.Label(self.root, text="G₀ (M):").grid(row=4, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.g0_entry = tk.Entry(self.root, textvariable=self.g0_var, justify='left')
        self.g0_entry.grid(row=4, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)

        tk.Label(self.root, text="D₀ (M):").grid(row=5, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.d0_entry = tk.Entry(self.root, textvariable=self.d0_var, justify='left')
        self.d0_entry.grid(row=5, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)

        tk.Label(self.root, text="Number of Fit Trials:").grid(row=6, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.fit_trials_entry = tk.Entry(self.root, textvariable=self.fit_trials_var, justify='left')
        self.fit_trials_entry.grid(row=6, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)

        tk.Label(self.root, text="RMSE Threshold Factor:").grid(row=7, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.rmse_threshold_entry = tk.Entry(self.root, textvariable=self.rmse_threshold_var, justify='left')
        self.rmse_threshold_entry.grid(row=7, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)

        tk.Label(self.root, text="R² Threshold:").grid(row=8, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.r2_threshold_entry = tk.Entry(self.root, textvariable=self.r2_threshold_var, justify='left')
        self.r2_threshold_entry.grid(row=8, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)

        tk.Checkbutton(self.root, text="Save Plots To", variable=self.save_plots_var, command=self.update_save_plot_widgets).grid(row=9, column=0, columnspan=1, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.results_dir_entry = tk.Entry(self.root, textvariable=self.results_dir_var, width=40, state=tk.DISABLED, justify='left')
        self.results_dir_entry.grid(row=9, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)
        self.results_dir_button = tk.Button(self.root, text="Browse", command=lambda: self.browse_directory(self.results_dir_entry), state=tk.DISABLED)
        self.results_dir_button.grid(row=9, column=2, padx=self.pad_x, pady=self.pad_y)

        self.save_plots_var.trace_add('write', lambda *args: self.update_save_plot_widgets())

        tk.Checkbutton(self.root, text="Save Results To", variable=self.save_results_var, command=self.update_save_results_widgets).grid(row=10, column=0, columnspan=1, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        self.results_save_dir_entry = tk.Entry(self.root, textvariable=self.results_save_dir_var, width=40, state=tk.DISABLED, justify='left')
        self.results_save_dir_entry.grid(row=10, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)
        self.results_save_dir_button = tk.Button(self.root, text="Browse", command=lambda: self.browse_directory(self.results_save_dir_entry), state=tk.DISABLED)
        self.results_save_dir_button.grid(row=10, column=2, padx=self.pad_x, pady=self.pad_y)

        self.save_results_var.trace_add('write', lambda *args: self.update_save_results_widgets())

        tk.Checkbutton(self.root, text="Display Plots", variable=self.display_plots_var).grid(row=11, column=0, columnspan=3, sticky=tk.W, padx=self.pad_x, pady=self.pad_y)
        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(row=12, column=0, columnspan=3, pady=10, padx=self.pad_x)

        # Bring the window to the front
        self.root.lift()
        self.root.focus_force()
        
    def browse_input_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_var.set(file_path)
            root_dir = os.path.dirname(file_path)
            self.results_dir_var.set(root_dir)
            self.results_save_dir_var.set(root_dir)

    def browse_file(self, entry):
        initial_dir = os.path.dirname(self.file_path_var.get()) if self.file_path_var.get() else os.getcwd()
        file_path = filedialog.askopenfilename(initialdir=initial_dir)
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def browse_directory(self, entry):
        initial_dir = os.path.dirname(self.file_path_var.get()) if self.file_path_var.get() else os.getcwd()
        directory_path = filedialog.askdirectory(initialdir=initial_dir)
        if directory_path:
            entry.delete(0, tk.END)
            entry.insert(0, directory_path)

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

    def show_message(self, message, is_error=False):
        if self.info_label:
            self.info_label.destroy()
        fg_color = 'red' if is_error else 'green'
        self.info_label = tk.Label(self.root, text=message, fg=fg_color)
        self.info_label.grid(row=13, column=0, columnspan=3, pady=10)

    def run_fitting(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
if __name__ == "__main__":
    root = tk.Tk()
    app = ExpandedFittingApp(root, "Expanded Fitting App")
    root.mainloop()
