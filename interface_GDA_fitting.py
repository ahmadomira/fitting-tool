import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plot_utils import plot_fitting_results, save_plot
from fitting_utils import load_data, compute_signal_gda, calculate_fit_metrics, residuals, save_replica_file, split_replicas, load_bounds_from_results_file

import traceback
    
def run_gda_fitting(file_path, results_file_path, Kd_in_M, h0_in_M, g0_in_M, number_of_fit_trials, rmse_threshold_factor, r2_threshold, save_plots, display_plots, plots_dir, save_results_bool, results_save_dir, assay='gda'):


    # try loading bounds from results file if available
    Id_lower, Id_upper, I0_lower, I0_upper, Ihd_lower, Ihd_upper = load_bounds_from_results_file(results_file_path)

    # Print boundary values for verification
    print(f"Loaded boundaries:\nId: [{Id_lower * 1e6:.3e}, {Id_upper * 1e6:.3e}] M⁻¹\nI0: [{I0_lower:.3e}, {I0_upper:.3e}]")
    
    # Convert constants to µM
    Kd = Kd_in_M / 1e6
    h0 = h0_in_M * 1e6
    g0 = g0_in_M * 1e6

    # Main fitting process
    data_lines = load_data(file_path)
    replicas = split_replicas(data_lines)

    print(f"Number of replicas detected: {len(replicas)}")
    
    figures = []  # List to store figures

    # Process each replica for fitting
    for replica_index, replica_data in enumerate(replicas, start=1):
        d0_values = replica_data[:, 0] * 1e6
        Signal_observed = replica_data[:, 1]

        if len(d0_values) < 2:
            print(f"Replica {replica_index} has insufficient data. Skipping.")
            continue

        # Update I0_upper if needed
        I0_upper = np.min(Signal_observed) if I0_upper is None or np.isinf(I0_upper) else I0_upper

        # Generate initial guesses for parameters
        Ihd_guess_smaller = Signal_observed[0] < Signal_observed[-1]
        initial_params_list = []
        for _ in range(number_of_fit_trials):
            I0_guess = np.random.uniform(I0_lower, I0_upper)
            Kg_guess = 10 ** np.random.uniform(np.log10(Kd) - 5, np.log10(Kd) + 5)
            if Ihd_guess_smaller:
                Id_guess = 10 ** np.random.uniform(np.log10(Id_lower), np.log10(Id_upper))
                Ihd_guess = Id_guess * np.random.uniform(0.1, 0.5)
            else:
                Ihd_guess = 10 ** np.random.uniform(np.log10(Ihd_lower), np.log10(Ihd_upper))
                Id_guess = Ihd_guess * np.random.uniform(0.1, 0.5)
            initial_params_list.append([I0_guess, Kg_guess, Id_guess, Ihd_guess])

        # Fit parameters for replica using least-squares minimization
        best_result, best_cost = None, np.inf
        fit_results = []
        for initial_params in initial_params_list:
            result = minimize(lambda params: np.sum(residuals(Signal_observed, compute_signal_gda, params, d0_values, Kd, h0, g0) ** 2),
                                initial_params, method='L-BFGS-B',
                                bounds=[(I0_lower, I0_upper), (1e-8, 1e8), (Id_lower, Id_upper), (Ihd_lower, Ihd_upper)])
            Signal_computed = compute_signal_gda(result.x, d0_values, Kd, h0, g0)
            rmse, r_squared = calculate_fit_metrics(Signal_observed, Signal_computed)
            fit_results.append((result.x, result.fun, rmse, r_squared))

            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result

        # Filter fits by RMSE and R² thresholds
        best_rmse = min(fit_rmse for _, _, fit_rmse, _ in fit_results)
        rmse_threshold = best_rmse * rmse_threshold_factor

        filtered_results = [
            (params, fit_rmse, fit_r2) for params, _, fit_rmse, fit_r2 in fit_results
            if fit_rmse <= rmse_threshold and fit_r2 >= r2_threshold
        ]

        # Calculate median parameters if valid results are found
        if filtered_results:
            median_params = np.median(np.array([result[0] for result in filtered_results]), axis=0)
        else:
            print("Warning: No fits meet the filtering criteria.")
            continue

        # Compute metrics for median fit
        Signal_computed = compute_signal_gda(median_params, d0_values, Kd, h0, g0)
        rmse, r_squared = calculate_fit_metrics(Signal_observed, Signal_computed)

        # Plot observed vs. simulated fitting curve
        fitting_curve_x, fitting_curve_y = [], []
        for i in range(len(d0_values) - 1):
            extra_points = np.linspace(d0_values[i], d0_values[i + 1], 21)
            fitting_curve_x.extend(extra_points)
            fitting_curve_y.extend(compute_signal_gda(median_params, extra_points, Kd, h0, g0))
        
        plot_title = f'Replica {replica_index}'
        fig = plot_fitting_results(d0_values, Signal_observed, fitting_curve_x, fitting_curve_y, median_params, rmse, r_squared, assay, plot_title)

        # the label is used in save_plot as the filename for saving the plot
        fig.set_label(f"fit_plot_replica_{replica_index}")
        figures.append(fig)

        # Save results to a file if save_results is True
        replica_filename = f"fit_results_replica_{replica_index}.txt"
        if save_results_bool:
            input_params = (g0_in_M, h0_in_M, Kd_in_M, Id_lower, Id_upper, I0_lower, I0_upper) 
            median_params = (*median_params, rmse, r_squared)
            fitting_params = (d0_values, Signal_observed, fitting_curve_x, fitting_curve_y, replica_index)
            save_replica_file(results_save_dir, replica_filename, filtered_results, input_params, median_params, fitting_params, assay)

    if save_plots:
        for fig in figures:
            save_plot(fig, plots_dir)
    
    if display_plots:
        plt.show()

# Function to browse file
class GDAFittingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GDA Fitting Interface")
        self.info_label = None
        
        # Variables
        self.file_path_var = tk.StringVar()
        self.use_results_file_var = tk.BooleanVar()
        self.results_file_path_var = tk.StringVar()
        self.Kd_var = tk.DoubleVar()
        self.h0_var = tk.DoubleVar()
        self.g0_var = tk.DoubleVar()
        self.fit_trials_var = tk.IntVar()
        self.rmse_threshold_var = tk.DoubleVar()
        self.r2_threshold_var = tk.DoubleVar()
        self.save_plots_var = tk.BooleanVar()
        self.plots_dir_var = tk.StringVar()
        self.display_plots_var = tk.BooleanVar()
        self.save_results_var = tk.BooleanVar()  # New variable for saving results
        self.results_dir_var = tk.StringVar()  # New variable for results save directory
        
        # Set default values
        self.file_path_var.set("/Users/ahmadomira/Downloads/interface_test/GDA_system.txt")
        self.plots_dir_var.set("/Users/ahmadomira/Downloads/interface_test/untitled folder")
        self.results_dir_var.set("/Users/ahmadomira/Downloads/interface_test/untitled folder")
        self.Kd_var.set(1.68e7)  # Binding constant for h_d binding in M^-1
        self.h0_var.set(4.3e-6)  # Initial host concentration (M)
        self.g0_var.set(6e-6)    # Initial guest concentration (M)
        self.fit_trials_var.set(10)  # Number of fit trials
        self.rmse_threshold_var.set(2)  # RMSE threshold factor
        self.r2_threshold_var.set(0.9)  # R² threshold
        self.display_plots_var.set(True)
        
        # Padding
        pad_x = 10
        pad_y = 5

        # Widgets
        tk.Label(self.root, text="Input File Path:").grid(row=0, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.file_path_entry = tk.Entry(self.root, textvariable=self.file_path_var, width=40, justify='left')
        self.file_path_entry.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        tk.Button(self.root, text="Browse", command=self.browse_input_file).grid(row=0, column=2, padx=pad_x, pady=pad_y)

        tk.Checkbutton(self.root, text="Read Boundaries from File: ", variable=self.use_results_file_var, command=self.update_use_results_widgets).grid(row=1, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.results_file_path_entry = tk.Entry(self.root, textvariable=self.results_file_path_var, width=40, justify='left', state=tk.DISABLED)
        self.results_file_path_entry.grid(row=1, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.results_file_button = tk.Button(self.root, text="Browse", command=lambda: self.browse_file(self.results_file_path_entry), state=tk.DISABLED)
        self.results_file_button.grid(row=1, column=2, padx=pad_x, pady=pad_y)
        self.use_results_file_var.trace_add('write', lambda *args: self.update_use_results_widgets())

        tk.Label(self.root, text=r"Kₐ (M⁻¹):").grid(row=3, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.Kd_entry = tk.Entry(self.root, textvariable=self.Kd_var, justify='left')
        self.Kd_entry.grid(row=3, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="H₀ (M):").grid(row=4, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.h0_entry = tk.Entry(self.root, textvariable=self.h0_var, justify='left')
        self.h0_entry.grid(row=4, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="G₀ (M):").grid(row=5, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.g0_entry = tk.Entry(self.root, textvariable=self.g0_var, justify='left')
        self.g0_entry.grid(row=5, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="Number of Fit Trials:").grid(row=6, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.fit_trials_entry = tk.Entry(self.root, textvariable=self.fit_trials_var, justify='left')
        self.fit_trials_entry.grid(row=6, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="RMSE Threshold Factor:").grid(row=7, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.rmse_threshold_entry = tk.Entry(self.root, textvariable=self.rmse_threshold_var, justify='left')
        self.rmse_threshold_entry.grid(row=7, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="R² Threshold:").grid(row=8, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.r2_threshold_entry = tk.Entry(self.root, textvariable=self.r2_threshold_var, justify='left')
        self.r2_threshold_entry.grid(row=8, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Checkbutton(self.root, text="Save Plots To", variable=self.save_plots_var, command=self.update_save_plot_widgets).grid(row=9, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.plots_dir_entry = tk.Entry(self.root, textvariable=self.plots_dir_var, width=40, state=tk.DISABLED, justify='left')
        self.plots_dir_entry.grid(row=9, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.plots_dir_button = tk.Button(self.root, text="Browse", command=lambda: self.browse_directory(self.plots_dir_entry), state=tk.DISABLED)
        self.plots_dir_button.grid(row=9, column=2, padx=pad_x, pady=pad_y)

        self.save_plots_var.trace_add('write', lambda *args: self.update_save_plot_widgets())

        tk.Checkbutton(self.root, text="Save Results To", variable=self.save_results_var, command=self.update_save_results_widgets).grid(row=10, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.results_save_dir_entry = tk.Entry(self.root, textvariable=self.results_dir_var, width=40, state=tk.DISABLED, justify='left')
        self.results_save_dir_entry.grid(row=10, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.results_save_dir_button = tk.Button(self.root, text="Browse", command=lambda: self.browse_directory(self.results_save_dir_entry), state=tk.DISABLED)
        self.results_save_dir_button.grid(row=10, column=2, padx=pad_x, pady=pad_y)

        self.save_results_var.trace_add('write', lambda *args: self.update_save_results_widgets())

        tk.Checkbutton(self.root, text="Display Plots", variable=self.display_plots_var).grid(row=11, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(row=12, column=0, columnspan=3, pady=10, padx=pad_x)

        # Bring the window to the front
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
        self.plots_dir_entry.config(state=state)
        self.plots_dir_button.config(state=state)

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
        # Adjust the run_fitting function to access self variables and implement the fitting logic
        try:
            # Get user inputs
            file_path = self.file_path_entry.get()
            results_file_path = self.results_file_path_entry.get() if self.use_results_file_var.get() else None
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
            
            # Run the fitting process
            run_gda_fitting(file_path, results_file_path, Kd_in_M, h0_in_M, g0_in_M, number_of_fit_trials, rmse_threshold_factor, r2_threshold, save_plots, display_plots, plots_dir, save_results, results_save_dir)
            
            self.show_message(f"Fitting completed!", is_error=False)
            
        except Exception as e:
            self.show_message(f"Error: {str(e)}", is_error=True)

# Main function to run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Automation Project")

    GDAFittingApp(root)

    root.mainloop()
