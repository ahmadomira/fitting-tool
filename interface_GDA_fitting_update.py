import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from scipy.optimize import brentq, minimize
import pandas as pd
from datetime import datetime
import os

import matplotlib.pyplot as plt

# Function to run the fitting process
def run_fitting():
    try:
        # Get user inputs
        file_path = file_path_entry.get()
        results_file_path = results_file_path_entry.get() if use_results_file_var.get() else None
        Kd_in_M = float(Kd_entry.get())
        h0_in_M = float(h0_entry.get())
        g0_in_M = float(g0_entry.get())
        number_of_fit_trials = int(fit_trials_entry.get())
        rmse_threshold_factor = float(rmse_threshold_entry.get())
        r2_threshold = float(r2_threshold_entry.get())
        save_plots = save_plots_var.get()
        display_plots = display_plots_var.get()

        # Load data from input file
        data_lines = load_data(file_path)
        if data_lines is None:
            raise ValueError("Data loading failed.")
        replicas = split_replicas(data_lines)
        if replicas is None:
            raise ValueError("Replica splitting failed.")
        print(f"Number of replicas detected: {len(replicas)}")

        # Initialize parameter ranges for fitting
        I0_range = (0, None)
        Id_range = (None, None)
        Ihd_range = (None, None)

        # Check and load boundaries from results file, if available
        if results_file_path and os.path.exists(results_file_path):
            try:
                with open(results_file_path, 'r') as f:
                    lines = f.readlines()

                # Extract Id prediction interval
                id_prediction_line = next((line for line in lines if 'Id prediction interval' in line), None)
                if id_prediction_line and 'not applicable' not in id_prediction_line:
                    Id_lower = float(id_prediction_line.split('[')[-1].split(',')[0].strip())
                    Id_upper = float(id_prediction_line.split(',')[-1].split(']')[0].strip())
                else:
                    average_Id = float(next(line for line in lines if 'Average Id' in line).split('\t')[-1].strip())
                    Id_lower = 0.5 * average_Id
                    Id_upper = 2.0 * average_Id

                # Extract I0 prediction interval
                i0_prediction_line = next((line for line in lines if 'I0 prediction interval' in line), None)
                if i0_prediction_line and 'not applicable' not in i0_prediction_line:
                    I0_lower = float(i0_prediction_line.split('[')[-1].split(',')[0].strip())
                    I0_upper = float(i0_prediction_line.split(',')[-1].split(']')[0].strip())
                else:
                    average_I0 = float(next(line for line in lines if 'Average I0' in line).split('\t')[-1].strip())
                    I0_lower = 0.5 * average_I0
                    I0_upper = 2.0 * average_I0

            except Exception as e:
                print(f"Error parsing boundaries from the results file: {e}")
                Id_lower, Id_upper = 1e3, 1e18
                I0_lower, I0_upper = 0, None
        else:
            Id_lower, Id_upper = 1e3, 1e18
            I0_lower, I0_upper = 0, None

        # Convert bounds to µM⁻¹ for fitting
        Id_lower /= 1e6
        Id_upper /= 1e6
        Ihd_lower = Ihd_range[0] / 1e6 if Ihd_range[0] is not None else 0.001
        Ihd_upper = Ihd_range[1] / 1e6 if Ihd_range[1] is not None else 1e12

        # Convert constants to µM
        Kd = Kd_in_M / 1e6
        h0 = h0_in_M * 1e6
        g0 = g0_in_M * 1e6

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
                result = minimize(lambda params: np.sum(residuals(params, d0_values, Signal_observed, Kd, h0, g0) ** 2),
                                  initial_params, method='L-BFGS-B',
                                  bounds=[(I0_lower, I0_upper), (1e-8, 1e8), (Id_lower, Id_upper), (Ihd_lower, Ihd_upper)])
                Signal_computed = compute_signal(result.x, d0_values, Kd, h0, g0)
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
            Signal_computed = compute_signal(median_params, d0_values, Kd, h0, g0)
            rmse, r_squared = calculate_fit_metrics(Signal_observed, Signal_computed)

            # Plot observed vs. simulated fitting curve
            fitting_curve_x, fitting_curve_y = [], []
            for i in range(len(d0_values) - 1):
                extra_points = np.linspace(d0_values[i], d0_values[i + 1], 21)
                fitting_curve_x.extend(extra_points)
                fitting_curve_y.extend(compute_signal(median_params, extra_points, Kd, h0, g0))

            plt.figure(figsize=(8, 6))
            plt.plot(d0_values, Signal_observed, 'o', label='Observed Signal')
            plt.plot(fitting_curve_x, fitting_curve_y, '--', color='blue', alpha=0.6, label='Simulated Fitting Curve')
            plt.xlabel('d0 (µM)')
            plt.ylabel('Signal')
            plt.title(f'Observed vs. Simulated Fitting Curve for Replica {replica_index}')
            plt.legend()
            plt.grid(True)

            # Annotate plot with median parameter values and fit metrics
            param_text = (f"Kg: {median_params[1] * 1e6:.2e} M^-1\n"
                          f"I0: {median_params[0]:.2e}\n"
                          f"Id: {median_params[2] * 1e6:.2e} signal/M\n"
                          f"Ihd: {median_params[3] * 1e6:.2e} signal/M\n"
                          f"RMSE: {rmse:.3f}\n"
                          f"R²: {r_squared:.3f}")

            plt.gca().annotate(param_text, xy=(1.05, 0.5), xycoords='axes fraction', fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))

            if save_plots:
                results_dir = results_dir_entry.get()
                plot_file = os.path.join(results_dir, f"fit_plot_replica_{replica_index}.png")
                plt.savefig(plot_file, bbox_inches='tight')
                print(f"Plot saved to {plot_file}")
            if display_plots:
                plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to load data from file
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.readlines()
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Function to split data into replicas
def split_replicas(data):
    if data is None:
        return None
    replicas, current_replica = [], []
    use_var_signal_split = any("var signal" in line.lower() for line in data)

    for line in data:
        if "var" in line.lower():
            if current_replica:
                replicas.append(np.array(current_replica))
                current_replica = []
        else:
            try:
                x, y = map(float, line.split())
                if use_var_signal_split:
                    current_replica.append((x, y))
                else:
                    if x == 0.0 and current_replica:
                        replicas.append(np.array(current_replica))
                        current_replica = []
                    current_replica.append((x, y))
            except ValueError:
                continue
    if current_replica:
        replicas.append(np.array(current_replica))
    return replicas if replicas else None

# Function to compute signal
def compute_signal(params, d0_values, Kd, h0, g0):
    I0, Kg, Id, Ihd = params
    Signal_values = []
    for d0 in d0_values:
        try:
            def equation_h(h):
                denom_Kd = 1 + Kd * h
                denom_Kg = 1 + Kg * h
                h_d = (Kd * h * d0) / denom_Kd
                h_g = (Kg * h * g0) / denom_Kg
                return h + h_d + h_g - h0

            h_sol = brentq(equation_h, 1e-20, h0, xtol=1e-14, maxiter=1000)
            denom_Kd = 1 + Kd * h_sol
            d_free = d0 / denom_Kd
            h_d = Kd * h_sol * d_free
            Signal = I0 + Id * d_free + Ihd * h_d
            Signal_values.append(Signal)
        except Exception:
            Signal_values.append(np.nan)
    return np.array(Signal_values)

# Function to compute residuals
def residuals(params, d0_values, Signal_observed, Kd, h0, g0):
    Signal_computed = compute_signal(params, d0_values, Kd, h0, g0)
    return np.nan_to_num(Signal_observed - Signal_computed, nan=1e6)

# Function to calculate fit metrics
def calculate_fit_metrics(Signal_observed, Signal_computed):
    rmse = np.sqrt(np.nanmean((Signal_observed - Signal_computed) ** 2))
    ss_res = np.nansum((Signal_observed - Signal_computed) ** 2)
    ss_tot = np.nansum((Signal_observed - np.nanmean(Signal_observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return rmse, r_squared

# Function to browse file
def browse_file(entry):
    file_path = filedialog.askopenfilename()
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

# Create the main window
root = tk.Tk()
root.title("GDA Fitting Update Interface")

# Create and place widgets
pad_x = 10  # Increase padding
pad_y = 5

tk.Label(root, text="Input File Path:").grid(row=0, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
file_path_entry = tk.Entry(root, width=40, justify='left')
file_path_entry.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
tk.Button(root, text="Browse", command=lambda: browse_file(file_path_entry)).grid(row=0, column=2, padx=pad_x, pady=pad_y)

use_results_file_var = tk.BooleanVar()
tk.Checkbutton(root, text="Read Boundaries from File: ", variable=use_results_file_var, command=lambda : update_use_results_widgets()).grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
results_file_path_entry = tk.Entry(root, width=40, justify='left', state=tk.DISABLED)
results_file_path_entry.grid(row=1, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
results_file_button = tk.Button(root, text="Browse", command=lambda: browse_file(results_file_path_entry), state=tk.DISABLED)
results_file_button.grid(row=1, column=2, padx=pad_x, pady=pad_y)

def update_use_results_widgets():
    state = tk.NORMAL if use_results_file_var.get() else tk.DISABLED
    results_file_path_entry.config(state=state)
    results_file_button.config(state=state)
    
use_results_file_var.trace_add('write', lambda *args: update_use_results_widgets())

tk.Label(root, text="Kd (M^-1):").grid(row=3, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
Kd_entry = tk.Entry(root, justify='left')
Kd_entry.grid(row=3, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

tk.Label(root, text="h0 (M):").grid(row=4, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
h0_entry = tk.Entry(root, justify='left')
h0_entry.grid(row=4, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

tk.Label(root, text="g0 (M):").grid(row=5, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
g0_entry = tk.Entry(root, justify='left')
g0_entry.grid(row=5, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

tk.Label(root, text="Number of Fit Trials:").grid(row=6, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
fit_trials_entry = tk.Entry(root, justify='left')
fit_trials_entry.grid(row=6, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

tk.Label(root, text="RMSE Threshold Factor:").grid(row=7, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
rmse_threshold_entry = tk.Entry(root, justify='left')
rmse_threshold_entry.grid(row=7, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

tk.Label(root, text="R² Threshold:").grid(row=8, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
r2_threshold_entry = tk.Entry(root, justify='left')
r2_threshold_entry.grid(row=8, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

save_plots_var = tk.BooleanVar()
tk.Checkbutton(root, text="Save Plots To", variable=save_plots_var, command=lambda: update_save_plot_widgets()).grid(row=9, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)

results_dir_entry = tk.Entry(root, width=40, state=tk.DISABLED, justify='left')
results_dir_entry.grid(row=9, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
results_dir_button = tk.Button(root, text="Browse", command=lambda: browse_file(results_dir_entry), state=tk.DISABLED)
results_dir_button.grid(row=9, column=2, padx=pad_x, pady=pad_y)

def update_save_plot_widgets():
    state = tk.NORMAL if save_plots_var.get() else tk.DISABLED
    results_dir_entry.config(state=state)
    results_dir_button.config(state=state)

save_plots_var.trace_add('write', lambda *args: update_save_plot_widgets())

display_plots_var = tk.BooleanVar()
tk.Checkbutton(root, text="Display Plots", variable=display_plots_var).grid(row=10, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)

tk.Button(root, text="Run Fitting", command=run_fitting).grid(row=11, column=0, columnspan=3, pady=10, padx=pad_x)

# Start the main loop
root.mainloop()