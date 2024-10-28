import queue
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
from scipy.optimize import brentq, minimize
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

import matplotlib.pyplot as plt
from pltstyle import create_plots

# TODO: add check and load boundries from result file if available
# # Check and load boundaries from results file, if available
# if os.path.exists(results_file_path):
#     try:
#         with open(results_file_path, 'r') as f:
#             lines = f.readlines()

#         # Extract Id prediction interval
#         id_prediction_line = next((line for line in lines if 'Id prediction interval' in line), None)
#         if id_prediction_line and 'not applicable' not in id_prediction_line:
#             Id_lower = float(id_prediction_line.split('[')[-1].split(',')[0].strip())
#             Id_upper = float(id_prediction_line.split(',')[-1].split(']')[0].strip())
#         else:
#             average_Id = float(next(line for line in lines if 'Average Id' in line).split('\t')[-1].strip())
#             Id_lower = 0.5 * average_Id
#             Id_upper = 2.0 * average_Id

#         # Extract I0 prediction interval
#         i0_prediction_line = next((line for line in lines if 'I0 prediction interval' in line), None)
#         if i0_prediction_line and 'not applicable' not in i0_prediction_line:
#             I0_lower = float(i0_prediction_line.split('[')[-1].split(',')[0].strip())
#             I0_upper = float(i0_prediction_line.split(',')[-1].split(']')[0].strip())
#         else:
#             average_I0 = float(next(line for line in lines if 'Average I0' in line).split('\t')[-1].strip())
#             I0_lower = 0.5 * average_I0
#             I0_upper = 2.0 * average_I0

#     except Exception as e:
#         print(f"Error parsing boundaries from the results file: {e}")
#         Id_lower, Id_upper = 1e3, 1e18  # Defaults if error occurs
#         I0_lower, I0_upper = 0, None
# else:
#     Id_lower, Id_upper = 1e3, 1e18
#     I0_lower, I0_upper = 0, None
    
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

# Function to perform fitting
def perform_fitting(file_path, Kd_in_M, h0_in_M, g0_in_M, rmse_threshold_factor, r2_threshold):
    
    # TODO: this is temporary, should be loaded from results file if available
    Id_lower, Id_upper = 1e3, 1e18
    I0_lower, I0_upper = 0, None   
    
    # Initialize parameter ranges for fitting
    I0_range = (0, None)  # Default range for I0: (0, Minimum Signal in the data)
    Id_range = (None, None)  # Default range for Id: (1e3, 1e18)
    Ihd_range = (None, None)  # Default range for Ihd: (1e3, 1e18)

    # Convert bounds to µM⁻¹ for fitting
    Id_lower /= 1e6
    Id_upper /= 1e6
    Ihd_lower = Ihd_range[0] / 1e6 if Ihd_range[0] is not None else 0.001
    Ihd_upper = Ihd_range[1] / 1e6 if Ihd_range[1] is not None else 1e12
    
    # Convert constants to µM
    Kd = Kd_in_M / 1e6
    h0 = h0_in_M * 1e6
    g0 = g0_in_M * 1e6    
    
    data_lines = load_data(file_path)
    if data_lines is None:
        raise ValueError("Data loading failed.")
    replicas = split_replicas(data_lines)
    if replicas is None:
        raise ValueError("Replica splitting failed.")
    print(f"Number of replicas detected: {len(replicas)}")

    results = []
    total_replicas = len(replicas)
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
        # TODO: 20 only for testing, should be 200
        for _ in range(10):
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
        for i, initial_params in enumerate(initial_params_list):
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
    
        fig, ax = create_plots(figsize=(8, 6))
        ax.plot(d0_values, Signal_observed, 'o', label='Observed Signal')
        ax.plot(fitting_curve_x, fitting_curve_y, '--', color='blue', alpha=0.6, label='Simulated Fitting Curve')
        ax.set_xlabel(r'$D_0$ $\rm{[\mu M]}$')
        ax.set_title(f'Observed vs. Simulated Fitting Curve for Replica {replica_index}')
        ax.legend()
        ax.grid(True)

        # Annotate plot with median parameter values and fit metrics
        param_text = (f"$K_g$: {median_params[1] * 1e6:.2e} $M^{-1}$\n"
                  f"$I_0$: {median_params[0]:.2e}\n"
                  f"$I_d$: {median_params[2] * 1e6:.2e} signal/M\n"
                  f"$I_{{hd}}$: {median_params[3] * 1e6:.2e} signal/M\n"
                  f"RMSE: {rmse:.3f}\n"
                  f"$R^2$: {r_squared:.3f}")


        ax.annotate(param_text, xy=(0.95, 0.05), xycoords='axes fraction', fontsize=10,
                ha='right', va='bottom', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))

        results.append((fig, ax, replica_index))

    return results

# Tkinter GUI
class GDAFittingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GDA Fitting Update")

        self.file_path = tk.StringVar(value="/Users/ahmadomira/Downloads/interface_test/GDA_system.txt")
        self.Kd_in_M = tk.DoubleVar(value=1.68e7)
        self.h0_in_M = tk.DoubleVar(value=4.3e-6)
        self.g0_in_M = tk.DoubleVar(value=6e-6)
        self.rmse_threshold_factor = tk.DoubleVar(value=2)
        self.r2_threshold = tk.DoubleVar(value=0.9)

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="File Path:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.file_path, width=50).grid(row=0, column=1, columnspan=2, sticky=tk.W)
        tk.Button(self.root, text="Browse", command=self.browse_file).grid(row=0, column=3, sticky=tk.W)

        tk.Label(self.root, text="Kd (M^-1):").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.Kd_in_M).grid(row=1, column=1, sticky=tk.W)

        tk.Label(self.root, text="h0 (M):").grid(row=2, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.h0_in_M).grid(row=2, column=1, sticky=tk.W)

        tk.Label(self.root, text="g0 (M):").grid(row=3, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.g0_in_M).grid(row=3, column=1, sticky=tk.W)

        tk.Label(self.root, text="RMSE Threshold Factor:").grid(row=4, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.rmse_threshold_factor).grid(row=4, column=1, sticky=tk.W)

        tk.Label(self.root, text="R² Threshold:").grid(row=5, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.r2_threshold).grid(row=5, column=1, sticky=tk.W)

        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(row=6, column=0, columnspan=4, pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path.set(file_path)

    def run_fitting(self):
        file_path = self.file_path.get()
        Kd_in_M = self.Kd_in_M.get()
        h0_in_M = self.h0_in_M.get()
        g0_in_M = self.g0_in_M.get()
        rmse_threshold_factor = self.rmse_threshold_factor.get()
        r2_threshold = self.r2_threshold.get()

        if not file_path:
            messagebox.showerror("Error", "Please select a file.")
            return
        
        try:
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Progress")
            progress_label = tk.Label(progress_window, text="Fitting in progress, please wait...")
            progress_label.pack(pady=20, padx=20)
            self.root.update()

            results = perform_fitting(file_path, Kd_in_M, h0_in_M, g0_in_M, rmse_threshold_factor, r2_threshold)
            progress_window.destroy()
            self.show_results(results)
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("Error", str(e))

    def show_results(self, results):
        for fig, ax, replica_index in results:
            result_window = tk.Toplevel(self.root)
            result_window.title(f"Replica {replica_index} Fitting Results")

            canvas = FigureCanvasTkAgg(fig, master=result_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = GDAFittingApp(root)
    root.mainloop()