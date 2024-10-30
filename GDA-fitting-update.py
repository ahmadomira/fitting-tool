import numpy as np
from scipy.optimize import brentq, minimize
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
import os

# File paths
file_path = '/Users/ahmadomira/git/automation_project/data/GDA_system.txt'
results_dir = os.path.dirname(file_path)  # Directory for input file
results_file_path = os.path.join(results_dir, 'dye_alone_linear_fit_results.txt')  # Results save path

# Input constants
Kd_in_M = 1.68e7  # Binding constant for h_d binding in M^-1
h0_in_M = 4.3e-6  # Initial host concentration (M)
g0_in_M = 6e-6    # Initial guest concentration (M)

#Fitting Thresholds
number_of_fit_trials = 200 # Set how often the fit is repeated with different random start guesses for Kg, I0, Id and Ihd
rmse_threshold_factor = 2  # Factor to multiply the RSME for acceptable fits compared to the best fit.
r2_threshold = 0.9    #R2 value used for filtering acceptable and unacceptable fits.

#################################################################################################################
# Do NOT change code after this line
#################################################################################################################

# Initialize parameter ranges for fitting
I0_range = (0, None)  # Default range for I0: (0, Minimum Signal in the data)
Id_range = (None, None)  # Default range for Id: (1e3, 1e18)
Ihd_range = (None, None)  # Default range for Ihd: (1e3, 1e18)

# Check and load boundaries from results file, if available
if os.path.exists(results_file_path):
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
        Id_lower, Id_upper = 1e3, 1e18  # Defaults if error occurs
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

# Print boundaries for verification
try:
    print(f"Loaded boundaries:\nId: [{Id_lower * 1e6:.3e}, {Id_upper * 1e6:.3e}] M⁻¹\nI0: [{I0_lower:.3e}, {I0_upper:.3e}]")
except TypeError:
    print(f"No boundaries loaded, defaulting to:\nId: [{Id_lower * 1e6:.3e}, {Id_upper * 1e6:.3e}] M⁻¹\nI0: [{I0_lower:.3e}, {I0_upper}]")

# Compute Signal for given parameters and d0 values (dye titration setup)
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

# Compute residuals for optimization
def residuals(params, d0_values, Signal_observed, Kd, h0, g0):
    Signal_computed = compute_signal(params, d0_values, Kd, h0, g0)
    return np.nan_to_num(Signal_observed - Signal_computed, nan=1e6)

# Calculate RMSE and R² metrics
def calculate_fit_metrics(Signal_observed, Signal_computed):
    rmse = np.sqrt(np.nanmean((Signal_observed - Signal_computed) ** 2))
    ss_res = np.nansum((Signal_observed - Signal_computed) ** 2)
    ss_tot = np.nansum((Signal_observed - np.nanmean(Signal_observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return rmse, r_squared

# Load data from input file
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.readlines()
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Split data into replicas based on "var" or concentration reset (0.0)
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

# Process each replica for fitting
data_lines = load_data(file_path)
if data_lines is None:
    raise ValueError("Data loading failed.")
replicas = split_replicas(data_lines)
if replicas is None:
    raise ValueError("Replica splitting failed.")
print(f"Number of replicas detected: {len(replicas)}")

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

    plot_file = os.path.join(results_dir, f"fit_plot_replica_{replica_index}.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.show()

    # Save results to a file
    replica_file = os.path.join(results_dir, f"fit_results_replica_{replica_index}.txt")
    with open(replica_file, 'w') as f:
        # Input section
        f.write("Input:\n")
        f.write(f"g0 (M): {g0_in_M:.6e}\n")
        f.write(f"h0 (M): {h0_in_M:.6e}\n")
        f.write(f"Kd (M^-1): {Kd_in_M:.6e}\n")
        f.write(f"Id lower bound (signal/M): {Id_lower * 1e6:.3e}\n")
        f.write(f"Id upper bound (signal/M): {Id_upper * 1e6:.3e}\n")
        f.write(f"I0 lower bound: {I0_lower:.3e}\n")
        f.write(f"I0 upper bound: {I0_upper:.3e}\n")
    
        # Output - Median parameters
        f.write("\nOutput:\nMedian parameters:\n")
        f.write(f"Kg (M^-1): {median_params[1] * 1e6:.2e}\n")
        f.write(f"I0: {median_params[0]:.2e}\n")
        f.write(f"Id (signal/M): {median_params[2] * 1e6:.2e}\n")
        f.write(f"Ihd (signal/M): {median_params[3] * 1e6:.2e}\n")
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"R²: {r_squared:.3f}\n")
    
        # Acceptable Fit Parameters
        f.write("\nAcceptable Fit Parameters:\n")
        f.write("Kg (M^-1)\tI0\tId (signal/M)\tIhd (signal/M)\tRMSE\tR²\n")
        for params, fit_rmse, fit_r2 in filtered_results:
            f.write(f"{params[1] * 1e6:.2e}\t{params[0]:.2e}\t{params[2] * 1e6:.2e}\t{params[3] * 1e6:.2e}\t{fit_rmse:.3f}\t{fit_r2:.3f}\n")
    
        # Calculate standard deviations for Kg, I0, Id, and Ihd if there are filtered results
        if filtered_results:
            Kg_values = [params[1] * 1e6 for params, _, _ in filtered_results]
            I0_values = [params[0] for params, _, _ in filtered_results]
            Id_values = [params[2] * 1e6 for params, _, _ in filtered_results]
            Ihd_values = [params[3] * 1e6 for params, _, _ in filtered_results]
    
            Kg_std = np.std(Kg_values)
            I0_std = np.std(I0_values)
            Id_std = np.std(Id_values)
            Ihd_std = np.std(Ihd_values)
        else:
            Kg_std = I0_std = Id_std = Ihd_std = np.nan  # Assign NaN if no filtered results
    
        # Standard Deviations section
        f.write("\nStandard Deviations:\n")
        f.write(f"Kg Std Dev (M^-1): {Kg_std:.2e}\n")  
        f.write(f"I0 Std Dev: {I0_std:.2e}\n")
        f.write(f"Id Std Dev (signal/M): {Id_std:.2e}\n")
        f.write(f"Ihd Std Dev (signal/M): {Ihd_std:.2e}\n")
        
        # Original Data section
        f.write("\nOriginal Data:\nConcentration do (M)\tSignal\n")
        for d0, signal in zip(d0_values / 1e6, Signal_observed):  
            f.write(f"{d0:.6e}\t{signal:.6e}\n")
        
        # Fitting Curve section
        f.write("\nFitting Curve:\n")
        f.write("Simulated Concentration (M)\tSimulated Signal\n")
        for x_sim, y_sim in zip(np.array(fitting_curve_x) / 1e6, fitting_curve_y):
            f.write(f"{x_sim:.6e}\t{y_sim:.6e}\n")
        
        # Date of Export
        f.write(f"\nDate of Export: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Results for Replica {replica_index} saved to {replica_file}")