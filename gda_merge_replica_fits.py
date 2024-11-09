import os
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from datetime import datetime

# User-defined variables for threshold adjustments
results_dir = '/Users/ahmadomira/Downloads/interface_test/untitled folder'  # Adjust this path as needed
outlier_relative_threshold = 0.25  # Relative threshold for outlier detection as a fraction of the median
rmse_threshold_factor = 3  # Number of standard deviations for RMSE filtering
kg_threshold_factor = 3  # Number of standard deviations for Kg filtering

#################################################################################################################
# Do NOT change code after this line
#################################################################################################################


def load_replica_data(file_path):
    # Initialize dictionary to store parsed data
    data = {
        'g0': None,
        'h0': None,
        'Kd': None,
        'concentrations': [],
        'signals': [],
        'median_params': {
            'Kg': None,
            'I0': None,
            'Id': None,
            'Ihd': None,
        },
        'rmse': None,
        'r_squared': None
    }

    # Open and read the file line by line
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Flags to identify sections in the file
    in_input_section = False
    in_original_data_section = False
    in_median_params_section = False

    for line in lines:
        # Check for Input section and retrieve g0, h0, and Kd
        if "Input:" in line:
            in_input_section = True
            continue  # Skip to next line

        if in_input_section:
            if "g0 (M):" in line:
                data['g0'] = float(re.sub(r'[^\d.eE+-]', '', line.split(":")[1]))
            elif "h0 (M):" in line:
                data['h0'] = float(re.sub(r'[^\d.eE+-]', '', line.split(":")[1]))
            elif "Kd (M^-1):" in line:
                data['Kd'] = float(re.sub(r'[^\d.eE+-]', '', line.split(":")[1]))
            elif line.strip() == "":
                in_input_section = False  # End of Input section

        # Check for Original Data section and retrieve concentrations and signals
        if "Original Data:" in line:
            in_original_data_section = True
            continue  # Skip to next line

        if in_original_data_section:
            if "Concentration" in line:
                continue  # Skip header line
            elif line.strip() == "":
                in_original_data_section = False  # End of Original Data section
            else:
                parts = line.split()
                try:
                    data['concentrations'].append(float(parts[0]))  # First column is concentration
                    data['signals'].append(float(parts[1]))  # Second column is signal
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse data line: {line.strip()}")

        # Check for Median Parameters section and retrieve Kg, I0, Id, Ihd
        if "Median parameters:" in line:
            in_median_params_section = True
            continue  # Skip to next line

        if in_median_params_section:
            if "Kg (M^-1):" in line:
                data['median_params']['Kg'] = float(re.sub(r'[^\d.eE+-]', '', line.split(":")[1]))
            elif "I0:" in line:
                data['median_params']['I0'] = float(re.sub(r'[^\d.eE+-]', '', line.split(":")[1]))
            elif "Id (signal/M):" in line:
                data['median_params']['Id'] = float(re.sub(r'[^\d.eE+-]', '', line.split(":")[1]))
            elif "Ihd (signal/M):" in line:
                data['median_params']['Ihd'] = float(re.sub(r'[^\d.eE+-]', '', line.split(":")[1]))
            elif "RMSE:" in line:
                data['rmse'] = float(re.sub(r'[^\d.eE+-]', '', line.split(":")[1]))
            elif "R²:" in line:
                data['r_squared'] = float(re.sub(r'[^\d.eE+-]', '', line.split(":")[1]))
            elif line.strip() == "":
                in_median_params_section = False  # End of Median Parameters section

    # Convert concentrations and signals lists to numpy arrays for further processing
    data['concentrations'] = np.array(data['concentrations'])
    data['signals'] = np.array(data['signals'])

    return data

# Function to compute the Signal for given parameters and g0 values
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

        except Exception as e:
            print(f"Error in compute_signal for d0={d0}: {e}")
            Signal_values.append(np.nan)

    return np.array(Signal_values)

# Function to save the plot as a PNG file
def save_plot(fig, filename, results_dir):
    plot_file = os.path.join(results_dir, filename)
    fig.savefig(plot_file, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")

# Function to calculate RMSE and R² for model accuracy assessment
def calculate_fit_metrics(observed, computed):
    rmse = np.sqrt(np.nanmean((observed - computed) ** 2))
    ss_res = np.nansum((observed - computed) ** 2)
    ss_tot = np.nansum((observed - np.nanmean(observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return rmse, r_squared

# Function to detect outliers based on relative deviation from the cross-replica median at each data point
def detect_outliers_per_point(data, reference, relative_threshold):
    deviations = np.abs(data - reference)
    outlier_indices = np.where(deviations > relative_threshold * reference)[0]
    return outlier_indices

# Function to export averaged data and fitting results to a text file with replica retention indication
def export_averaged_data(avg_concentrations, avg_signals, avg_fitting_curve_x, avg_fitting_curve_y, avg_params, stdev_params, rmse, r_squared, results_dir, input_values, retained_replicas_info):
    averaged_data_file = os.path.join(results_dir, "averaged_fit_results.txt")
    with open(averaged_data_file, 'w') as f:
        f.write("Input:\n")
        for key, value in input_values.items():
            f.write(f"{key}: {value}\n")
        f.write("\nRetained Replicas:\n")
        f.write("Replica\tKg (M^-1)\tI0\tId (signal/M)\tIhd (signal/M)\tRMSE\tR²\n")
        for replica_info in retained_replicas_info:
            original_index, params, fit_rmse, fit_r2 = replica_info
            f.write(f"{original_index}\t{params[1] * 1e6:.2e}\t{params[0]:.2e}\t{params[2] * 1e6:.2e}\t{params[3] * 1e6:.2e}\t{fit_rmse:.3f}\t{fit_r2:.3f}\n")
        f.write("\nOutput:\nAveraged Parameters:\n")
        f.write(f"Kg: {avg_params[1]:.2e} M^-1 (STDEV: {stdev_params[1]:.2e})\n")
        f.write(f"I0: {avg_params[0]:.2e} (STDEV: {stdev_params[0]:.2e})\n")
        f.write(f"Id: {avg_params[2]:.2e} signal/M (STDEV: {stdev_params[2]:.2e})\n")
        f.write(f"Ihd: {avg_params[3]:.2e} signal/M (STDEV: {stdev_params[3]:.2e})\n")
        f.write(f"RMSE: {rmse:.3f}\nR²: {r_squared:.3f}\n")
        f.write("\nAveraged Data:\nConcentration (M)\tSignal\n")
        for conc, signal in zip(avg_concentrations, avg_signals):
            f.write(f"{conc:.6e}\t{signal:.6e}\n")
        f.write("\nAveraged Fitting Curve:\nSimulated Concentration (M)\tSimulated Signal\n")
        for x_fit, y_fit in zip(avg_fitting_curve_x, avg_fitting_curve_y):
            f.write(f"{x_fit:.6e}\t{y_fit:.6e}\n")
        f.write(f"\nDate of Export: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Averaged data and fitting results saved to {averaged_data_file}")

# Load replica data
replica_files = [f for f in os.listdir(results_dir) if f.startswith('fit_results_replica_')]
replicas = [load_replica_data(os.path.join(results_dir, f)) for f in replica_files]

# Calculate median signal per concentration across replicas
median_signals_per_point = np.median([replica['signals'] for replica in replicas], axis=0)

# Initialize list to store signals for averaging that exclude outliers
filtered_signals_per_point = [[] for _ in range(len(median_signals_per_point))]

# Plot each replica's data and fit curves before filtering, marking outliers
plt.figure(figsize=(10, 6))
colors = plt.cm.jet(np.linspace(0, 1, len(replicas)))

for idx, replica in enumerate(replicas):
    concentrations = np.array(replica['concentrations'])
    signals = np.array(replica['signals'])
    Kd = replica['Kd']
    h0 = replica['h0']
    g0 = replica['g0']

    # Detect outliers for the current replica
    outlier_indices = detect_outliers_per_point(signals, median_signals_per_point, outlier_relative_threshold)

    # Add non-outliers to filtered signals
    for i, signal in enumerate(signals):
        if i not in outlier_indices:
            filtered_signals_per_point[i].append(signal)

    # Generate fitting line for each replica
    median_params = [
        replica['median_params']['I0'],
        replica['median_params']['Kg'],
        replica['median_params']['Id'],
        replica['median_params']['Ihd']
    ]
    fitting_curve_x, fitting_curve_y = [], []
    for i in range(len(concentrations) - 1):
        extra_points = np.linspace(concentrations[i], concentrations[i + 1], 21)
        fitting_curve_x.extend(extra_points)
        fitting_curve_y.extend(compute_signal(median_params, extra_points, Kd, h0, g0))

    last_signal = compute_signal(median_params, [concentrations[-1]], Kd, h0, g0)[0]
    fitting_curve_x.append(concentrations[-1])
    fitting_curve_y.append(last_signal)

    plt.plot(concentrations, signals, 'o', color=colors[idx], label=f'Replica {idx + 1} Data')
    plt.plot(fitting_curve_x, fitting_curve_y, '--', color=colors[idx], alpha=0.7, label=f'Replica {idx + 1} Fit')
    if len(outlier_indices) > 0:
        plt.plot(concentrations[outlier_indices], signals[outlier_indices], 'x', color=colors[idx], markersize=8, label=f"Replica {idx + 1} Outliers")

plt.xlabel('Concentration (M)')
plt.ylabel('Signal')
plt.title('All Replica Data and Fitting Curves with Outlier Detection')
plt.legend(loc='best')
plt.grid(True)
fig = plt.gcf()
save_plot(fig, "all_replicas_fitting_plot_with_outliers.png", results_dir)
plt.show()

# Further filtering based on RMSE and Kg deviation from the average
valid_replicas = []
for idx, replica in enumerate(replicas):
    if 'g0' in replica and 'h0' in replica and 'Kd' in replica and 'median_params' in replica:
        median_params = [
            replica['median_params']['I0'],
            replica['median_params']['Kg'],
            replica['median_params']['Id'],
            replica['median_params']['Ihd']
        ]
        computed_signals = compute_signal(median_params, replica['concentrations'], replica['Kd'], replica['h0'], replica['g0'])
        rmse, r_squared = calculate_fit_metrics(replica['signals'], computed_signals)
        valid_replicas.append((idx + 1, replica, median_params, rmse, r_squared))

# Calculate thresholds for RMSE and Kg
mean_rmse = np.mean([v[3] for v in valid_replicas])
std_rmse = np.std([v[3] for v in valid_replicas])
mean_kg = np.mean([v[2][1] for v in valid_replicas])
std_kg = np.std([v[2][1] for v in valid_replicas])
rmse_threshold = mean_rmse + rmse_threshold_factor * std_rmse
kg_threshold = mean_kg + kg_threshold_factor * std_kg

retained_replicas = [
    (original_index, replica, params, rmse, r2)
    for original_index, replica, params, rmse, r2 in valid_replicas
    if rmse <= rmse_threshold and abs(params[1] - mean_kg) <= kg_threshold
]

# Confirm retained replicas
print("\nFinal Retained Replicas:")
for original_index, _, params, rmse, r2 in retained_replicas:
    print(f"Replica {original_index}: RMSE = {rmse:.2f}, R² = {r2:.3f}, Kg = {params[1]:.2e}")

# Check for and print information of replicas not retained
excluded_replicas = [
    (original_index, replica, params, rmse, r2)
    for original_index, replica, params, rmse, r2 in valid_replicas
    if (rmse > rmse_threshold) or (abs(params[1] - mean_kg) > kg_threshold)
]

if excluded_replicas:
    print("\nReplicas Not Retained:")
    for original_index, _, params, rmse, r2 in excluded_replicas:
        print(f"Replica {original_index}: RMSE = {rmse:.2f}, R² = {r2:.3f}, Kg = {params[1]:.2e}")
else:
    print("\nNo replicas were excluded; all replicas met the retention criteria.")

# Calculate averages for retained replicas
avg_concentrations = np.mean([r[1]['concentrations'] for r in retained_replicas], axis=0)
avg_signals = np.array([np.nanmean(filtered_signals_per_point[i]) for i in range(len(filtered_signals_per_point))])
avg_params = [
    np.nanmean([params[0] for _, _, params, _, _ in retained_replicas]),
    np.nanmean([params[1] for _, _, params, _, _ in retained_replicas]),
    np.nanmean([params[2] for _, _, params, _, _ in retained_replicas]),
    np.nanmean([params[3] for _, _, params, _, _ in retained_replicas])
]
stdev_params = [
    np.nanstd([params[0] for _, _, params, _, _ in retained_replicas]),
    np.nanstd([params[1] for _, _, params, _, _ in retained_replicas]),
    np.nanstd([params[2] for _, _, params, _, _ in retained_replicas]),
    np.nanstd([params[3] for _, _, params, _, _ in retained_replicas])
]

# Generate averaged fitting curve for plotting and export
avg_fitting_curve_x, avg_fitting_curve_y = [], []
for i in range(len(avg_concentrations) - 1):
    extra_points = np.linspace(avg_concentrations[i], avg_concentrations[i + 1], 21)
    avg_fitting_curve_x.extend(extra_points)
    avg_fitting_curve_y.extend(compute_signal(avg_params, extra_points, retained_replicas[0][1]['Kd'], retained_replicas[0][1]['h0'], retained_replicas[0][1]['g0']))

computed_signals_at_avg_conc = compute_signal(avg_params, avg_concentrations, retained_replicas[0][1]['Kd'], retained_replicas[0][1]['h0'], retained_replicas[0][1]['g0'])
rmse, r_squared = calculate_fit_metrics(avg_signals, computed_signals_at_avg_conc)

# Plot averaged data and fitting curve after filtering
plt.figure(figsize=(10, 6))
plt.plot(avg_concentrations, avg_signals, 'o', label='Averaged Data', color='black')
plt.plot(avg_fitting_curve_x, avg_fitting_curve_y, '--', color='red', linewidth=1, label='Averaged Fit')

param_text = (f"Averaged Parameters:\nKg: {avg_params[1]:.2e} M^-1 (STDEV: {stdev_params[1]:.2e})\n"
              f"I0: {avg_params[0]:.2e} (STDEV: {stdev_params[0]:.2e})\n"
              f"Id: {avg_params[2]:.2e} signal/M (STDEV: {stdev_params[2]:.2e})\n"
              f"Ihd: {avg_params[3]:.2e} signal/M (STDEV: {stdev_params[3]:.2e})\n"
              f"RMSE: {rmse:.3f}\nR²: {r_squared:.3f}")
plt.gca().annotate(param_text, xy=(1.05, 0.5), xycoords='axes fraction', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))

plt.xlabel('Concentration (M)')
plt.ylabel('Signal')
plt.title('Averaged Data and Fitting Curve')
plt.legend()
plt.grid(True)
fig = plt.gcf()
save_plot(fig, "averaged_fitting_plot.png", results_dir)
plt.show()

def export_averaged_data(
    avg_concentrations, avg_signals, avg_fitting_curve_x, avg_fitting_curve_y, 
    avg_params, stdev_params, rmse, r_squared, results_dir, input_values, retained_replicas_info
):
    averaged_data_file = os.path.join(results_dir, "averaged_fit_results.txt")
    
    with open(averaged_data_file, 'w') as f:
        # Input section
        f.write("Input:\n")
        f.write(f"g0 (M): {input_values['g0 (M)']:.6e}\n")
        f.write(f"h0 (M): {input_values['h0 (M)']:.6e}\n")
        f.write(f"Kd (M^-1): {input_values['Kd (M^-1)']:.6e}\n")

        # Output - Averaged Parameters
        f.write("\nOutput:\nAveraged Parameters:\n")
        f.write(f"Kg (M^-1): {avg_params[1]:.2e}\n")
        f.write(f"I0: {avg_params[0]:.2e}\n")
        f.write(f"Id (signal/M): {avg_params[2]:.2e}\n")
        f.write(f"Ihd (signal/M): {avg_params[3]:.2e}\n")
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"R²: {r_squared:.3f}\n")

        # Standard Deviations for Averaged Parameters
        f.write("\nStandard Deviations:\n")
        f.write(f"Kg Std Dev (M^-1): {stdev_params[1]:.2e}\n")  
        f.write(f"I0 Std Dev: {stdev_params[0]:.2e}\n")
        f.write(f"Id Std Dev (signal/M): {stdev_params[2]:.2e}\n")
        f.write(f"Ihd Std Dev (signal/M): {stdev_params[3]:.2e}\n")

        # Averaged Data section
        f.write("\nAveraged Data:\nConcentration d0 (M)\tSignal\n")
        for conc, signal in zip(avg_concentrations, avg_signals):
            f.write(f"{conc:.6e}\t{signal:.6e}\n")

        # Averaged Fitting Curve section
        f.write("\nAveraged Fitting Curve:\nSimulated Concentration d0 (M)\tSimulated Signal\n")
        for x_fit, y_fit in zip(avg_fitting_curve_x, avg_fitting_curve_y):
            f.write(f"{x_fit:.6e}\t{y_fit:.6e}\n")

        # Retained Replicas section, positioned near the end
        f.write("\nRetained Replicas:\n")
        f.write("Replica\tKg (M^-1)\tI0\tId (signal/M)\tIhd (signal/M)\tRMSE\tR²\n")
        for original_index, params, fit_rmse, fit_r2 in retained_replicas_info:
            f.write(f"{original_index}\t{params[1] * 1e6:.2e}\t{params[0]:.2e}\t{params[2] * 1e6:.2e}\t{params[3] * 1e6:.2e}\t{fit_rmse:.3f}\t{fit_r2:.3f}\n")

        # Date of Export
        f.write(f"\nDate of Export: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    print(f"Averaged data and fitting results saved to {averaged_data_file}")

# Example function call
export_averaged_data(
    avg_concentrations, avg_signals, avg_fitting_curve_x, avg_fitting_curve_y,
    avg_params, stdev_params, rmse, r_squared, results_dir,
    {
        'g0 (M)': retained_replicas[0][1]['g0'],
        'h0 (M)': retained_replicas[0][1]['h0'],
        'Kd (M^-1)': retained_replicas[0][1]['Kd']
    },
    [(original_index, params, rmse, r2) for original_index, _, params, rmse, r2 in retained_replicas]
)