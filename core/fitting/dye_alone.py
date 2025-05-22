"""
Dye Alone fitting logic extracted from the GUI for reuse and testing.
"""
import os
from datetime import datetime
import numpy as np
from scipy.stats import linregress, ttest_1samp, t
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utils.plot_utils import create_plots
from core.fitting_utils import unique_filename

def prediction_interval(data, avg_value):
    n = len(data)
    mean = np.mean(data)
    if n > 1:
        std_dev = np.std(data, ddof=1)
        margin_of_error = std_dev * np.sqrt(1 + 1/n) * t.ppf(0.975, n-1)
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        lower_bound = min(lower_bound, avg_value * 0.75)
        upper_bound = max(upper_bound, avg_value * 1.25)
    else:
        margin_of_error = "not applicable"
        std_dev = "not applicable"
        lower_bound = "not applicable"
        upper_bound = "not applicable"
    return mean, lower_bound, upper_bound, std_dev

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return lines
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def split_replicas(data):
    if data is None:
        print("Data is None. Cannot split replicas.")
        return None
    replicas = []
    current_replica = []
    use_var_signal_split = False
    for line in data:
        if "var\tsignal" in line.lower():
            use_var_signal_split = True
            break
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
    if not replicas:
        print("No replicas detected.")
        return None
    return np.array(replicas) * 1e6

def fit_replicas(replicas):
    slopes = []
    intercepts = []
    retained_results = []
    for i, replica in enumerate(replicas):
        x_values = replica[:, 0]
        y_values = replica[:, 1]
        slope, intercept, _, _, _ = linregress(x_values, y_values)
        slopes.append(slope)
        intercepts.append(intercept)
        retained_results.append((slope, intercept))
    return retained_results

def filter_and_average_results(results, significance_level=0.05):
    slopes = np.array([result[0] for result in results])
    intercepts = np.array([result[1] for result in results])
    retained_indices = list(range(len(slopes)))
    if len(slopes) > 1:
        t_stat_slope, p_value_slope = ttest_1samp(slopes, slopes.mean())
        t_stat_intercept, p_value_intercept = ttest_1samp(intercepts, intercepts.mean())
        if p_value_slope <= significance_level or p_value_intercept <= significance_level:
            retained_indices = [
                i for i, (slope, intercept) in enumerate(zip(slopes, intercepts))
                if abs(slope - slopes.mean()) < slopes.std() and abs(intercept - intercepts.mean()) < intercepts.std()
            ]
    retained_slopes = slopes[retained_indices]
    retained_intercepts = intercepts[retained_indices]
    avg_slope = np.mean(retained_slopes)
    avg_intercept = np.mean(retained_intercepts)
    return avg_slope, avg_intercept, retained_slopes, retained_intercepts, retained_indices

def round_to_sigfigs(value, sigfigs=4):
    if isinstance(value, (int, float)):
        return f"{value:.{sigfigs}g}"
    return value

def perform_fitting(input_file_path, output_file_path, save_plots, display_plots, plots_dir):
    if not output_file_path.endswith(".txt"):
        output_file_path += ".txt"
    data_lines = load_data(input_file_path)
    if data_lines is None or len(data_lines) == 0:
        raise ValueError("Data loading failed or data is empty.")
    replicas = split_replicas(data_lines)
    if replicas is None:
        raise ValueError("No replicas detected.")
    fit_results = fit_replicas(replicas)
    avg_slope, avg_intercept, retained_slopes, retained_intercepts, retained_indices = filter_and_average_results(fit_results)
    Id_mean, Id_lower_bound, Id_upper_bound, Id_stdev = prediction_interval(retained_slopes, avg_slope)
    I0_mean, I0_lower_bound, I0_upper_bound, I0_stdev = prediction_interval(retained_intercepts, avg_intercept)
    if len(retained_slopes) == 1:
        Id_mean = retained_slopes[0]
        I0_mean = retained_intercepts[0]
        Id_lower_bound = "not applicable"
        Id_upper_bound = "not applicable"
        I0_lower_bound = "not applicable"
        I0_upper_bound = "not applicable"
        Id_stdev = "not applicable"
        I0_stdev = "not applicable"
    Id_lower_bound = round_to_sigfigs(Id_lower_bound)
    Id_upper_bound = round_to_sigfigs(Id_upper_bound)
    Id_stdev = round_to_sigfigs(Id_stdev)
    I0_lower_bound = round_to_sigfigs(I0_lower_bound)
    I0_upper_bound = round_to_sigfigs(I0_upper_bound)
    I0_stdev = round_to_sigfigs(I0_stdev)
    fig, ax = create_plots()
    colors = plt.cm.jet(np.linspace(0, 1, len(replicas)))
    def scientific_notation(val, pos=0):
        return f'{val:.2e}'.replace('e', r'\cdot 10^{') + '}'
    formatter = FuncFormatter(scientific_notation)
    for i, replica in enumerate(replicas):
        x_values = replica[:, 0]
        y_values = replica[:, 1]
        slope, intercept = fit_results[i]
        ax.plot(x_values, y_values, 'o', color=colors[i], label=f'Replica {i+1} Data')
        y_fit = slope * x_values + intercept
        ax.plot(x_values, y_fit, '-', color=colors[i], label=f'Fit {i+1}: $Y = {formatter(slope)}X + {formatter(intercept)}$')
    x_fit = np.linspace(0, max(np.array([replica[:, 0] for replica in replicas]).flatten()), 100)
    y_fit = avg_slope * x_fit + avg_intercept
    ax.plot(x_fit, y_fit, '--', color='orange', linewidth=2, label=rf'Average Fit: $Y = {formatter(avg_slope)}X + {formatter(avg_intercept)}$')
    ax.set_title('Linear Fit of Signal vs. Concentration for Multiple Replicas')
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    if save_plots:
        plot_file = os.path.join(plots_dir, "dye_alone_fit_plot.png")
        fig.savefig(unique_filename(plot_file), bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    if display_plots:
        plt.show()
    total_replicas = len(fit_results)
    retained_replicas_count = len(retained_indices)
    print(f"{retained_replicas_count} out of {total_replicas} replicas were retained.")
    with open(unique_filename(output_file_path), 'w') as f:
        f.write("Linear Fit Results\n")
        f.write(f"Average Id\t{Id_mean:.3e}\n")
        f.write(f"Id prediction interval (95%) at least 25% above and below average value: [{Id_lower_bound}, {Id_upper_bound}]\n")
        f.write(f"Id Stdev: {Id_stdev}\n")
        f.write(f"Average I0\t{I0_mean:.3e}\n")
        f.write(f"I0 prediction interval (95%) at least 25% above and below average value: [{I0_lower_bound}, {I0_upper_bound}]\n")
        f.write(f"I0 Stdev: {I0_stdev}\n")
        f.write("\nRetained Individual Fits:\n")
        for i, (slope, intercept) in enumerate(zip(retained_slopes, retained_intercepts)):
            f.write(f"Replica {i+1}\tId: {slope:.3e}\tI0: {intercept:.3e}\n")
        f.write(f"\nDate of Export: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Results saved to {output_file_path}")
