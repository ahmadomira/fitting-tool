# Standard library imports
import os
from datetime import datetime

# Third-party imports
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.stats import linregress, ttest_1samp, t
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Local imports
from pltstyle import create_plots

# Function to calculate the 95% prediction interval upper and lower bounds
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

def unique_filename(file):
    base, extension = os.path.splitext(file)
    counter = 1
    file = f"{base}{extension}"
    while os.path.exists(file):
        file = f"{base}_{counter}{extension}"
        counter += 1
    return file

# Load data from the file
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return lines
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Function to split the data into replicas based on "var" or concentration reset (0.0 value)
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
    
    return np.array(replicas)

# Perform linear fit for each replica and collect results
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

# Calculate statistics for fitting results and filter outliers
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

# Round to four significant figures
def round_to_sigfigs(value, sigfigs=4):
    if isinstance(value, (int, float)):
        return f"{value:.{sigfigs}g}"
    return value

# Main function to perform the fitting and plotting
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
    colors = plt.cm.Dark2(np.linspace(0, 1, len(replicas)))
    
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

# Tkinter UI
class DyeAloneFittingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IDA Fitting Replica Dye Alone")
        
        # Variables
        self.file_path_var = tk.StringVar()
        self.save_path_var = tk.StringVar()
        self.save_plots_var = tk.BooleanVar()
        self.display_plots_var = tk.BooleanVar()
        self.plots_dir_var = tk.StringVar()
        
        # Set default values
        self.display_plots_var.set(True)
        
        tk.Label(root, text="Input File:").grid(row=0, column=0, sticky=tk.W)
        self.file_path_entry = tk.Entry(root, textvariable=self.file_path_var, width=50)
        self.file_path_entry.grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_input_file).grid(row=0, column=2)

        tk.Label(root, text="Save Result To:").grid(row=1, column=0, sticky=tk.W)
        self.save_path_entry = tk.Entry(root, textvariable=self.save_path_var, width=50)
        self.save_path_entry.grid(row=1, column=1)
        tk.Button(root, text="Browse", command=self.browse_save_path).grid(row=1, column=2)

        tk.Checkbutton(root, text="Save Plots", variable=self.save_plots_var, command=self.update_save_plot_widgets).grid(row=2, column=0, sticky=tk.W)
        self.plots_dir_entry = tk.Entry(root, textvariable=self.plots_dir_var, width=50, state=tk.DISABLED)
        self.plots_dir_entry.grid(row=2, column=1)
        self.plots_dir_button = tk.Button(root, text="Browse", command=self.browse_plots_dir, state=tk.DISABLED)
        self.plots_dir_button.grid(row=2, column=2)

        tk.Checkbutton(root, text="Display Plots", variable=self.display_plots_var).grid(row=3, column=0, columnspan=3, sticky=tk.W)

        tk.Button(root, text="Run Fitting", command=self.run_fitting).grid(row=4, column=1, pady=10)
        self.info_label = None

        self.save_plots_var.trace_add('write', lambda *args: self.update_save_plot_widgets())

    def show_message(self, message, is_error=False):
        if self.info_label:
            self.info_label.destroy()
        fg_color = 'red' if is_error else 'green'
        self.info_label = tk.Label(self.root, text=message, fg=fg_color)
        self.info_label.grid(row=5, column=0, columnspan=3, pady=10)

    def browse_input_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_var.set(file_path)
            root_dir = os.path.dirname(file_path)
            self.save_path_var.set(os.path.join(root_dir, f"dye_alone_results.txt"))
            self.plots_dir_var.set(os.path.join(root_dir))

    def browse_save_path(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
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
            perform_fitting(input_path, output_path, save_plots, display_plots, plots_dir)
            self.show_message(f"Results saved to: {output_path}")
        except Exception as e:
            self.show_message(f"Error: {str(e)}", is_error=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = DyeAloneFittingApp(root)
    root.mainloop()