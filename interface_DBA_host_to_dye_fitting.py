import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
from pltstyle import create_plots

# Add number_of_fit_trials to function parameters
def run_dba_fitting(file_path, results_dir, d0_in_M, rmse_threshold_factor, r2_threshold, save_plots, display_plots, plots_dir, save_results, results_save_dir, number_of_fit_trials):
    # Convert initial concentration to µM units
    d0 = d0_in_M * 1e6  # d0 in µM

    # Initialize parameter ranges with default boundaries
    I0_lower, I0_upper = 0, None
    Id_lower, Id_upper = 1e3 / 1e6, 1e18 / 1e6  # Convert Id range to µM⁻¹

    # load boundaries from existing results file if available
    if results_dir:
        results_file_path = os.path.join(results_dir, 'dye_alone_linear_fit_results.txt')

        # Load prediction intervals for I0 and Id from existing results file if available
        if os.path.exists(results_file_path):
            try:
                with open(results_file_path, 'r') as f:
                    lines = f.readlines()

                # Extract Id prediction interval from the file if available
                id_prediction_line = next((line for line in lines if 'Id prediction interval' in line), None)
                if id_prediction_line and 'not applicable' not in id_prediction_line:
                    Id_lower = float(id_prediction_line.split('[')[-1].split(',')[0].strip()) / 1e6
                    Id_upper = float(id_prediction_line.split(',')[-1].split(']')[0].strip()) / 1e6
                else:
                    average_Id = float(next(line for line in lines if 'Average Id' in line).split('\t')[-1].strip()) / 1e6
                    Id_lower, Id_upper = 0.5 * average_Id, 2.0 * average_Id

                # Extract I0 prediction interval from the file if available
                i0_prediction_line = next((line for line in lines if 'I0 prediction interval' in line), None)
                if i0_prediction_line and 'not applicable' not in i0_prediction_line:
                    I0_lower = float(i0_prediction_line.split('[')[-1].split(',')[0].strip())
                    I0_upper = float(i0_prediction_line.split(',')[-1].split(']')[0].strip())
                else:
                    average_I0 = float(next(line for line in lines if 'Average I0' in line).split('\t')[-1].strip())
                    I0_lower, I0_upper = 0.5 * average_I0, 2.0 * average_I0

            except Exception as e:
                print(f"Error parsing boundaries from the results file: {e}")
                Id_lower, Id_upper = 1e3 / 1e6, 1e18 / 1e6  # Default bounds if parsing fails
                I0_lower, I0_upper = 0, np.inf
        else:
            # Set default ranges if no results file is present
            Id_lower, Id_upper = 1e3 / 1e6, 1e18 / 1e6
            I0_lower, I0_upper = 0, np.inf

        # Print boundary values for verification
        print(f"Loaded boundaries:\nId: [{Id_lower * 1e6:.3e}, {Id_upper * 1e6:.3e}] M⁻¹\nI0: [{I0_lower:.3e}, {I0_upper:.3e}]")

    # Load data from file
    def load_data(file_path):
        try:
            with open(file_path, 'r') as f:
                return f.readlines()
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")

    # Function to split data into replicas
    def split_replicas(data):
        replicas, current_replica = [], []
        for line in data:
            if "var" in line.lower():
                if current_replica:
                    replicas.append(np.array(current_replica))
                    current_replica = []
            else:
                try:
                    x, y = map(float, line.split())
                    if x == 0.0 and current_replica:
                        replicas.append(np.array(current_replica))
                        current_replica = []
                    current_replica.append((x, y))
                except ValueError:
                    continue

        if current_replica:
            replicas.append(np.array(current_replica))

        return replicas if replicas else None

    # Define function to calculate signal based on parameters and h0 values
    def compute_signal(params, h0_values, d0):
        I0, Kd, Id, Ihd = params
        Signal_values = []
        for h0 in h0_values:
            delta = h0 - d0
            a = Kd
            b = Kd * delta + 1
            c = -d0
            discriminant = b**2 - 4 * a * c

            if discriminant < 0:
                Signal_values.append(np.nan)
                continue

            sqrt_discriminant = np.sqrt(discriminant)
            d1 = (-b + sqrt_discriminant) / (2 * a)
            d2 = (-b - sqrt_discriminant) / (2 * a)

            d = d1 if d1 >= 0 else d2 if d2 >= 0 else np.nan
            if np.isnan(d):
                Signal_values.append(np.nan)
                continue

            h = d + delta
            hd = Kd * h * d
            Signal = I0 + Id * d + Ihd * hd
            Signal_values.append(Signal)

        return np.array(Signal_values)

    # Function to compute residuals
    def residuals(params, h0_values, Signal_observed, d0):
        Signal_computed = compute_signal(params, h0_values, d0)
        residual = Signal_observed - Signal_computed
        residual = np.nan_to_num(residual, nan=1e6)
        return residual

    # Function to calculate fit metrics
    def calculate_fit_metrics(Signal_observed, Signal_computed):
        rmse = np.sqrt(np.nanmean((Signal_observed - Signal_computed) ** 2))
        ss_res = np.nansum((Signal_observed - Signal_computed) ** 2)
        ss_tot = np.nansum((Signal_observed - np.nanmean(Signal_observed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        return rmse, r_squared

    # Main fitting process
    data_lines = load_data(file_path)
    if data_lines is None:
        raise ValueError("Data loading failed.")

    replicas = split_replicas(data_lines)
    if replicas is None:
        raise ValueError("Replica splitting failed.")

    print(f"Number of replicas detected: {len(replicas)}")

    # Process each replica
    figures = []  # List to store figures
    
    for replica_index, replica_data in enumerate(replicas, start=1):
        print(f"Processing replica {replica_index}, data length: {len(replica_data)}")
        h0_values = replica_data[:, 0] * 1e6  # Convert h0 values to µM
        Signal_observed = replica_data[:, 1]

        # Skip replicas with insufficient data points
        if len(h0_values) < 2:
            print(f"Replica {replica_index} has insufficient data. Skipping.")
            continue

        # Update I0_upper based on minimum observed signal if it was undefined
        I0_upper = np.min(Signal_observed) if I0_upper is None or np.isinf(I0_upper) else I0_upper

        # Generate initial parameter guesses within specified bounds
        Ihd_guess_smaller = Signal_observed[0] < Signal_observed[-1]
        initial_params_list = []
        for _ in range(number_of_fit_trials):  # <-- Change here
            I0_guess = np.random.uniform(I0_lower, I0_upper)
            Kd_guess = 10 ** np.random.uniform(-5, 5)
            if Ihd_guess_smaller:
                Id_guess = 10 ** np.random.uniform(np.log10(Id_lower), np.log10(Id_upper))
                Ihd_guess = Id_guess * np.random.uniform(0.1, 0.5)
            else:
                Ihd_guess = 10 ** np.random.uniform(np.log10(Id_lower), np.log10(Id_upper))
                Id_guess = Ihd_guess * np.random.uniform(0.1, 0.5)
            initial_params_list.append([I0_guess, Kd_guess, Id_guess, Ihd_guess])

        # Perform optimization and collect results for each initial guess
        best_result, best_cost = None, np.inf
        fit_results = []
        for initial_params in initial_params_list:
            result = minimize(lambda params: np.sum(residuals(params, h0_values, Signal_observed, d0) ** 2),
                            initial_params, method='L-BFGS-B',
                            bounds=[(I0_lower, I0_upper), (1e-8, 1e8), (Id_lower, Id_upper), (1e-8, 1e8)])

            Signal_computed = compute_signal(result.x, h0_values, d0)
            rmse, r_squared = calculate_fit_metrics(Signal_observed, Signal_computed)
            fit_results.append((result.x, result.fun, rmse, r_squared))

            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result

        # Filter fit results based on RMSE and R² thresholds
        best_rmse = min(fit_rmse for _, _, fit_rmse, _ in fit_results)
        rmse_threshold = best_rmse * rmse_threshold_factor
        #r2_threshold = 0.9

        filtered_results = [
            (params, fit_rmse, fit_r2) for params, _, fit_rmse, fit_r2 in fit_results
            if fit_rmse <= rmse_threshold and fit_r2 >= r2_threshold
        ]

        # Calculate the median of the filtered parameters if they exist
        if filtered_results:
            median_params = np.median(np.array([result[0] for result in filtered_results]), axis=0)
        else:
            # TODO: catch this warning in the interface
            print("Warning: No fits meet the filtering criteria.")
            continue

        # Compute signal and metrics for the median parameters
        Signal_computed = compute_signal(median_params, h0_values, d0)
        rmse, r_squared = calculate_fit_metrics(Signal_observed, Signal_computed)

        # Generate points for the fitting curve to overlay on observed data
        fitting_curve_x, fitting_curve_y = [], []
        for i in range(len(h0_values) - 1):
            extra_points = np.linspace(h0_values[i], h0_values[i + 1], 21)
            fitting_curve_x.extend(extra_points)
            fitting_curve_y.extend(compute_signal(median_params, extra_points, d0))

        last_signal = compute_signal(median_params, [h0_values[-1]], d0)[0]
        fitting_curve_x.append(h0_values[-1])
        fitting_curve_y.append(last_signal)

        # Plot observed vs. simulated fitting curve
        fig, ax = create_plots(x_label=r'$h_0$ $\rm{[\mu M]}$', y_label=r'Signal $\rm{[AU]}$')

        ax.plot(h0_values, Signal_observed, 'o', label='Observed Signal')
        ax.plot(fitting_curve_x, fitting_curve_y, '--', color='blue', alpha=0.6, label='Simulated Fitting Curve')
        ax.set_title(f'Observed vs. Simulated Fitting Curve for Replica {replica_index}')
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))

        param_text = (f"$K_d$: {median_params[1] * 1e6:.2e} $M^{{-1}}$\n"
                      f"$I_0$: {median_params[0]:.2e}\n"
                      f"$I_d$: {median_params[2] * 1e6:.2e} $M^{{-1}}$\n"
                      f"$I_{{hd}}$: {median_params[3] * 1e6:.2e} $M^{{-1}}$\n"
                      f"$RMSE$: {rmse:.3f}\n"
                      f"$R^2$: {r_squared:.3f}")

        ax.annotate(param_text, xy=(0.8, 0.04), xycoords='axes fraction', fontsize=10,
                    ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey", alpha=0.5))

        if save_plots:
            plot_file = os.path.join(plots_dir, f"fit_plot_replica_{replica_index}.png")
            fig.savefig(plot_file, bbox_inches='tight')
            print(f"Plot saved to {plot_file}")

        figures.append(fig)

        # Calculate RMSE and R² for the median fit
        median_rmse, median_r2 = calculate_fit_metrics(Signal_observed, Signal_computed)

        # Export fit parameters and results to a text file
        if save_results:
            replica_file = os.path.join(results_save_dir, f"fit_results_replica_{replica_index}.txt")
            with open(replica_file, 'w') as f:
                f.write(f"Input:\nd0 (M): {d0_in_M:.6e}\n")
                f.write(f"Id lower bound: {Id_lower * 1e6:.3e} signal/M\n")
                f.write(f"Id upper bound: {Id_upper * 1e6:.3e} signal/M\n")
                f.write(f"I0 lower bound: {I0_lower:.3e}\n")
                f.write(f"I0 upper bound: {I0_upper:.3e}\n")

                f.write("\nOutput:\nMedian parameters:\n")
                f.write(f"I0: {median_params[0]:.2e}\n")
                f.write(f"Id: {median_params[2] * 1e6:.2e} signal/M\n")
                f.write(f"Ihd: {median_params[3] * 1e6:.2e} signal/M\n")
                f.write(f"RMSE: {median_rmse:.3f}\n")
                f.write(f"R²: {median_r2:.3f}\n")

                # Export only acceptable filtered fit parameters
                f.write("\nAcceptable Fit Parameters:\n")
                f.write("Kd (M^-1)\tI0\tId (signal/M)\tIhd (signal/M)\tRMSE\tR²\n")
                for params, fit_rmse, fit_r2 in filtered_results:
                    f.write(f"{params[1] * 1e6:.2e}\t{params[0]:.2e}\t{params[2] * 1e6:.2e}\t{params[3] * 1e6:.2e}\t{fit_rmse:.3f}\t{fit_r2:.3f}\n")

                # Calculate standard deviations for Kg, I0, Id, and Ihd if there are filtered results
                if filtered_results:
                    Kd_values = [params[1] * 1e6 for params, _, _ in filtered_results]
                    I0_values = [params[0] for params, _, _ in filtered_results]
                    Id_values = [params[2] * 1e6 for params, _, _ in filtered_results]
                    Ihd_values = [params[3] * 1e6 for params, _, _ in filtered_results]
            
                    Kd_std = np.std(Kd_values)
                    I0_std = np.std(I0_values)
                    Id_std = np.std(Id_values)
                    Ihd_std = np.std(Ihd_values)
                else:
                    Kd_std = I0_std = Id_std = Ihd_std = np.nan  # Assign NaN if no filtered results
                
                # Write the standard deviations
                f.write("\nStandard Deviations:\n")
                f.write(f"Kd Std Dev: {Kd_std:.2e} M^-1\n")
                f.write(f"I0 Std Dev: {I0_std:.2e}\n")
                f.write(f"Id Std Dev: {Id_std:.2e} signal/M\n")
                f.write(f"Ihd Std Dev: {Ihd_std:.2e} signal/M\n")

                # Write the input signal data
                f.write("\nOriginal Data:\nConcentration (M)\tSignal\n")
                for h0, signal in zip(h0_values / 1e6, Signal_observed):  # Convert h0_values to M
                    f.write(f"{h0:.6e}\t{signal:.6e}\n")

                # Write the fitting curve data
                f.write("\nFitting Curve:\n")
                f.write("Simulated Concentration (M)\tSimulated Signal\n")
                for x_sim, y_sim in zip(np.array(fitting_curve_x) / 1e6, fitting_curve_y):  # Convert fitting_curve_x to M
                    f.write(f"{x_sim:.6e}\t{y_sim:.6e}\n")
                f.write(f"\nDate of Export: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")   # Exports time stamp

            print(f"Results for Replica {replica_index} saved to {replica_file}")

    if display_plots:
        plt.show()

class DBAFittingAppHtoD:
    def __init__(self, root):
        self.root = root
        self.root.title("DBA Host-to-Dye Fitting Interface")
        self.info_label = None
        
        # Variables
        self.file_path_var = tk.StringVar()
        self.use_results_file_var = tk.BooleanVar()
        self.results_file_path_var = tk.StringVar()
        self.d0_var = tk.DoubleVar()
        self.fit_trials_var = tk.IntVar()
        self.rmse_threshold_var = tk.DoubleVar()
        self.r2_threshold_var = tk.DoubleVar()
        self.save_plots_var = tk.BooleanVar()
        self.results_dir_var = tk.StringVar()
        self.display_plots_var = tk.BooleanVar()
        self.save_results_var = tk.BooleanVar()
        self.results_save_dir_var = tk.StringVar()

        # Set default values
        self.file_path_var.set('/path/to/data/file.txt')
        self.results_dir_var.set('/path/to/results/')
        self.results_save_dir_var.set('/path/to/results/')
        self.d0_var.set(6e-6)
        self.fit_trials_var.set(200)
        self.rmse_threshold_var.set(2)
        self.r2_threshold_var.set(0.9)
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

        tk.Label(self.root, text="D₀ (M):").grid(row=3, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.d0_entry = tk.Entry(self.root, textvariable=self.d0_var, justify='left')
        self.d0_entry.grid(row=3, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="Number of Fit Trials:").grid(row=4, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.fit_trials_entry = tk.Entry(self.root, textvariable=self.fit_trials_var, justify='left')
        self.fit_trials_entry.grid(row=4, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="RMSE Threshold Factor:").grid(row=5, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.rmse_threshold_entry = tk.Entry(self.root, textvariable=self.rmse_threshold_var, justify='left')
        self.rmse_threshold_entry.grid(row=5, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="R² Threshold:").grid(row=6, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.r2_threshold_entry = tk.Entry(self.root, textvariable=self.r2_threshold_var, justify='left')
        self.r2_threshold_entry.grid(row=6, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Checkbutton(self.root, text="Save Plots To", variable=self.save_plots_var, command=self.update_save_plot_widgets).grid(row=7, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.results_dir_entry = tk.Entry(self.root, textvariable=self.results_dir_var, width=40, state=tk.DISABLED, justify='left')
        self.results_dir_entry.grid(row=7, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.results_dir_button = tk.Button(self.root, text="Browse", command=lambda: self.browse_directory(self.results_dir_entry), state=tk.DISABLED)
        self.results_dir_button.grid(row=7, column=2, padx=pad_x, pady=pad_y)
        self.save_plots_var.trace_add('write', lambda *args: self.update_save_plot_widgets())

        tk.Checkbutton(self.root, text="Save Results To", variable=self.save_results_var, command=self.update_save_results_widgets).grid(row=8, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.results_save_dir_entry = tk.Entry(self.root, textvariable=self.results_save_dir_var, width=40, state=tk.DISABLED, justify='left')
        self.results_save_dir_entry.grid(row=8, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.results_save_dir_button = tk.Button(self.root, text="Browse", command=lambda: self.browse_directory(self.results_save_dir_entry), state=tk.DISABLED)
        self.results_save_dir_button.grid(row=8, column=2, padx=pad_x, pady=pad_y)
        self.save_results_var.trace_add('write', lambda *args: self.update_save_results_widgets())

        tk.Checkbutton(self.root, text="Display Plots", variable=self.display_plots_var).grid(row=9, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(row=10, column=0, columnspan=3, pady=10, padx=pad_x)

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
        self.info_label.grid(row=11, column=0, columnspan=3, pady=10)

    def run_fitting(self):
        try:
            file_path = self.file_path_entry.get()
            results_dir = os.path.dirname(self.results_file_path_entry.get()) if self.use_results_file_var.get() else None
            d0_in_M = self.d0_var.get()
            rmse_threshold_factor = self.rmse_threshold_var.get()
            r2_threshold = self.r2_threshold_var.get()
            save_plots = self.save_plots_var.get()
            display_plots = self.display_plots_var.get()
            plots_dir = self.results_dir_entry.get()
            save_results = self.save_results_var.get()
            results_save_dir = self.results_save_dir_entry.get()
            number_of_fit_trials = self.fit_trials_var.get()  # Add this line
            
            # Show a progress indicator
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Fitting in Progress")
            progress_label = tk.Label(progress_window, text="Fitting in progress, please wait...")
            progress_label.pack(padx=20, pady=20)
            self.root.update_idletasks()
            
            run_dba_fitting(
                file_path=file_path,
                results_dir=results_dir,
                d0_in_M=d0_in_M,
                rmse_threshold_factor=rmse_threshold_factor,
                r2_threshold=r2_threshold,
                save_plots=save_plots,
                display_plots=display_plots,
                plots_dir=plots_dir,
                save_results=save_results,
                results_save_dir=results_save_dir,
                number_of_fit_trials=number_of_fit_trials  # Add this parameter
            )

            progress_window.destroy()
            self.show_message("Fitting complete!", is_error=False)

        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            self.show_message(f"Error: {str(e)}", is_error=True)

if __name__ == "__main__":
    root = tk.Tk()
    DBAFittingAppHtoD(root)
    root.mainloop()

