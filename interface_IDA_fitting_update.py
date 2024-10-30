import tkinter as tk
from tkinter import filedialog, messagebox
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pltstyle import create_plots

import numpy as np
from scipy.optimize import brentq, minimize
import os
import matplotlib.pyplot as plt
from datetime import datetime
    
def run_ida_fitting(file_path, results_file_path, Kd_in_M, h0_in_M, g0_in_M, number_of_fit_trials, rmse_threshold_factor, r2_threshold, save_plots, display_plots, plots_dir, save_results, results_save_dir):
    import numpy as np
    from scipy.optimize import brentq, minimize
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime
    from pltstyle import create_plots

    # Initialize parameter ranges for optimization
    I0_range = (0, None)
    Id_range = (None, None)
    Ihd_range = (None, None)

    # Load bounds from results file if available
    if results_file_path and os.path.exists(results_file_path):
        try:
            with open(results_file_path, 'r') as f:
                lines = f.readlines()
            id_prediction_line = next((line for line in lines if 'Id prediction interval' in line), None)
            if id_prediction_line and 'not applicable' not in id_prediction_line:
                Id_lower = float(id_prediction_line.split('[')[-1].split(',')[0].strip())
                Id_upper = float(id_prediction_line.split(',')[-1].split(']')[0].strip())
            else:
                average_Id = float(next(line for line in lines if 'Average Id' in line).split('\t')[-1].strip())
                Id_lower = 0.5 * average_Id
                Id_upper = 2.0 * average_Id
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

    Id_lower /= 1e6
    Id_upper /= 1e6
    Ihd_lower = Ihd_range[0] / 1e6 if Ihd_range[0] is not None else 0.001
    Ihd_upper = Ihd_range[1] / 1e6 if Ihd_range[1] is not None else 1e12

    Kd = Kd_in_M / 1e6
    h0 = h0_in_M * 1e6
    d0 = g0_in_M * 1e6

    def compute_signal(params, g0_values, Kd, h0, d0):
        I0, Kg, Id, Ihd = params
        Signal_values = []
        for g0 in g0_values:
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

    def residuals(params, g0_values, Signal_observed, Kd, h0, d0):
        Signal_computed = compute_signal(params, g0_values, Kd, h0, d0)
        residual = Signal_observed - Signal_computed
        return np.nan_to_num(residual, nan=1e6)

    def calculate_fit_metrics(Signal_observed, Signal_computed):
        rmse = np.sqrt(np.nanmean((Signal_observed - Signal_computed) ** 2))
        ss_res = np.nansum((Signal_observed - Signal_computed) ** 2)
        ss_tot = np.nansum((Signal_observed - np.nanmean(Signal_observed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        return rmse, r_squared

    def load_data(file_path):
        try:
            with open(file_path, 'r') as f:
                return f.readlines()
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

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

    data_lines = load_data(file_path)
    if data_lines is None:
        raise ValueError("Data loading failed.")
    replicas = split_replicas(data_lines)
    if replicas is None:
        raise ValueError("Replica splitting failed.")

    figures = []  # List to store figures

    for replica_index, replica_data in enumerate(replicas, start=1):
        g0_values = replica_data[:, 0] * 1e6
        Signal_observed = replica_data[:, 1]

        if len(g0_values) < 2:
            print(f"Replica {replica_index} has insufficient data. Skipping.")
            continue

        I0_upper = np.min(Signal_observed) if I0_upper is None or np.isinf(I0_upper) else I0_upper

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

        best_result, best_cost = None, np.inf
        fit_results = []
        for initial_params in initial_params_list:
            result = minimize(lambda params: np.sum(residuals(params, g0_values, Signal_observed, Kd, h0, d0) ** 2),
                              initial_params, method='L-BFGS-B',
                              bounds=[(I0_lower, I0_upper), (1e-8, 1e8), (Id_lower, Id_upper), (Ihd_lower, Ihd_upper)])

            Signal_computed = compute_signal(result.x, g0_values, Kd, h0, d0)
            rmse, r_squared = calculate_fit_metrics(Signal_observed, Signal_computed)
            fit_results.append((result.x, result.fun, rmse, r_squared))

            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result

        best_rmse = min(fit_rmse for _, _, fit_rmse, _ in fit_results)
        rmse_threshold = best_rmse * rmse_threshold_factor

        filtered_results = [
            (params, fit_rmse, fit_r2) for params, _, fit_rmse, fit_r2 in fit_results
            if fit_rmse <= rmse_threshold and fit_r2 >= r2_threshold
        ]

        if filtered_results:
            median_params = np.median(np.array([result[0] for result in filtered_results]), axis=0)
        else:
            print("Warning: No fits meet the filtering criteria.")
            continue

        Signal_computed = compute_signal(median_params, g0_values, Kd, h0, d0)
        rmse, r_squared = calculate_fit_metrics(Signal_observed, Signal_computed)

        fitting_curve_x, fitting_curve_y = [], []
        for i in range(len(g0_values) - 1):
            extra_points = np.linspace(g0_values[i], g0_values[i + 1], 21)
            fitting_curve_x.extend(extra_points)
            fitting_curve_y.extend(compute_signal(median_params, extra_points, Kd, h0, d0))

        fig, ax = create_plots(x_label=r'$g_0$ $\rm{[\mu M]}$', y_label=r'Signal $\rm{[AU]}$')

        ax.plot(g0_values, Signal_observed, 'o', label='Observed Signal')
        ax.plot(fitting_curve_x, fitting_curve_y, '--', color='blue', alpha=0.6, label='Simulated Fitting Curve')
        ax.set_title(f'Observed vs. Simulated Fitting Curve for Replica {replica_index}')
        ax.legend()

        param_text = (f"$K_g$: {median_params[1] * 1e6:.2e} $M^{{-1}}$\n"
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

        figures.append(fig)  # Store the figure

        if save_results:
            replica_file = os.path.join(results_save_dir, f"fit_results_replica_{replica_index}.txt")
            with open(replica_file, 'w') as f:
                f.write("Input:\n")
                f.write(f"d0 (M): {g0_in_M:.6e}\n")
                f.write(f"h0 (M): {h0_in_M:.6e}\n")
                f.write(f"Kd (M^-1): {Kd_in_M:.6e}\n")
                f.write(f"Id lower bound (signal/M): {Id_lower * 1e6:.3e}\n")
                f.write(f"Id upper bound (signal/M): {Id_upper * 1e6:.3e}\n")
                f.write(f"I0 lower bound: {I0_lower:.3e}\n")
                f.write(f"I0 upper bound: {I0_upper:.3e}\n")

                f.write("\nOutput:\nMedian parameters:\n")
                f.write(f"Kg (M^-1): {median_params[1] * 1e6:.2e}\n")
                f.write(f"I0: {median_params[0]:.2e}\n")
                f.write(f"Id (signal/M): {median_params[2] * 1e6:.2e}\n")
                f.write(f"Ihd (signal/M): {median_params[3] * 1e6:.2e}\n")
                f.write(f"RMSE: {rmse:.3f}\n")
                f.write(f"R²: {r_squared:.3f}\n")

                f.write("\nAcceptable Fit Parameters:\n")
                f.write("Kg (M^-1)\tI0\tId (signal/M)\tIhd (signal/M)\tRMSE\tR²\n")
                for params, fit_rmse, fit_r2 in filtered_results:
                    f.write(f"{params[1] * 1e6:.2e}\t{params[0]:.2e}\t{params[2] * 1e6:.2e}\t{params[3] * 1e6:.2e}\t{fit_rmse:.3f}\t{fit_r2:.3f}\n")

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
                    Kg_std = I0_std = Id_std = Ihd_std = np.nan

                f.write("\nStandard Deviations:\n")
                f.write(f"Kg Std Dev (M^-1): {Kg_std:.2e}\n")
                f.write(f"I0 Std Dev: {I0_std:.2e}\n")
                f.write(f"Id Std Dev (signal/M): {Id_std:.2e}\n")
                f.write(f"Ihd Std Dev (signal/M): {Ihd_std:.2e}\n")

                f.write("\nOriginal Data:\nConcentration g0 (M)\tSignal\n")
                for g0, signal in zip(g0_values / 1e6, Signal_observed):
                    f.write(f"{g0:.6e}\t{signal:.6e}\n")

                f.write("\nFitting Curve:\n")
                f.write("Simulated Concentration (M)\tSimulated Signal\n")
                for x_sim, y_sim in zip(np.array(fitting_curve_x) / 1e6, fitting_curve_y):
                    f.write(f"{x_sim:.6e}\t{y_sim:.6e}\n")

                f.write(f"\nDate of Export: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                print(f"Results for Replica {replica_index} saved to {replica_file}")

    if display_plots:
        plt.show()

class IDAFittingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IDA Fitting Interface")
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
        self.results_dir_var = tk.StringVar()
        self.display_plots_var = tk.BooleanVar()
        self.save_results_var = tk.BooleanVar()
        self.results_save_dir_var = tk.StringVar()

        # Set default values
        self.file_path_var.set('/Users/ahmadomira/Downloads/interface_test/IDA_system.txt')
        self.results_dir_var.set('/path/to/results/')
        self.results_save_dir_var.set('/path/to/results/')
        self.Kd_var.set(1.68e7)
        self.h0_var.set(4.3e-6)
        self.g0_var.set(6e-6)
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
        self.results_dir_entry = tk.Entry(self.root, textvariable=self.results_dir_var, width=40, state=tk.DISABLED, justify='left')
        self.results_dir_entry.grid(row=9, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.results_dir_button = tk.Button(self.root, text="Browse", command=lambda: self.browse_directory(self.results_dir_entry), state=tk.DISABLED)
        self.results_dir_button.grid(row=9, column=2, padx=pad_x, pady=pad_y)

        self.save_plots_var.trace_add('write', lambda *args: self.update_save_plot_widgets())

        tk.Checkbutton(self.root, text="Save Results To", variable=self.save_results_var, command=self.update_save_results_widgets).grid(row=10, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.results_save_dir_entry = tk.Entry(self.root, textvariable=self.results_save_dir_var, width=40, state=tk.DISABLED, justify='left')
        self.results_save_dir_entry.grid(row=10, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.results_save_dir_button = tk.Button(self.root, text="Browse", command=lambda: self.browse_directory(self.results_save_dir_entry), state=tk.DISABLED)
        self.results_save_dir_button.grid(row=10, column=2, padx=pad_x, pady=pad_y)

        self.save_results_var.trace_add('write', lambda *args: self.update_save_results_widgets())

        tk.Checkbutton(self.root, text="Display Plots", variable=self.display_plots_var).grid(row=11, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
        tk.Button(self.root, text="Run Fitting", command=self.run_fitting).grid(row=12, column=0, columnspan=3, pady=10, padx=pad_x)

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
        try:
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
            plots_dir = self.results_dir_entry.get()
            save_results = self.save_results_var.get()
            results_save_dir = self.results_save_dir_entry.get()
            
            # Show a progress indicator
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Fitting in Progress")
            progress_label = tk.Label(progress_window, text="Fitting in progress, please wait...")
            progress_label.pack(padx=20, pady=20)
            self.root.update_idletasks()
            
            run_ida_fitting(file_path, results_file_path, Kd_in_M, h0_in_M, g0_in_M, number_of_fit_trials, rmse_threshold_factor, r2_threshold, save_plots, display_plots, plots_dir, save_results, results_save_dir)
            
            progress_window.destroy()
            
            self.show_message(f"Fitting completed!", is_error=False)
            
        except Exception as e:
            self.show_message(f"Error: {str(e)}", is_error=True)

if __name__ == "__main__":
    root = tk.Tk()
    
    IDAFittingApp(root)
    root.mainloop()