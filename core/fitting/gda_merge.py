"""
GDA merge fitting logic, refactored from the original GUI for modular use.
"""

import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from core.fitting_utils import load_replica_file
from utils.plot_utils import create_plots, format_value


def load_replica_data(file_path):
    """
    Load replica data using the centralized function.
    Maintains backward compatibility by returning data in expected format.
    """
    data = load_replica_file(file_path, "gda")

    # Convert to the format expected by existing code
    return {
        "g0": data.get("g0"),
        "h0": data.get("h0"),
        "Kd": data.get("Kd"),
        "concentrations": data["concentrations"],
        "signals": data["signals"],
        "median_params": {
            "Kg": data["median_params"]["Kg"],
            "I0": data["median_params"]["I0"],
            "Id": data["median_params"]["Id"],
            "Ihd": data["median_params"]["Ihd"],
        },
        "rmse": data["rmse"],
        "r_squared": data["r_squared"],
    }
    data["concentrations"] = np.array(data["concentrations"])
    data["signals"] = np.array(data["signals"])
    return data


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


def calculate_fit_metrics(observed, computed):
    rmse = np.sqrt(np.nanmean((observed - computed) ** 2))
    ss_res = np.nansum((observed - computed) ** 2)
    ss_tot = np.nansum((observed - np.nanmean(observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return rmse, r_squared


def detect_outliers_per_point(data, reference, relative_threshold):
    deviations = np.abs(data - reference)
    outlier_indices = np.where(deviations > relative_threshold * reference)[0]
    return outlier_indices


def export_averaged_data(
    avg_concentrations,
    avg_signals,
    avg_fitting_curve_x,
    avg_fitting_curve_y,
    avg_params,
    stdev_params,
    rmse,
    r_squared,
    results_dir,
    input_values,
    retained_replicas_info,
):
    averaged_data_file = os.path.join(results_dir, "averaged_fit_results.txt")
    with open(averaged_data_file, "w") as f:
        f.write("Input:\n")
        for key, value in input_values.items():
            f.write(f"{key}: {value}\n")
        f.write("\nRetained Replicas:\n")
        f.write("Replica\tKg (M^-1)\tI0\tId (signal/M)\tIhd (signal/M)\tRMSE\tR²\n")
        for replica_info in retained_replicas_info:
            original_index, params, fit_rmse, fit_r2 = replica_info
            f.write(
                f"{original_index}\t{params[1] * 1e6:.2e}\t{params[0]:.2e}\t{params[2] * 1e6:.2e}\t{params[3] * 1e6:.2e}\t{fit_rmse:.3f}\t{fit_r2:.3f}\n"
            )
        f.write("\nOutput:\nAveraged Parameters:\n")
        f.write(f"Kg: {avg_params[1]:.2e} M^-1 (STDEV: {stdev_params[1]:.2e})\n")
        f.write(f"I0: {avg_params[0]:.2e} (STDEV: {stdev_params[0]:.2e})\n")
        f.write(f"Id: {avg_params[2]:.2e} signal/M (STDEV: {stdev_params[2]:.2e})\n")
        f.write(f"Ihd: {avg_params[3]:.2e} signal/M (STDEV: {stdev_params[3]:.2e})\n")
        f.write(f"RMSE: {rmse:.3f}\nR²: {r_squared:.3f}\n")
        f.write("\nAveraged Data:\nConcentration (M)\tSignal\n")
        for conc, signal in zip(avg_concentrations, avg_signals):
            f.write(f"{conc:.6e}\t{signal:.6e}\n")
        f.write(
            "\nAveraged Fitting Curve:\nSimulated Concentration (M)\tSimulated Signal\n"
        )
        for x_fit, y_fit in zip(avg_fitting_curve_x, avg_fitting_curve_y):
            f.write(f"{x_fit:.6e}\t{y_fit:.6e}\n")
        f.write(f"\nDate of Export: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def run_gda_merge_fits(
    results_dir,
    outlier_relative_threshold,
    rmse_threshold_factor,
    kg_threshold_factor,
    save_plots,
    display_plots,
    save_results,
    results_save_dir,
    plot_title,
):
    replica_files = [
        f for f in os.listdir(results_dir) if f.startswith("fit_results_replica_")
    ]
    replicas = [load_replica_data(os.path.join(results_dir, f)) for f in replica_files]
    median_signals_per_point = np.median(
        [replica["signals"] for replica in replicas], axis=0
    )
    filtered_signals_per_point = [[] for _ in range(len(median_signals_per_point))]

    # Plot each replica's data and fit curves before filtering, marking outliers
    fig1, ax1 = create_plots(
        x_label=r"$D_0$ $\rm{[\mu M]}$",
        y_label=r"Signal $\rm{[AU]}$",
        plot_title=plot_title,
    )
    colors = plt.cm.jet(np.linspace(0, 1, len(replicas)))
    for idx, replica in enumerate(replicas):
        concentrations = np.array(replica["concentrations"])
        signals = np.array(replica["signals"])
        Kd = replica["Kd"]
        h0 = replica["h0"]
        g0 = replica["g0"]
        outlier_indices = detect_outliers_per_point(
            signals, median_signals_per_point, outlier_relative_threshold
        )
        for i, signal in enumerate(signals):
            if i not in outlier_indices:
                filtered_signals_per_point[i].append(signal)
        median_params = [
            replica["median_params"]["I0"],
            replica["median_params"]["Kg"],
            replica["median_params"]["Id"],
            replica["median_params"]["Ihd"],
        ]
        fitting_curve_x, fitting_curve_y = [], []
        for i in range(len(concentrations) - 1):
            extra_points = np.linspace(concentrations[i], concentrations[i + 1], 21)
            fitting_curve_x.extend(extra_points)
            fitting_curve_y.extend(
                compute_signal(median_params, extra_points, Kd, h0, g0)
            )
        last_signal = compute_signal(median_params, [concentrations[-1]], Kd, h0, g0)[0]
        fitting_curve_x.append(concentrations[-1])
        fitting_curve_y.append(last_signal)
        concentrations_plot = np.array(concentrations) * 1e6
        fitting_curve_x_plot = np.array(fitting_curve_x) * 1e6
        ax1.plot(
            concentrations_plot,
            signals,
            "o",
            color=colors[idx],
            label=f"Replica {idx + 1} Data",
        )
        ax1.plot(
            fitting_curve_x_plot,
            fitting_curve_y,
            "--",
            color=colors[idx],
            alpha=0.7,
            label=f"Replica {idx + 1} Fit",
        )
        if len(outlier_indices) > 0:
            ax1.plot(
                concentrations[outlier_indices],
                signals[outlier_indices],
                "x",
                color=colors[idx],
                markersize=8,
                label=f"Replica {idx + 1} Outliers",
            )

    ax1.legend(loc="best")
    fig1.tight_layout()
    if save_plots:
        plot_file = os.path.join(
            results_dir, "all_replicas_fitting_plot_with_outliers.png"
        )
        fig1.savefig(plot_file, bbox_inches="tight")

    valid_replicas = []
    for idx, replica in enumerate(replicas):
        median_params = [
            replica["median_params"]["I0"],
            replica["median_params"]["Kg"],
            replica["median_params"]["Id"],
            replica["median_params"]["Ihd"],
        ]
        computed_signals = compute_signal(
            median_params,
            replica["concentrations"],
            replica["Kd"],
            replica["h0"],
            replica["g0"],
        )
        rmse, r_squared = calculate_fit_metrics(replica["signals"], computed_signals)
        valid_replicas.append((idx + 1, replica, median_params, rmse, r_squared))
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
    avg_concentrations = np.mean(
        [r[1]["concentrations"] for r in retained_replicas], axis=0
    )
    avg_signals = np.array(
        [
            np.nanmean(filtered_signals_per_point[i])
            for i in range(len(filtered_signals_per_point))
        ]
    )
    avg_params = [
        np.nanmean([params[0] for _, _, params, _, _ in retained_replicas]),
        np.nanmean([params[1] for _, _, params, _, _ in retained_replicas]),
        np.nanmean([params[2] for _, _, params, _, _ in retained_replicas]),
        np.nanmean([params[3] for _, _, params, _, _ in retained_replicas]),
    ]
    stdev_params = [
        np.nanstd([params[0] for _, _, params, _, _ in retained_replicas]),
        np.nanstd([params[1] for _, _, params, _, _ in retained_replicas]),
        np.nanstd([params[2] for _, _, params, _, _ in retained_replicas]),
        np.nanstd([params[3] for _, _, params, _, _ in retained_replicas]),
    ]
    avg_fitting_curve_x, avg_fitting_curve_y = [], []
    for i in range(len(avg_concentrations) - 1):
        extra_points = np.linspace(avg_concentrations[i], avg_concentrations[i + 1], 21)
        avg_fitting_curve_x.extend(extra_points)
        avg_fitting_curve_y.extend(
            compute_signal(
                avg_params,
                extra_points,
                retained_replicas[0][1]["Kd"],
                retained_replicas[0][1]["h0"],
                retained_replicas[0][1]["g0"],
            )
        )
    computed_signals_at_avg_conc = compute_signal(
        avg_params,
        avg_concentrations,
        retained_replicas[0][1]["Kd"],
        retained_replicas[0][1]["h0"],
        retained_replicas[0][1]["g0"],
    )
    rmse, r_squared = calculate_fit_metrics(avg_signals, computed_signals_at_avg_conc)

    # Plot averaged data and fitting curve
    fig2, ax2 = create_plots(
        x_label=r"$D_0$ $\rm{[\mu M]}$",
        y_label=r"Signal $\rm{[AU]}$",
        plot_title=plot_title,
    )
    ax2.plot(
        np.array(avg_concentrations) * 1e6,
        avg_signals,
        "o",
        label="Averaged Data",
        color="black",
    )
    ax2.plot(
        np.array(avg_fitting_curve_x) * 1e6,
        avg_fitting_curve_y,
        "--",
        color="red",
        linewidth=1,
        label="Averaged Fit",
    )
    param_text = (
        f"$K_g$: {avg_params[1]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[1]:.2e})\n"
        f"$I_0$: {avg_params[0]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[0]:.2e})\n"
        f"$I_d$: {avg_params[2]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[2]:.2e})\n"
        f"$I_{{hd}}$: {avg_params[3]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[3]:.2e})\n"
        f"$RMSE$: {format_value(rmse)}\n"
        f"$R^2$: {r_squared:.3f}"
    )
    ax2.annotate(
        param_text,
        xy=(0.97, 0.04),
        xycoords="axes fraction",
        fontsize=10,
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="black",
            facecolor="lightgrey",
            alpha=0.5,
        ),
        multialignment="left",
    )
    ax2.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
    fig2.tight_layout()
    if save_plots:
        plot_file = os.path.join(results_dir, "averaged_fitting_plot.png")
        fig2.savefig(plot_file, bbox_inches="tight")
    if display_plots:
        plt.show()
    else:
        plt.close("all")

    if save_results:
        export_averaged_data(
            avg_concentrations,
            avg_signals,
            avg_fitting_curve_x,
            avg_fitting_curve_y,
            avg_params,
            stdev_params,
            rmse,
            r_squared,
            results_save_dir,
            {
                "g0 (M)": retained_replicas[0][1]["g0"],
                "h0 (M)": retained_replicas[0][1]["h0"],
                "Kd (M^-1)": retained_replicas[0][1]["Kd"],
            },
            [
                (original_index, params, rmse, r2)
                for original_index, _, params, rmse, r2 in retained_replicas
            ],
        )

    # Plot averaged data and fitting curve
    fig2, ax2 = create_plots(
        x_label=r"$D_0$ $\rm{[\mu M]}$",
        y_label=r"Signal $\rm{[AU]}$",
        plot_title=plot_title,
    )

    ax2.plot(
        np.array(avg_concentrations) * 1e6,
        avg_signals,
        "o",
        label="Averaged Data",
        color="black",
    )
    ax2.plot(
        np.array(avg_fitting_curve_x) * 1e6,
        avg_fitting_curve_y,
        "--",
        color="red",
        linewidth=1,
        label="Averaged Fit",
    )

    param_text = (
        f"$K_g$: {avg_params[1]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[1]:.2e})\n"
        f"$I_0$: {avg_params[0]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[0]:.2e})\n"
        f"$I_d$: {avg_params[2]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[2]:.2e})\n"
        f"$I_{{hd}}$: {avg_params[3]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[3]:.2e})\n"
        f"$RMSE$: {format_value(rmse)}\n"
        f"$R^2$: {r_squared:.3f}"
    )

    # Place annotation in a good location
    ax2.annotate(
        param_text,
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="black",
            facecolor="lightgrey",
            alpha=0.5,
        ),
        multialignment="left",
    )

    ax2.legend(loc="best")
    fig2.tight_layout()
    if save_plots:
        plot_file = os.path.join(results_dir, "averaged_fitting_plot.png")
        fig2.savefig(plot_file, bbox_inches="tight")
        print(f"Plot saved to {plot_file}")

    # Show all plots at once
    if display_plots:
        plt.show()
    else:
        plt.close("all")

    return {
        "avg_concentrations": avg_concentrations,
        "avg_signals": avg_signals,
        "avg_params": avg_params,
        "stdev_params": stdev_params,
        "rmse": rmse,
        "r_squared": r_squared,
        "retained_replicas": retained_replicas,
    }


class GDAMergeFitsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GDA Merge Fits Interface")
        self.info_label = None

        # Variables
        self.results_dir_var = tk.StringVar()
        self.outlier_threshold_var = tk.DoubleVar()
        self.rmse_threshold_factor_var = tk.DoubleVar()
        self.kg_threshold_factor_var = tk.DoubleVar()
        self.save_plots_var = tk.BooleanVar()
        self.save_plots_entry_var = tk.StringVar()
        self.display_plots_var = tk.BooleanVar()
        self.save_results_var = tk.BooleanVar()
        self.save_results_entry_var = tk.StringVar()
        self.plot_title_var = tk.StringVar()

        # Set default values
        self.outlier_threshold_var.set(0.25)
        self.rmse_threshold_factor_var.set(3)
        self.kg_threshold_factor_var.set(3)
        self.save_plots_var.set(False)
        self.display_plots_var.set(True)
        self.save_results_var.set(False)

        # Padding
        pad_x = 10
        pad_y = 5

        # Widgets
        tk.Label(self.root, text="Results Directory:").grid(
            row=0, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.results_dir_entry = tk.Entry(
            self.root, textvariable=self.results_dir_var, width=40, justify="left"
        )
        self.results_dir_entry.grid(
            row=0, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.results_dir_entry),
        ).grid(row=0, column=2, padx=pad_x, pady=pad_y)

        tk.Label(self.root, text="Plot Title:").grid(
            row=1, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.plot_title_entry = tk.Entry(
            self.root, textvariable=self.plot_title_var, width=40, justify="left"
        )
        self.plot_title_entry.grid(row=1, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)

        tk.Label(self.root, text="Outlier Relative Threshold:").grid(
            row=2, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.outlier_threshold_entry = tk.Entry(
            self.root, textvariable=self.outlier_threshold_var, justify="left"
        )
        self.outlier_threshold_entry.grid(
            row=2, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )

        tk.Label(self.root, text="RMSE Threshold Factor:").grid(
            row=3, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.rmse_threshold_factor_entry = tk.Entry(
            self.root, textvariable=self.rmse_threshold_factor_var, justify="left"
        )
        self.rmse_threshold_factor_entry.grid(
            row=3, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )

        tk.Label(self.root, text="Kg Threshold Factor:").grid(
            row=4, column=0, sticky=tk.W, padx=pad_x, pady=pad_y
        )
        self.kg_threshold_factor_entry = tk.Entry(
            self.root, textvariable=self.kg_threshold_factor_var, justify="left"
        )
        self.kg_threshold_factor_entry.grid(
            row=4, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )

        tk.Checkbutton(
            self.root,
            text="Save Plots To",
            variable=self.save_plots_var,
            command=self.update_save_plot_widgets,
        ).grid(row=5, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.plots_dir_entry = tk.Entry(
            self.root,
            textvariable=self.save_plots_entry_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.plots_dir_entry.grid(row=5, column=1, padx=pad_x, pady=pad_y, sticky=tk.W)
        self.plots_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.plots_dir_entry),
            state=tk.DISABLED,
        )
        self.plots_dir_button.grid(row=5, column=2, padx=pad_x, pady=pad_y)

        self.save_plots_var.trace_add(
            "write", lambda *args: self.update_save_plot_widgets()
        )

        tk.Checkbutton(
            self.root,
            text="Save Results To",
            variable=self.save_results_var,
            command=self.update_save_results_widgets,
        ).grid(row=6, column=0, columnspan=1, sticky=tk.W, padx=pad_x, pady=pad_y)
        self.save_results_dir_entry = tk.Entry(
            self.root,
            textvariable=self.save_results_entry_var,
            width=40,
            state=tk.DISABLED,
            justify="left",
        )
        self.save_results_dir_entry.grid(
            row=6, column=1, padx=pad_x, pady=pad_y, sticky=tk.W
        )
        self.save_results_dir_button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(self.save_results_dir_entry),
            state=tk.DISABLED,
        )
        self.save_results_dir_button.grid(row=6, column=2, padx=pad_x, pady=pad_y)

        self.save_results_var.trace_add(
            "write", lambda *args: self.update_save_results_widgets()
        )

        tk.Checkbutton(
            self.root, text="Display Plots", variable=self.display_plots_var
        ).grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=pad_x, pady=pad_y)
        tk.Button(self.root, text="Run Merge Fits", command=self.run_merge_fits).grid(
            row=8, column=0, columnspan=3, pady=10, padx=pad_x
        )

    def browse_directory(self, entry):
        initial_dir = (
            os.path.dirname(self.results_dir_var.get())
            if self.results_dir_var.get()
            else os.getcwd()
        )
        directory_path = filedialog.askdirectory(
            initialdir=initial_dir, title="Select Directory"
        )
        if directory_path:
            entry.delete(0, tk.END)
            entry.insert(0, directory_path)

    def update_save_plot_widgets(self):
        state = tk.NORMAL if self.save_plots_var.get() else tk.DISABLED
        self.plots_dir_entry.config(state=state)
        self.plots_dir_button.config(state=state)
        self.save_plots_entry_var.set(self.results_dir_var.get())

    def update_save_results_widgets(self):
        state = tk.NORMAL if self.save_results_var.get() else tk.DISABLED
        self.save_results_dir_entry.config(state=state)
        self.save_results_dir_button.config(state=state)
        self.save_results_entry_var.set(self.results_dir_var.get())

    def show_message(self, message, is_error=False):
        if self.info_label:
            self.info_label.destroy()
        fg_color = "red" if is_error else "green"
        self.info_label = tk.Label(self.root, text=message, fg=fg_color)
        self.info_label.grid(row=9, column=0, columnspan=3, pady=10)

    def run_merge_fits(self):
        # try:
        results_dir = self.results_dir_var.get()
        outlier_relative_threshold = self.outlier_threshold_var.get()
        rmse_threshold_factor = self.rmse_threshold_factor_var.get()
        kg_threshold_factor = self.kg_threshold_factor_var.get()
        save_plots = self.save_plots_var.get()
        display_plots = self.display_plots_var.get()
        save_results = self.save_results_var.get()
        results_save_dir = self.save_results_entry_var.get()
        plot_title = self.plot_title_var.get()

        # Show a progress indicator
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Merging Fits in Progress")
        progress_label = tk.Label(
            progress_window, text="Merging fits in progress, please wait..."
        )
        progress_label.pack(padx=20, pady=20)
        self.root.update()

        # Run the GDA merge fits process
        run_gda_merge_fits(
            results_dir,
            outlier_relative_threshold,
            rmse_threshold_factor,
            kg_threshold_factor,
            save_plots,
            display_plots,
            save_results,
            results_save_dir,
            plot_title,
        )

        # Close the progress window and show success message
        progress_window.destroy()
        self.show_message("Merging fits completed successfully.")

        # except Exception as e:
        #     # Close the progress window and show error message
        #     progress_window.destroy()
        #     self.show_message(f"Error: {e}", is_error=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = GDAMergeFitsApp(root)
    root.mainloop()
