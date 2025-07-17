"""
DBA Merge Fits logic extracted from the GUI for reuse and testing.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from core.fitting_utils import load_replica_file
from utils.plot_utils import create_plots, format_value


def load_replica_data(file_path, assay_type="dba_HtoD"):
    """
    Load replica data using the centralized function.
    Maintains backward compatibility by returning data in expected format.
    """
    data = load_replica_file(file_path, assay_type)

    # Convert to the format expected by existing code
    return {
        "d0": data.get("d0"),
        "h0": data.get("h0"),  # For DBA_DtoH assay
        "concentrations": data["concentrations"],
        "signals": data["signals"],
        "median_params": {
            "Kd": data["median_params"]["Kd"],
            "I0": data["median_params"]["I0"],
            "Id": data["median_params"]["Id"],
            "Ihd": data["median_params"]["Ihd"],
        },
        "rmse": data["rmse"],
        "r_squared": data["r_squared"],
    }


def compute_signal(params, h0_values, d0, kd):
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


def save_plot(fig, filename, results_dir):
    plot_file = os.path.join(results_dir, filename)
    fig.savefig(plot_file, bbox_inches="tight")
    print(f"Plot saved to {plot_file}")


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
        f.write("Replica\tKd (M^-1)\tI0\tId (signal/M)\tIhd (signal/M)\tRMSE\tR²\n")
        for replica_info in retained_replicas_info:
            original_index, params, fit_rmse, fit_r2 = replica_info
            f.write(
                f"{original_index}\t{params[1] * 1e6:.2e}\t{params[0]:.2e}\t{params[2] * 1e6:.2e}\t{params[3] * 1e6:.2e}\t{fit_rmse:.3f}\t{fit_r2:.3f}\n"
            )
        f.write("\nOutput:\nAveraged Parameters:\n")
        f.write(f"Kd: {avg_params[1]:.2e} M^-1 (STDEV: {stdev_params[1]:.2e})\n")
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
    print(f"Averaged data and fitting results saved to {averaged_data_file}")


def run_dba_merge_fits(
    results_dir,
    outlier_relative_threshold,
    rmse_threshold_factor,
    kd_threshold_factor,
    save_plots,
    display_plots,
    save_results,
    results_save_dir,
    plot_title,
    assay_type="dba_HtoD",
):
    """
    Run DBA merge fits with configurable assay type.

    Parameters:
    assay_type (str): Either 'dba_HtoD' or 'dba_DtoH' to specify the assay direction
    """
    replica_files = [
        f for f in os.listdir(results_dir) if f.startswith("fit_results_replica_")
    ]
    replicas = [
        load_replica_data(os.path.join(results_dir, f), assay_type)
        for f in replica_files
    ]
    median_signals_per_point = np.median(
        [replica["signals"] for replica in replicas], axis=0
    )
    filtered_signals_per_point = [[] for _ in range(len(median_signals_per_point))]

    # Determine variable name for plotting
    var_label = (
        r"$H_0$ $\rm{[\mu M]}$" if assay_type == "dba_HtoD" else r"$D_0$ $\rm{[\mu M]}$"
    )

    fig1, ax1 = create_plots(
        x_label=var_label, y_label=r"Signal $\rm{[AU]}$", plot_title=plot_title
    )
    colors = plt.cm.jet(np.linspace(0, 1, len(replicas)))
    for idx, replica in enumerate(replicas):
        concentrations = np.array(replica["concentrations"])
        signals = np.array(replica["signals"])
        Kd = replica["median_params"]["Kd"]
        d0 = replica["d0"]
        h0 = replica.get("h0")  # For DtoH assay

        outlier_indices = detect_outliers_per_point(
            signals, median_signals_per_point, outlier_relative_threshold
        )
        for i, signal in enumerate(signals):
            if i not in outlier_indices:
                filtered_signals_per_point[i].append(signal)

        median_params = [
            replica["median_params"]["I0"],
            replica["median_params"]["Kd"],
            replica["median_params"]["Id"],
            replica["median_params"]["Ihd"],
        ]

        fitting_curve_x, fitting_curve_y = [], []
        for i in range(len(concentrations) - 1):
            extra_points = np.linspace(concentrations[i], concentrations[i + 1], 21)
            fitting_curve_x.extend(extra_points)
            if assay_type == "dba_HtoD":
                fitting_curve_y.extend(
                    compute_signal(median_params, extra_points, d0, Kd)
                )
            else:  # dba_DtoH
                fitting_curve_y.extend(
                    compute_signal(median_params, extra_points, d0, Kd)
                )

        if assay_type == "dba_HtoD":
            last_signal = compute_signal(median_params, [concentrations[-1]], d0, Kd)[0]
        else:  # dba_DtoH
            last_signal = compute_signal(median_params, [concentrations[-1]], d0, Kd)[0]

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
                concentrations_plot[outlier_indices],
                signals[outlier_indices],
                "x",
                color=colors[idx],
                markersize=8,
                label=f"Replica {idx + 1} Outliers",
            )

    ax1.legend(loc="best")
    fig1.tight_layout()
    if save_plots:
        save_plot(fig1, "all_replicas_fitting_plot_with_outliers.png", results_dir)

    valid_replicas = []
    for idx, replica in enumerate(replicas):
        median_params = [
            replica["median_params"]["I0"],
            replica["median_params"]["Kd"],
            replica["median_params"]["Id"],
            replica["median_params"]["Ihd"],
        ]

        if assay_type == "dba_HtoD":
            computed_signals = compute_signal(
                median_params,
                replica["concentrations"],
                replica["d0"],
                replica["median_params"]["Kd"],
            )
        else:  # dba_DtoH
            computed_signals = compute_signal(
                median_params,
                replica["concentrations"],
                replica["d0"],
                replica["median_params"]["Kd"],
            )

        rmse, r_squared = calculate_fit_metrics(replica["signals"], computed_signals)
        valid_replicas.append((idx + 1, replica, median_params, rmse, r_squared))
    mean_rmse = np.mean([v[3] for v in valid_replicas])
    std_rmse = np.std([v[3] for v in valid_replicas])
    mean_kd = np.mean([v[2][1] for v in valid_replicas])
    std_kd = np.std([v[2][1] for v in valid_replicas])
    rmse_threshold = mean_rmse + rmse_threshold_factor * std_rmse
    kd_threshold = mean_kd + kd_threshold_factor * std_kd
    retained_replicas = [
        (original_index, replica, params, rmse, r2)
        for original_index, replica, params, rmse, r2 in valid_replicas
        if rmse <= rmse_threshold and abs(params[1] - mean_kd) <= kd_threshold
    ]
    print("\nFinal Retained Replicas:")
    for original_index, _, params, rmse, r2 in retained_replicas:
        print(
            f"Replica {original_index}: RMSE = {rmse:.2f}, R² = {r2:.3f}, Kd = {params[1]:.2e}"
        )
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
        if assay_type == "dba_HtoD":
            avg_fitting_curve_y.extend(
                compute_signal(
                    avg_params,
                    extra_points,
                    retained_replicas[0][1]["d0"],
                    retained_replicas[0][1]["median_params"]["Kd"],
                )
            )
        else:  # dba_DtoH
            avg_fitting_curve_y.extend(
                compute_signal(
                    avg_params,
                    extra_points,
                    retained_replicas[0][1]["d0"],
                    retained_replicas[0][1]["median_params"]["Kd"],
                )
            )

    if assay_type == "dba_HtoD":
        computed_signals_at_avg_conc = compute_signal(
            avg_params,
            avg_concentrations,
            retained_replicas[0][1]["d0"],
            retained_replicas[0][1]["median_params"]["Kd"],
        )
    else:  # dba_DtoH
        computed_signals_at_avg_conc = compute_signal(
            avg_params,
            avg_concentrations,
            retained_replicas[0][1]["d0"],
            retained_replicas[0][1]["median_params"]["Kd"],
        )

    rmse, r_squared = calculate_fit_metrics(avg_signals, computed_signals_at_avg_conc)

    if save_results:
        input_values = {"d0 (M)": retained_replicas[0][1]["d0"]}
        if assay_type == "dba_DtoH" and retained_replicas[0][1].get("h0") is not None:
            input_values["h0 (M)"] = retained_replicas[0][1]["h0"]
        if retained_replicas[0][1]["median_params"]["Kd"] is not None:
            input_values["Kd (M^-1)"] = retained_replicas[0][1]["median_params"]["Kd"]

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
            input_values,
            [
                (original_index, params, rmse, r2)
                for original_index, _, params, rmse, r2 in retained_replicas
            ],
        )

    # Use appropriate variable label for final plot
    fig2, ax2 = create_plots(
        x_label=var_label, y_label=r"Signal $\rm{[AU]}$", plot_title=plot_title
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
        f"$K_d$: {avg_params[1]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[1]:.2e})\n"
        f"$I_0$: {avg_params[0]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[0]:.2e})\n"
        f"$I_d$: {avg_params[2]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[2]:.2e})\n"
        f"$I_{{hd}}$: {avg_params[3]:.2e} $M^{{-1}}$ (STDEV: {stdev_params[3]:.2e})\n"
        f"$RMSE$: {format_value(rmse)}\n"
        f"$R^2$: {r_squared:.3f}"
    )
    ax2.annotate(
        param_text,
        xy=(0.95, 0.05),
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
    ax2.legend()
    fig2.tight_layout()
    if save_plots:
        save_plot(fig2, "averaged_fitting_plot.png", results_dir)
    if display_plots:
        plt.show()
    else:
        plt.close()

    return {
        "avg_concentrations": avg_concentrations,
        "avg_signals": avg_signals,
        "avg_params": avg_params,
        "stdev_params": stdev_params,
        "rmse": rmse,
        "r_squared": r_squared,
        "retained_replicas": retained_replicas,
    }
