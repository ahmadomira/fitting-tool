"""
Base merge fitting logic with unified workflow for all assay types.
This module provides a common framework for merging replica fits across different assays.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from core.fitting_utils import (
    calculate_fit_metrics,
    export_merge_results,
    load_replica_file,
)
from core.forward_model import (
    compute_signal_dba,
    compute_signal_gda,
    compute_signal_ida,
)
from utils.plot_utils import (
    create_plots,
    format_value,
    place_legend_and_annotation_safely,
    scientific_notation,
)
from utils.stats_utils import detect_outliers_per_point

# Assay configuration registry
k_unit = r"$M^{-1}$"
ASSAY_CONFIGS = {
    "dba_HtoD": {
        "name": "DBA Host-to-Dye",
        "x_label": r"$H_0$ $\rm{[\mu M]}$",
        "signal_func": compute_signal_dba,
        "k_param": "Kd",
        "k_label": r"$K_{a(D)}$",
        "k_unit": k_unit,
        "fixed_param": "d0",
        "variable_param": "h0",
    },
    "dba_DtoH": {
        "name": "DBA Dye-to-Host",
        "x_label": r"$D_0$ $\rm{[\mu M]}$",
        "signal_func": compute_signal_dba,
        "k_param": "Kd",
        "k_label": r"$K_{a(D)}$",
        "k_unit": k_unit,
        "fixed_param": "h0",
        "variable_param": "d0",
    },
    "gda": {
        "name": "GDA",
        "x_label": r"$D_0$ $\rm{[\mu M]}$",
        "signal_func": compute_signal_gda,
        "k_param": "Kg",
        "k_label": r"$K_{a(G)}$",
        "k_unit": k_unit,
        "fixed_param": "g0",
        "variable_param": "d0",
    },
    "ida": {
        "name": "IDA",
        "x_label": r"$G_0$ $\rm{[\mu M]}$",
        "signal_func": compute_signal_ida,
        "k_param": "Kg",
        "k_label": r"$K_{a(G)}$",
        "k_unit": k_unit,
        "fixed_param": "d0",
        "variable_param": "g0",
    },
}


def load_replica_data(file_path, assay_type):
    """
    Load replica data using the centralized function.

    Parameters:
    file_path (str): Path to replica file
    assay_type (str): Assay type identifier

    Returns:
    dict: Loaded replica data in consistent format
    """
    if assay_type not in ASSAY_CONFIGS:
        raise ValueError(f"Unknown assay type: {assay_type}")

    return load_replica_file(file_path, assay_type)


def compute_replica_signals(params, concentrations, assay_type, assay_params):
    """
    Compute signals for a replica using the appropriate signal function.

    Parameters:
    params (list): Fitting parameters [I0, K, Id, Ihd]
    concentrations (array): Concentration values
    assay_type (str): Assay type identifier
    assay_params (dict): Additional parameters needed for signal computation

    Returns:
    numpy.ndarray: Computed signals
    """
    config = ASSAY_CONFIGS[assay_type]
    signal_func = config["signal_func"]

    if assay_type in ["dba_HtoD", "dba_DtoH"]:
        # DBA assays: compute_signal_dba(params, x_titrations, y_fixed)
        if assay_type == "dba_HtoD":
            # Host-to-Dye: titrated = H0, fixed = D0
            return signal_func(params, concentrations, assay_params["d0"])
        else:  # dba_DtoH
            # Dye-to-Host: titrated = D0, fixed = H0
            return signal_func(params, concentrations, assay_params["h0"])

    elif assay_type == "gda":
        # GDA: compute_signal_gda(params, d0_values, Kd, h0, g0)
        return signal_func(
            params,
            concentrations,
            assay_params["Kd"],
            assay_params["h0"],
            assay_params["g0"],
        )

    elif assay_type == "ida":
        # IDA: compute_signal_ida(params, g0_values, Kd, h0, d0)
        return signal_func(
            params,
            concentrations,
            assay_params["Kd"],
            assay_params["h0"],
            assay_params["d0"],
        )

    else:
        raise ValueError(
            f"Signal computation not implemented for assay type: {assay_type}"
        )


def generate_fitting_curve(
    params, concentrations, assay_type, assay_params, n_points=21
):
    """
    Generate a smooth fitting curve for plotting.

    Parameters:
    params (list): Fitting parameters
    concentrations (array): Original concentration points
    assay_type (str): Assay type identifier
    assay_params (dict): Additional parameters for signal computation
    n_points (int): Number of interpolation points between each pair

    Returns:
    tuple: (x_curve, y_curve) arrays for the fitting curve
    """
    fitting_curve_x, fitting_curve_y = [], []

    # Interpolate between concentration points
    for i in range(len(concentrations) - 1):
        extra_points = np.linspace(concentrations[i], concentrations[i + 1], n_points)
        fitting_curve_x.extend(extra_points)
        fitting_curve_y.extend(
            compute_replica_signals(params, extra_points, assay_type, assay_params)
        )

    # Add the final point
    last_signal = compute_replica_signals(
        params, [concentrations[-1]], assay_type, assay_params
    )[0]
    fitting_curve_x.append(concentrations[-1])
    fitting_curve_y.append(last_signal)

    return np.array(fitting_curve_x), np.array(fitting_curve_y)


def plot_replicas_with_outliers(
    replicas, assay_type, plot_title, outlier_threshold, custom_x_label=None
):
    """
    Plot all replicas with their fits, marking outliers.

    Parameters:
    replicas (list): List of replica data dictionaries
    assay_type (str): Assay type identifier
    plot_title (str): Plot title
    outlier_threshold (float): Threshold for outlier detection

    Returns:
    tuple: (fig, ax) matplotlib objects
    """
    config = ASSAY_CONFIGS[assay_type]

    # Use custom x_label if provided, otherwise use default from config
    x_label = (
        custom_x_label + r" $\rm{[\mu M]}$" if custom_x_label else config["x_label"]
    )

    fig, ax = create_plots(
        x_label=x_label,
        y_label=r"Signal $\rm{[AU]}$",
        plot_title=plot_title,
    )

    # Calculate median signals for outlier detection
    median_signals_per_point = np.median([r["signals"] for r in replicas], axis=0)

    colors = plt.cm.jet(np.linspace(0, 1, len(replicas)))

    for idx, replica in enumerate(replicas):
        concentrations = replica["concentrations"]
        signals = replica["signals"]

        # Detect outliers
        outlier_indices = detect_outliers_per_point(
            signals, median_signals_per_point, outlier_threshold
        )

        # Get fitting parameters and compute curve
        median_params = [
            replica["median_params"]["I0"],
            replica["median_params"][config["k_param"]],
            replica["median_params"]["Id"],
            replica["median_params"]["Ihd"],
        ]

        # Prepare assay parameters for signal computation
        assay_params = {}
        if assay_type in ["dba_HtoD", "dba_DtoH"]:
            assay_params = {"d0": replica.get("d0"), "h0": replica.get("h0")}
        elif assay_type == "gda":
            assay_params = {
                "Kd": replica.get("Kd"),
                "h0": replica.get("h0"),
                "g0": replica.get("g0"),
            }
        elif assay_type == "ida":
            assay_params = {
                "Kd": replica.get("Kd"),
                "h0": replica.get("h0"),
                "d0": replica.get("d0"),
            }

        fitting_curve_x, fitting_curve_y = generate_fitting_curve(
            median_params, concentrations, assay_type, assay_params
        )

        # Convert to µM for plotting
        concentrations_plot = concentrations * 1e6
        fitting_curve_x_plot = fitting_curve_x * 1e6

        # Plot data points
        ax.plot(
            concentrations_plot,
            signals,
            "o",
            color=colors[idx],
            label=f"Replica {idx + 1} Data",
        )

        # Plot fitting curve
        ax.plot(
            fitting_curve_x_plot,
            fitting_curve_y,
            "--",
            color=colors[idx],
            alpha=0.7,
            label=f"Replica {idx + 1} Fit",
        )

        # Mark outliers
        if len(outlier_indices) > 0:
            ax.plot(
                concentrations_plot[outlier_indices],
                signals[outlier_indices],
                "x",
                color=colors[idx],
                markersize=8,
                label=f"Replica {idx + 1} Outliers",
            )

    ax.legend(loc="best")
    fig.tight_layout()

    return fig, ax


def plot_averaged_fit(
    avg_concentrations,
    avg_signals,
    avg_params,
    stdev_params,
    rmse,
    r_squared,
    assay_type,
    assay_params,
    plot_title,
    custom_x_label=None,
):
    """
    Plot averaged data with fitting curve and parameter annotations.

    Parameters:
    avg_concentrations (array): Averaged concentrations
    avg_signals (array): Averaged signals
    avg_params (list): Averaged parameters [I0, K, Id, Ihd]
    stdev_params (list): Parameter standard deviations
    rmse (float): Root mean square error
    r_squared (float): Coefficient of determination
    assay_type (str): Assay type identifier
    assay_params (dict): Parameters for signal computation
    plot_title (str): Plot title

    Returns:
    tuple: (fig, ax) matplotlib objects
    """
    config = ASSAY_CONFIGS[assay_type]

    # Use custom x_label if provided, otherwise use default from config
    x_label = (
        custom_x_label + r" $\rm{[\mu M]}$" if custom_x_label else config["x_label"]
    )

    fig, ax = create_plots(
        x_label=x_label,
        y_label=r"Signal $\rm{[AU]}$",
        plot_title=plot_title,
    )

    # Plot averaged data
    ax.plot(
        avg_concentrations * 1e6,
        avg_signals,
        "o",
        label="Averaged Data",
        color="black",
    )

    # Generate and plot fitting curve
    fitting_curve_x, fitting_curve_y = generate_fitting_curve(
        avg_params, avg_concentrations, assay_type, assay_params
    )

    ax.plot(
        fitting_curve_x * 1e6,
        fitting_curve_y,
        "--",
        color="red",
        linewidth=1,
        label="Averaged Fit",
    )

    # Create parameter text annotation
    param_text = (
        f"{config['k_label']}: ${scientific_notation(avg_params[1])}$ {config['k_unit']} (STDEV: ${scientific_notation(stdev_params[1])}$)\n"
        f"$I_0$: ${scientific_notation(avg_params[0])}$ (STDEV: ${scientific_notation(stdev_params[0])}$)\n"
        f"$I_d$: ${scientific_notation(avg_params[2])}$ {k_unit} (STDEV: ${scientific_notation(stdev_params[2])}$)\n"
        f"$I_{{hd}}$: ${scientific_notation(avg_params[3])}$ {k_unit} (STDEV: ${scientific_notation(stdev_params[3])}$)\n"
        f"$RMSE$: {format_value(rmse)}\n"
        f"$R^2$: {r_squared:.3f}"
    )

    legend, annotation = place_legend_and_annotation_safely(ax, param_text)

    fig.tight_layout()

    return fig, ax


def run_merge_fits(
    results_dir,
    assay_type,
    outlier_relative_threshold=0.1,
    rmse_threshold_factor=2.0,
    k_threshold_factor=2.0,
    save_plots=False,
    display_plots=False,
    save_results=False,
    results_save_dir=None,
    plot_title="",
    custom_x_label=None,
):
    """
    Run the complete merge fits workflow for any assay type.

    Parameters:
    results_dir (str): Directory containing replica files
    assay_type (str): Type of assay (dba_HtoD, dba_DtoH, ida, gda)
    outlier_relative_threshold (float): Threshold for outlier detection
    rmse_threshold_factor (float): Factor for RMSE-based replica filtering
    k_threshold_factor (float): Factor for K parameter-based replica filtering
    save_plots (bool): Whether to save plots
    display_plots (bool): Whether to display plots
    save_results (bool): Whether to save results to file
    results_save_dir (str): Directory to save results (if different from results_dir)
    plot_title (str): Title for plots

    Returns:
    dict: Dictionary containing merge results
    """
    if assay_type not in ASSAY_CONFIGS:
        raise ValueError(
            f"Unknown assay type: {assay_type}. Supported types: {list(ASSAY_CONFIGS.keys())}"
        )

    config = ASSAY_CONFIGS[assay_type]
    results_save_dir = results_save_dir or results_dir

    # Load replica files
    replica_files = [
        f for f in os.listdir(results_dir) if f.startswith("fit_results_replica_")
    ]

    if not replica_files:
        raise FileNotFoundError(f"No replica files found in {results_dir}")

    replicas = []
    for f in replica_files:
        try:
            replica = load_replica_data(os.path.join(results_dir, f), assay_type)
            replicas.append(replica)
        except Exception as e:
            print(f"Warning: Could not load replica file {f}: {e}")
            continue

    if not replicas:
        raise ValueError("No valid replica data could be loaded")

    # Calculate median signals for outlier detection
    median_signals_per_point = np.median([r["signals"] for r in replicas], axis=0)
    filtered_signals_per_point = [[] for _ in range(len(median_signals_per_point))]

    # Plot all replicas with outliers marked
    fig1, ax1 = plot_replicas_with_outliers(
        replicas, assay_type, plot_title, outlier_relative_threshold, custom_x_label
    )

    if save_plots:
        plot_file = os.path.join(
            results_dir, "all_replicas_fitting_plot_with_outliers.png"
        )
        fig1.savefig(plot_file, bbox_inches="tight")

    # Filter outliers and prepare data for averaging
    for replica in replicas:
        signals = replica["signals"]
        outlier_indices = detect_outliers_per_point(
            signals, median_signals_per_point, outlier_relative_threshold
        )

        for i, signal in enumerate(signals):
            if i not in outlier_indices:
                filtered_signals_per_point[i].append(signal)

    # Evaluate replica quality and filter
    valid_replicas = []
    for idx, replica in enumerate(replicas):
        median_params = [
            replica["median_params"]["I0"],
            replica["median_params"][config["k_param"]],
            replica["median_params"]["Id"],
            replica["median_params"]["Ihd"],
        ]

        # Prepare assay parameters for signal computation
        assay_params = {}
        if assay_type in ["dba_HtoD", "dba_DtoH"]:
            assay_params = {"d0": replica.get("d0"), "h0": replica.get("h0")}
        elif assay_type == "gda":
            assay_params = {
                "Kd": replica.get("Kd"),
                "h0": replica.get("h0"),
                "g0": replica.get("g0"),
            }
        elif assay_type == "ida":
            assay_params = {
                "Kd": replica.get("Kd"),
                "h0": replica.get("h0"),
                "d0": replica.get("d0"),
            }

        # Compute signals and evaluate fit quality
        computed_signals = compute_replica_signals(
            median_params, replica["concentrations"], assay_type, assay_params
        )
        rmse, r_squared = calculate_fit_metrics(replica["signals"], computed_signals)

        valid_replicas.append((idx + 1, replica, median_params, rmse, r_squared))

    if not valid_replicas:
        raise ValueError("No valid replicas after quality evaluation")

    # Apply statistical filtering
    mean_rmse = np.mean([v[3] for v in valid_replicas])
    std_rmse = np.std([v[3] for v in valid_replicas])
    mean_k = np.mean([v[2][1] for v in valid_replicas])  # K parameter is at index 1
    std_k = np.std([v[2][1] for v in valid_replicas])

    rmse_threshold = mean_rmse + rmse_threshold_factor * std_rmse
    k_threshold = mean_k + k_threshold_factor * std_k

    retained_replicas = [
        (original_index, replica, params, rmse, r2)
        for original_index, replica, params, rmse, r2 in valid_replicas
        if rmse <= rmse_threshold and abs(params[1] - mean_k) <= k_threshold
    ]

    if not retained_replicas:
        print(
            "Warning: No replicas passed statistical filtering. Using all valid replicas."
        )
        retained_replicas = valid_replicas

    print(f"\nRetained {len(retained_replicas)} out of {len(replicas)} replicas")
    for original_index, _, params, rmse, r2 in retained_replicas:
        print(
            f"Replica {original_index}: RMSE = {rmse:.2f}, R² = {r2:.3f}, {config['k_param']} = {params[1]:.2e}"
        )

    # Calculate averaged results
    avg_concentrations = np.mean(
        [r[1]["concentrations"] for r in retained_replicas], axis=0
    )
    avg_signals = np.array(
        [
            (
                np.nanmean(filtered_signals_per_point[i])
                if filtered_signals_per_point[i]
                else np.nan
            )
            for i in range(len(filtered_signals_per_point))
        ]
    )

    avg_params = [
        np.nanmean([params[i] for _, _, params, _, _ in retained_replicas])
        for i in range(4)
    ]
    stdev_params = [
        np.nanstd([params[i] for _, _, params, _, _ in retained_replicas])
        for i in range(4)
    ]

    # Prepare average assay parameters
    if assay_type in ["dba_HtoD", "dba_DtoH"]:
        avg_assay_params = {
            "d0": retained_replicas[0][1].get("d0"),
            "h0": retained_replicas[0][1].get("h0"),
        }
    elif assay_type == "gda":
        avg_assay_params = {
            "Kd": retained_replicas[0][1].get("Kd"),
            "h0": retained_replicas[0][1].get("h0"),
            "g0": retained_replicas[0][1].get("g0"),
        }
    elif assay_type == "ida":
        avg_assay_params = {
            "Kd": retained_replicas[0][1].get("Kd"),
            "h0": retained_replicas[0][1].get("h0"),
            "d0": retained_replicas[0][1].get("d0"),
        }

    # Calculate final fit quality metrics
    computed_signals_at_avg_conc = compute_replica_signals(
        avg_params, avg_concentrations, assay_type, avg_assay_params
    )
    rmse, r_squared = calculate_fit_metrics(avg_signals, computed_signals_at_avg_conc)

    # Generate averaged fitting curve for export
    avg_fitting_curve_x, avg_fitting_curve_y = generate_fitting_curve(
        avg_params, avg_concentrations, assay_type, avg_assay_params
    )

    # Plot averaged results
    fig2, ax2 = plot_averaged_fit(
        avg_concentrations,
        avg_signals,
        avg_params,
        stdev_params,
        rmse,
        r_squared,
        assay_type,
        avg_assay_params,
        plot_title,
        custom_x_label,
    )

    if save_plots:
        plot_file = os.path.join(results_dir, "averaged_fitting_plot.png")
        fig2.savefig(plot_file, bbox_inches="tight")

    # Handle plot display
    if display_plots:
        plt.show()
    else:
        plt.close("all")

    # Export results if requested
    if save_results:
        # Prepare input values dictionary for export
        input_values = {}
        for key, value in avg_assay_params.items():
            if value is not None:
                if key in ["d0", "h0", "g0"]:
                    input_values[f"{key} (M)"] = value
                elif key == "Kd":
                    input_values["Kd (M^-1)"] = value

        export_merge_results(
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
            assay_type,
        )

    return {
        "avg_concentrations": avg_concentrations,
        "avg_signals": avg_signals,
        "avg_params": avg_params,
        "stdev_params": stdev_params,
        "rmse": rmse,
        "r_squared": r_squared,
        "retained_replicas": retained_replicas,
        "assay_type": assay_type,
    }
