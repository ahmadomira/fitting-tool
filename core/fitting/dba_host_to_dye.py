"""
DBA Host-to-Dye fitting logic extracted from the GUI for reuse and testing.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from core.fitting_utils import (
    calculate_fit_metrics,
    load_bounds_from_results_file,
    load_data,
    residuals,
    save_replica_file,
    split_replicas,
)
from core.forward_model import compute_signal_dba
from utils.plot_utils import plot_fitting_results, save_plot


def run_dba_host_to_dye_fitting(
    file_path,
    results_file_path,
    d0_in_M,
    rmse_threshold_factor,
    r2_threshold,
    save_plots,
    display_plots,
    plots_dir,
    save_results_bool,
    results_save_dir,
    number_of_fit_trials,
    assay="dba_HtoD",
    custom_x_label=None,
):
    Id_lower, Id_upper, I0_lower, I0_upper, _, _ = load_bounds_from_results_file(
        results_file_path
    )

    # convert to µM
    d0 = d0_in_M * 1e6

    print(
        f"Loaded boundaries:\nId: [{Id_lower * 1e6:.3e}, {Id_upper * 1e6:.3e}] M⁻¹\nI0: [{I0_lower:.3e}, {I0_upper:.3e}]"
    )
    data_lines = load_data(file_path)
    replicas = split_replicas(data_lines)
    print(f"Number of replicas detected: {len(replicas)}")
    figures = []
    plt.close("all")  # Close any previous plots

    for replica_index, replica_data in enumerate(replicas, start=1):
        print(f"Processing replica {replica_index}, data length: {len(replica_data)}")
        h0_values = replica_data[:, 0] * 1e6
        Signal_observed = replica_data[:, 1]
        if len(h0_values) < 2:
            print(f"Replica {replica_index} has insufficient data. Skipping.")
            continue
        I0_upper = (
            np.min(Signal_observed)
            if I0_upper is None or np.isinf(I0_upper)
            else I0_upper
        )
        Ihd_guess_smaller = Signal_observed[0] < Signal_observed[-1]
        initial_params_list = []
        for _ in range(number_of_fit_trials):
            I0_guess = np.random.uniform(I0_lower, I0_upper)
            Kd_guess = 10 ** np.random.uniform(-5, 5)
            if Ihd_guess_smaller:
                Id_guess = 10 ** np.random.uniform(
                    np.log10(Id_lower), np.log10(Id_upper)
                )
                Ihd_guess = Id_guess * np.random.uniform(0.1, 0.5)
            else:
                Ihd_guess = 10 ** np.random.uniform(
                    np.log10(Id_lower), np.log10(Id_upper)
                )
                Id_guess = Ihd_guess * np.random.uniform(0.1, 0.5)
            initial_params_list.append([I0_guess, Kd_guess, Id_guess, Ihd_guess])
        best_result, best_cost = None, np.inf
        fit_results = []
        for initial_params in initial_params_list:
            result = minimize(
                lambda params: np.sum(
                    residuals(
                        Signal_observed, compute_signal_dba, params, h0_values, d0
                    )
                    ** 2
                ),
                initial_params,
                method="L-BFGS-B",
                bounds=[
                    (I0_lower, I0_upper),
                    (1e-8, 1e8),
                    (Id_lower, Id_upper),
                    (1e-8, 1e8),
                ],
            )
            Signal_computed = compute_signal_dba(result.x, h0_values, d0)
            rmse, r_squared = calculate_fit_metrics(Signal_observed, Signal_computed)
            fit_results.append((result.x, result.fun, rmse, r_squared))
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result
        best_rmse = min(fit_rmse for _, _, fit_rmse, _ in fit_results)
        rmse_threshold = best_rmse * rmse_threshold_factor
        filtered_results = [
            (params, fit_rmse, fit_r2)
            for params, _, fit_rmse, fit_r2 in fit_results
            if fit_rmse <= rmse_threshold and fit_r2 >= r2_threshold
        ]
        if filtered_results:
            median_params = np.median(
                np.array([result[0] for result in filtered_results]), axis=0
            )
        else:
            print("Warning: No fits meet the filtering criteria.")
            continue
        Signal_computed = compute_signal_dba(median_params, h0_values, d0)
        rmse, r_squared = calculate_fit_metrics(Signal_observed, Signal_computed)
        fitting_curve_x, fitting_curve_y = [], []
        for i in range(len(h0_values) - 1):
            extra_points = np.linspace(h0_values[i], h0_values[i + 1], 21)
            fitting_curve_x.extend(extra_points)
            fitting_curve_y.extend(compute_signal_dba(median_params, extra_points, d0))
        last_signal = compute_signal_dba(median_params, [h0_values[-1]], d0)[0]
        fitting_curve_x.append(h0_values[-1])
        fitting_curve_y.append(last_signal)
        input_params = (d0_in_M, None, None, Id_lower, Id_upper, I0_lower, I0_upper)
        median_params = (*median_params, rmse, r_squared)
        fitting_params = (
            h0_values,
            Signal_observed,
            fitting_curve_x,
            fitting_curve_y,
            replica_index,
        )
        fig = plot_fitting_results(fitting_params, median_params, assay, custom_x_label)
        figures.append(fig)
        if save_results_bool:
            save_replica_file(
                results_save_dir,
                filtered_results,
                input_params,
                median_params,
                fitting_params,
                assay,
            )
    if save_plots:
        for fig in figures:
            save_plot(fig, plots_dir)
    if display_plots:
        plt.show()
