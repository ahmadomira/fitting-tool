"""
Full plate fitting logic that processes BMG Excel files directly.
Direct path from Excel files to fitted results by averaging replicas first, then fitting.
Unit policy (Option 2): external inputs/outputs in physical units (M, M^-1); internal
computations in µM / µM^-1 after a single conversion on input, with single reconversion
for reporting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from core.fitting.base_merge import ASSAY_CONFIGS
from core.fitting_utils import (
    calculate_fit_metrics,
    load_bounds_from_results_file,
    to_M,
    to_uM,
)
from utils.bmg_to_txt import read_bmg_xlsx
from utils.plot_utils import (
    create_plots,
    format_value,
    place_legend_and_annotation_safely,
    scientific_notation,
)


def run_full_plate_fit(
    excel_file_path,
    concentration_vector,  # external titrated species in M
    assay_type,  # 'ida', 'dba_HtoD', 'dba_DtoH', 'gda'
    assay_params,  # fixed params in physical units (M or M^-1 for Kd)
    number_of_fit_trials=50,
    parameter_bounds=None,  # optional physical bounds dict
    results_file_path=None,  # optional dye-alone results file for Id/I0/Ihd bounds
    save_plots=True,
    display_plots=False,
    save_results=True,
    results_save_dir=None,
    plots_dir=None,
    custom_x_label=None,
    custom_plot_title=None,
):
    """Run full plate fitting directly from BMG Excel file.

    Parameters
    ----------
    concentration_vector : list[float]
        Titrated concentrations (physical, M) for the variable species per assay.
    assay_params : dict
        Fixed parameters (physical units). Examples:
          DBA_HtoD: {'d0': M}; DBA_DtoH: {'h0': M};
          IDA: {'Kd': M^-1, 'h0': M, 'd0': M}; GDA: {'Kd': M^-1, 'h0': M, 'g0': M}.
    parameter_bounds : dict | None
        Physical bounds {'I0':(lo,hi), 'K':(lo,hi), 'Id':(lo,hi), 'Ihd':(lo,hi)} (K/Id/Ihd in M^-1, I0 in AU).
    results_file_path : str | None
        Path to dye-alone results file; if provided, Id/I0/Ihd bounds are derived & merged.
    Returns
    -------
    dict with results, including best_params (internal units) and fitting curve (M).
    """
    if assay_type not in ASSAY_CONFIGS:
        raise ValueError(
            f"Unknown assay type: {assay_type}. Supported: {list(ASSAY_CONFIGS.keys())}"
        )
    config = ASSAY_CONFIGS[assay_type]

    excel_file = Path(excel_file_path)
    if results_save_dir is None:
        results_save_dir = excel_file.parent / "full_plate_results"
    if plots_dir is None:
        plots_dir = results_save_dir / "plots"
    Path(results_save_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    print(f"Processing {excel_file.name} for {config['name']} analysis...")
    try:
        data, _ = read_bmg_xlsx(excel_file_path)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    if len(concentration_vector) != data.shape[1]:
        raise ValueError(
            f"Concentration vector length ({len(concentration_vector)}) does not match data columns ({data.shape[1]})"
        )

    concentrations_M = np.array(concentration_vector, dtype=float)
    concentrations_uM = to_uM(concentrations_M)

    avg_signals = data.mean(axis=0).values
    std_signals = data.std(axis=0).values
    n_replicas = data.shape[0]

    print(
        f"Averaged {n_replicas} replicas | Range: {concentrations_M.min():.2e}-{concentrations_M.max():.2e} M"
    )

    parameter_bounds_internal, bounds_physical = _prepare_bounds(
        parameter_bounds, results_file_path, avg_signals
    )
    assay_params_internal = _validate_and_convert_assay_params(assay_params, assay_type)

    fit_results = _run_optimization(
        concentrations_uM,
        avg_signals,
        assay_type,
        assay_params_internal,
        parameter_bounds_internal,
        number_of_fit_trials,
    )
    if not fit_results:
        raise ValueError("No valid fits found. Check data/parameters.")
    
    best_params_internal = fit_results["best_params"]  # [I0, K_int, Id_int, Ihd_int]
    fitting_curve_x_uM, fitting_curve_y = _generate_fitting_curve(
        best_params_internal, concentrations_uM, assay_type, assay_params_internal
    )
    fitting_curve_x_M = to_M(fitting_curve_x_uM)

    computed_signals = _compute_signals(
        best_params_internal, concentrations_uM, assay_type, assay_params_internal
    )
    rmse, r_squared = calculate_fit_metrics(avg_signals, computed_signals)

    if save_plots or display_plots:
        plot_title = custom_plot_title or f"{config['name']} - {excel_file.stem}"
        fig = _create_fit_plot(
            concentrations_uM,
            avg_signals,
            std_signals,
            fitting_curve_x_uM,
            fitting_curve_y,
            best_params_internal,
            rmse,
            r_squared,
            config,
            plot_title,
            custom_x_label,
        )
        if save_plots:
            plot_file = plots_dir / f"{excel_file.stem}_full_plate_fit.png"
            fig.savefig(plot_file, bbox_inches="tight", dpi=300)
            print(f"Plot saved: {plot_file}")
        if display_plots:
            plt.show()
        else:
            plt.close(fig)

    if save_results:
        results_file = results_save_dir / f"{excel_file.stem}_full_plate_results.txt"
        _save_results(
            results_file,
            concentrations_M,
            avg_signals,
            std_signals,
            fitting_curve_x_M,
            fitting_curve_y,
            best_params_internal,
            rmse,
            r_squared,
            assay_params_internal,
            fit_results["all_results"],
            assay_type,
            n_replicas,
        )
        print(f"Results saved: {results_file}")

    return {
        "concentrations_M": concentrations_M,
        "avg_signals": avg_signals,
        "std_signals": std_signals,
        "best_params_internal": best_params_internal,
        "rmse": rmse,
        "r_squared": r_squared,
        "fitting_curve_x": fitting_curve_x_M,
        "fitting_curve_y": fitting_curve_y,
        "n_replicas": n_replicas,
        "assay_type": assay_type,
        "excel_file": excel_file_path,
        "bounds_physical": bounds_physical,
    }


def _prepare_bounds(user_bounds, results_file_path, signals):
    """Create internal (µM / µM^-1) bounds and retain physical bounds.

    Precedence: dye-alone file (if provided) -> user overrides -> defaults.
    Physical defaults: broad ranges; K/Id/Ihd M^-1 → internal divide by 1e6.
    """
    Id_lo, Id_hi, I0_lo, I0_hi, Ihd_lo, Ihd_hi = load_bounds_from_results_file(
        results_file_path
    )
    physical_defaults = {
    "I0": (I0_lo, I0_hi),
    "K": (1e-8, 1e8),
    "Id": (Id_lo, Id_hi),
    "Ihd": (Ihd_lo, Ihd_hi),
    }
    
    if user_bounds:
        for k, v in user_bounds.items():
            if v is not None:
                physical_defaults[k] = v
    
    # infer I0 from signal range if I0 is inf
    if np.isinf(physical_defaults["I0"][1]):
        physical_defaults.update({"I0": (0, signals.min())})

    scale = to_uM(1.0)
    internal_bounds = {
        "I0": physical_defaults["I0"],
        "K": (physical_defaults["K"][0] / scale, physical_defaults["K"][1] / scale),
        "Id": (physical_defaults["Id"][0] / scale, physical_defaults["Id"][1] / scale),
        "Ihd": (
            physical_defaults["Ihd"][0] / scale,
            physical_defaults["Ihd"][1] / scale,
        ),
    }
    return internal_bounds, physical_defaults


def _validate_and_convert_assay_params(assay_params, assay_type):
    """Validate & convert fixed assay parameters to internal units (µM / µM^-1)."""
    internal = {}
    if assay_type == "dba_HtoD":
        required = ["d0"]
    elif assay_type == "dba_DtoH":
        required = ["h0"]
    elif assay_type == "ida":
        required = ["Kd", "h0", "d0"]
    elif assay_type == "gda":
        required = ["Kd", "h0", "g0"]
    else:
        raise ValueError(f"Unknown assay type: {assay_type}")
    for p in required:
        if p not in assay_params:
            raise ValueError(f"Missing required parameter for {assay_type}: {p}")
    scale = to_uM(1.0)
    if "d0" in assay_params:
        internal["d0"] = to_uM(assay_params["d0"])
    if "h0" in assay_params:
        internal["h0"] = to_uM(assay_params["h0"])
    if "g0" in assay_params:
        internal["g0"] = to_uM(assay_params["g0"])
    if "Kd" in assay_params:
        internal["Kd"] = assay_params["Kd"] / scale  # M^-1 → µM^-1
    return internal


def _run_optimization(
    concentrations_uM,
    signals,
    assay_type,
    assay_params_internal,
    bounds_internal,
    n_trials,
):
    best_result = None
    best_cost = np.inf
    all_results = []
    for trial in range(n_trials):
        initial_params = _generate_initial_params(bounds_internal, signals)
        scipy_bounds = [
            bounds_internal["I0"],
            bounds_internal["K"],
            bounds_internal["Id"],
            bounds_internal["Ihd"],
        ]
        try:
            result = minimize(
                lambda params: np.sum(
                    (
                        signals
                        - _compute_signals(
                            params, concentrations_uM, assay_type, assay_params_internal
                        )
                    )
                    ** 2
                ),
                initial_params,
                method="L-BFGS-B",
                bounds=scipy_bounds,
            )
            if result.success and result.fun < best_cost:
                best_cost = result.fun
                best_result = result
            computed_signals = _compute_signals(
                result.x, concentrations_uM, assay_type, assay_params_internal
            )
            rmse, r_squared = calculate_fit_metrics(signals, computed_signals)
            all_results.append(
                {
                    "params": result.x,
                    "cost": result.fun,
                    "rmse": rmse,
                    "r_squared": r_squared,
                    "success": result.success,
                }
            )
        except Exception as e:
            print(f"Trial {trial+1} failed: {e}")
            continue
    if best_result is None:
        return None
    return {
        "best_params": best_result.x,
        "best_cost": best_cost,
        "all_results": all_results,
        "n_successful": sum(1 for r in all_results if r["success"]),
    }


def _generate_initial_params(bounds, signals):
    I0 = np.random.uniform(bounds["I0"][0], bounds["I0"][1])
    K = 10 ** np.random.uniform(np.log10(bounds["K"][0]), np.log10(bounds["K"][1]))
    if signals[0] < signals[-1]:
        Id = 10 ** np.random.uniform(
            np.log10(bounds["Id"][0]), np.log10(bounds["Id"][1])
        )
        Ihd = Id * np.random.uniform(0.1, 2.0)
    else:
        Ihd = 10 ** np.random.uniform(
            np.log10(bounds["Ihd"][0]), np.log10(bounds["Ihd"][1])
        )
        Id = Ihd * np.random.uniform(0.1, 2.0)
    return [I0, K, Id, Ihd]


def _compute_signals(params, concentrations_uM, assay_type, assay_params_internal):
    signal_func = ASSAY_CONFIGS[assay_type]["signal_func"]
    if assay_type == "dba_HtoD":
        return signal_func(params, concentrations_uM, assay_params_internal["d0"])
    if assay_type == "dba_DtoH":
        return signal_func(params, concentrations_uM, assay_params_internal["h0"])
    if assay_type == "gda":
        return signal_func(
            params,
            concentrations_uM,
            assay_params_internal["Kd"],
            assay_params_internal.get("h0"),
            assay_params_internal.get("g0"),
        )
    if assay_type == "ida":
        return signal_func(
            params,
            concentrations_uM,
            assay_params_internal["Kd"],
            assay_params_internal.get("h0"),
            assay_params_internal.get("d0"),
        )
    raise ValueError(f"Signal computation not implemented for assay type: {assay_type}")


def _generate_fitting_curve(
    params, concentrations_uM, assay_type, assay_params_internal, n_points=21
):
    fitting_curve_x, fitting_curve_y = [], []
    for i in range(len(concentrations_uM) - 1):
        extra = np.linspace(concentrations_uM[i], concentrations_uM[i + 1], n_points)
        fitting_curve_x.extend(extra)
        fitting_curve_y.extend(
            _compute_signals(params, extra, assay_type, assay_params_internal)
        )
    last_signal = _compute_signals(
        params, [concentrations_uM[-1]], assay_type, assay_params_internal
    )[0]
    fitting_curve_x.append(concentrations_uM[-1])
    fitting_curve_y.append(last_signal)
    return np.array(fitting_curve_x), np.array(fitting_curve_y)


def _create_fit_plot(
    concentrations_uM,
    signals,
    signal_std,
    fitting_curve_x_uM,
    fitting_curve_y,
    params_internal,
    rmse,
    r_squared,
    config,
    plot_title,
    custom_x_label=None,
):
    x_label = (
        custom_x_label + r" $\rm{[\mu M]}$" if custom_x_label else config["x_label"]
    )
    fig, ax = create_plots(
        x_label=x_label, y_label=r"Signal $\rm{[AU]}$", plot_title=plot_title
    )
    ax.plot(
        fitting_curve_x_uM,
        fitting_curve_y,
        "--",
        color="darkgray",
        linewidth=2,
        label="Fit",
    )
    ax.errorbar(
        concentrations_uM,
        signals,
        yerr=signal_std,
        fmt="o",
        label="Avg ± STD",
        color="red",
        markersize=6,
        capsize=5,
        capthick=1,
        ecolor="black",
        elinewidth=1.5,
    )

    scale = to_M(1.0)
    param_text = (
        f"{config['k_label']}: ${scientific_notation(params_internal[1] / scale)}$ {config['k_unit']}\n"
        f"$I_0$: ${scientific_notation(params_internal[0])}$\n"
        f"$I_d$: ${scientific_notation(params_internal[2] / scale)}$ {config['k_unit']}\n"
        f"$I_{'{hd}'}$: ${scientific_notation(params_internal[3] / scale)}$ {config['k_unit']}\n"
        f"$RMSE$: {format_value(rmse)}\n"
        f"$R^2$: {r_squared:.3f}"
    )
    place_legend_and_annotation_safely(ax, param_text)
    fig.tight_layout()
    return fig


def _save_results(
    results_file,
    concentrations_M,
    signals,
    signal_std,
    fitting_curve_x_M,
    fitting_curve_y,
    params_internal,
    rmse,
    r_squared,
    assay_params_internal,
    all_results,
    assay_type,
    n_replicas,
):
    config = ASSAY_CONFIGS[assay_type]
    scale = to_uM(1.0)
    with open(results_file, "w") as f:
        f.write(f"Full Plate Fitting Results - {config['name']}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of replicas averaged: {n_replicas}\n")
        f.write(f"Assay type: {assay_type}\n")
        f.write(f"Number of concentration points: {len(concentrations_M)}\n\n")
        f.write("Assay Parameters:\n")
        for param, value in assay_params_internal.items():
            if param == "Kd":
                f.write(f"{param}: {value * scale:.3e} M^-1\n")
            else:
                f.write(f"{param}: {to_M(value):.3e} M\n")
        f.write("\n")
        f.write("Fitted Parameters (physical units):\n")
        f.write(f"I0: {params_internal[0]:.6e}\n")
        f.write(f"{config['k_param']}: {params_internal[1] * scale:.6e} M^-1\n")
        f.write(f"Id: {params_internal[2] * scale:.6e} M^-1\n")
        f.write(f"Ihd: {params_internal[3] * scale:.6e} M^-1\n\n")
        f.write("Fit Quality:\n")
        f.write(f"RMSE: {rmse:.6f} \n")
        f.write(f"R²: {r_squared:.6f}\n\n")
        successful_fits = [r for r in all_results if r["success"]]
        f.write("Optimization Summary:\n")
        f.write(f"Total trials: {len(all_results)}\n")
        f.write(f"Successful trials: {len(successful_fits)}\n")
        if successful_fits:
            rmse_vals = [r["rmse"] for r in successful_fits]
            r2_vals = [r["r_squared"] for r in successful_fits]
            f.write(f"RMSE range: {min(rmse_vals):.6f} - {max(rmse_vals):.6f}\n")
            f.write(f"R² range: {min(r2_vals):.6f} - {max(r2_vals):.6f}\n")
        f.write("\n")
        f.write("Experimental Data:\n")
        f.write("Concentration_M\tSignal_Observed\tSignal_StdDev\tSignal_Fitted\n")
        fitted_signals = _compute_signals(
            params_internal, to_uM(concentrations_M), assay_type, assay_params_internal
        )
        for conc, obs, std, fit in zip(
            concentrations_M, signals, signal_std, fitted_signals
        ):
            f.write(f"{conc:.6e}\t{obs:.6f}\t{std:.6f}\t{fit:.6f}\n")
        f.write("\nFitting Curve Data:\n")
        f.write("Concentration_M\tSignal_Fitted\n")
        for conc, sig in zip(fitting_curve_x_M, fitting_curve_y):
            f.write(f"{conc:.6e}\t{sig:.6f}\n")


if __name__ == "__main__":
    # Placeholder example (Excel reading expects .xlsx)
    pass
