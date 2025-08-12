import os
from datetime import datetime

import numpy as np

# Unified assay mappings consistent with load_replica_file
assay_mappings = {
    "dba_HtoD": {
        "constant": "d0",
        "variable": "h0",
        "k_param": "Kd",
    },
    "dba_DtoH": {
        "constant": "h0",
        "variable": "d0",
        "k_param": "Kd",
    },
    "ida": {"constant": "d0", "variable": "g0", "k_param": "Kg"},
    "gda": {"constant": "g0", "variable": "d0", "k_param": "Kg"},
}

def to_uM(M):
    return M * 1e6

def to_M(uM):
    return uM * 1e-6

def load_data(file_path):
    try:
        with open(file_path, "r") as f:
            return f.readlines()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading file: {e}")
    except IOError as e:
        raise IOError(f"Error loading file: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


def split_replicas(data):
    replicas, current_replica = [], []
    # remove empty lines and strip whitespace
    data = [line.strip() for line in data if line.strip()]
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
                raise ValueError(
                    f"Invalid line in your input data. Line number: {data.index(line) + 1} (empty lines don't count). Line content: {line}. Please double-check your data.\n\n"
                )
    if current_replica:
        replicas.append(np.array(current_replica))
    if not replicas:
        raise ValueError("Replica splitting failed.")
    return replicas


def residuals(Signal_observed, compute_signal_func, *args):
    Signal_computed = compute_signal_func(*args)
    residual = Signal_observed - Signal_computed
    residual = np.nan_to_num(residual, nan=1e6)
    return residual


def calculate_fit_metrics(Signal_observed, Signal_computed):
    rmse = np.sqrt(np.nanmean((Signal_observed - Signal_computed) ** 2))
    ss_res = np.nansum((Signal_observed - Signal_computed) ** 2)
    ss_tot = np.nansum((Signal_observed - np.nanmean(Signal_observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return rmse, r_squared


def unique_filename(file):
    base, extension = os.path.splitext(file)
    counter = 1
    file = f"{base}{extension}"
    while os.path.exists(file):
        file = f"{base}_{counter}{extension}"
        counter += 1
    return file


def load_bounds_from_results_file(results_file_path):
    r"""Tries to load boundaries from previous fit results. If unsuccessful, sets default boundaries. Returns bounds in M⁻¹."""
    if results_file_path:
        try:
            with open(results_file_path, "r") as f:
                lines = f.readlines()
            id_prediction_line = next(
                (line for line in lines if "Id prediction interval" in line), None
            )
            if id_prediction_line and "not applicable" not in id_prediction_line:
                Id_lower = float(
                    id_prediction_line.split("[")[-1].split(",")[0].strip()
                )
                Id_upper = float(
                    id_prediction_line.split(",")[-1].split("]")[0].strip()
                )
            else:
                average_Id = float(
                    next(line for line in lines if "Average Id" in line)
                    .split("\t")[-1]
                    .split()[0]
                    .strip()
                )

                Id_lower = 0.5 * average_Id
                Id_upper = 2.0 * average_Id
            i0_prediction_line = next(
                (line for line in lines if "I0 prediction interval" in line), None
            )
            if i0_prediction_line and "not applicable" not in i0_prediction_line:
                I0_lower = float(
                    i0_prediction_line.split("[")[-1].split(",")[0].strip()
                )
                I0_upper = float(
                    i0_prediction_line.split(",")[-1].split("]")[0].strip()
                )
            else:
                average_I0 = float(
                    next(line for line in lines if "Average I0" in line)
                    .split("\t")[-1]
                    .split()[0]
                    .strip()
                )

                I0_lower = 0.5 * average_I0
                I0_upper = 2.0 * average_I0
        except Exception as e:
            raise ValueError(f"Error loading dye-alone results from file: {e}")
    else:
        Id_lower, Id_upper = 1e3, 1e18  # default Id bounds in M⁻¹
        I0_lower, I0_upper = 0, np.inf  # default I0 bounds in AU

    # TODO: ask frank about this Ihd
    Ihd_lower = 0.001 # in M⁻¹
    Ihd_upper = 1e12

    # return all in M⁻¹
    return Id_lower, Id_upper, I0_lower, I0_upper, Ihd_lower, Ihd_upper


def save_replica_file(
    results_save_dir,
    filtered_results, # a fit result = ((I0, Kd, Id, Ihd), func, rmse, r_squared)
    input_params, # input e.g. DBA (h0_uM, None, None, Id_lower, Id_upper, I0_lower, I0_upper), e.g. IDA (d0_uM, h0_uM, Kd_uM, ...)
    median_params, 
    fitting_params,
    assay,
):
    if assay not in assay_mappings:
        raise ValueError(f"Unknown assay type: {assay}")

    mapping = assay_mappings[assay]
    K_param = mapping["k_param"]

    constant_analyte_in_uM, h0_in_uM, Kd_uM, Id_lower_uM, Id_upper, I0_lower_uM, I0_upper = (
        input_params
    )
    I0, k_uM, I_d_uM, I_hd_uM, rmse, r_squared = median_params
    
    (
        variable_analyte_values_uM,
        Signal_observed,
        fitting_curve_x_uM,
        fitting_curve_y,
        replica_index,
    ) = fitting_params

    replica_filename = f"fit_results_replica_{replica_index}.txt"
    replica_file = os.path.join(results_save_dir, replica_filename)
    with open(replica_file, "w") as f:
        f.write("Input:\n")
        f.write(f"{mapping['constant']} (M): {to_M(constant_analyte_in_uM):.6e}\n")
        if h0_in_uM:
            f.write(f"h0 (M): {to_M(h0_in_uM):.6e}\n")
        if Kd_uM:
            f.write(f"Kd (M^-1): {Kd_uM / to_M(1.0):.6e}\n")
        f.write(f"Id lower bound (signal/M): {Id_lower_uM / to_M(1.0):.3e}\n")
        f.write(f"Id upper bound (signal/M): {Id_upper / to_M(1.0):.3e}\n")
        f.write(f"I0 lower bound: {I0_lower_uM:.3e}\n")
        f.write(f"I0 upper bound: {I0_upper:.3e}\n")

        f.write("\nOutput:\nMedian parameters:\n")
        f.write(f"{K_param} (M^-1): {k_uM / to_M(1.0):.2e}\n")
        f.write(f"I_0: {I0:.2e}\n")
        f.write(f"I_d (signal/M): {I_d_uM / to_M(1.0):.2e}\n")
        f.write(f"I_hd (signal/M): {I_hd_uM / to_M(1.0):.2e}\n")
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"R²: {r_squared:.3f}\n")

        f.write("\nAcceptable Fit Parameters:\n")
        f.write(f"{K_param} (M^-1)\tI0\tId (signal/M)\tIhd (signal/M)\tRMSE\tR²\n")

        for params, fit_rmse, fit_r2 in filtered_results:
            # params: I0, Kd, Id, Ihd
            f.write(
                f"{params[1] / to_M(1.0):.2e}\t{params[0]:.2e}\t{params[2] / to_M(1.0):.2e}\t{params[3] / to_M(1.0):.2e}\t{fit_rmse:.3f}\t{fit_r2:.3f}\n"
            )
        if filtered_results:
            Kg_values_M = [params[1] / to_M(1.0) for params, _, _ in filtered_results]
            I0_values = [params[0] for params, _, _ in filtered_results]
            Id_values_M = [params[2] / to_M(1.0) for params, _, _ in filtered_results]
            Ihd_values_M = [params[3] / to_M(1.0) for params, _, _ in filtered_results]
            Kg_std_M = np.std(Kg_values_M)
            I0_std = np.std(I0_values)
            Id_std_M = np.std(Id_values_M)
            Ihd_std_M = np.std(Ihd_values_M)
        else:
            Kg_std_M = I0_std = Id_std_M = Ihd_std_M = np.nan

        f.write("\nStandard Deviations:\n")
        f.write(f"{K_param} Std Dev (M^-1): {Kg_std_M:.2e}\n")
        f.write(f"I_0 Std Dev: {I0_std:.2e}\n")
        f.write(f"I_d Std Dev (signal/M): {Id_std_M:.2e}\n")
        f.write(f"I_hd Std Dev (signal/M): {Ihd_std_M:.2e}\n")

        f.write(f"\nOriginal Data:\nConcentration {mapping['variable']} (M)\tSignal\n")
        for titration_step, signal in zip(
            to_M(variable_analyte_values_uM), Signal_observed
        ):
            f.write(f"{titration_step:.6e}\t{signal:.6e}\n")

        f.write("\nFitting Curve:\n")
        f.write("Simulated Concentration (M)\tSimulated Signal\n")
        for x_sim, y_sim in zip(to_M(np.array(fitting_curve_x_uM)), fitting_curve_y):
            f.write(f"{x_sim:.6e}\t{y_sim:.6e}\n")
        f.write(f"\nDate of Export: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Results for Replica {replica_index} saved to {replica_file}")


def load_replica_file(file_path, assay):
    """
    Load replica data from a results file saved by save_replica_file.

    Parameters:
    file_path (str): Path to the replica file
    assay (str): One of 'dba_HtoD', 'dba_DtoH', 'ida', 'gda'

    Returns:
    dict: Dictionary containing all replica data including input parameters,
          median parameters, concentrations, signals, and fit metrics
    """
    import re

    if assay not in assay_mappings:
        raise ValueError(f"Unknown assay type: {assay}")

    mapping = assay_mappings[assay]

    # Initialize data structure
    data = {
        # Input parameters
        mapping["constant"]: None,
        "h0": None,
        "Kd": None,
        "Id_lower": None,
        "Id_upper": None,
        "I0_lower": None,
        "I0_upper": None,
        # Original data
        "concentrations": [],
        "signals": [],
        # Median parameters
        "median_params": {
            mapping["k_param"]: None,
            "I0": None,
            "Id": None,
            "Ihd": None,
        },
        # Fit metrics
        "rmse": None,
        "r_squared": None,
        # Acceptable fit parameters (for debugging/analysis)
        "acceptable_params": [],
        "std_devs": {
            mapping["k_param"]: None,
            "I0": None,
            "Id": None,
            "Ihd": None,
        },
        # Fitting curve data
        "fitting_curve_x": [],
        "fitting_curve_y": [],
        # Metadata
        "assay_type": assay,
        "export_date": None,
    }

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Replica file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading replica file: {e}")

    # State tracking for parsing
    current_section = None

    for line in lines:
        line = line.strip()

        # Section detection
        if "Input:" in line:
            current_section = "input"
            continue
        elif "Output:" in line:
            current_section = "output"
            continue
        elif "Median parameters:" in line:
            current_section = "median_params"
            continue
        elif "Acceptable Fit Parameters:" in line:
            current_section = "acceptable_params"
            continue
        elif "Standard Deviations:" in line:
            current_section = "std_devs"
            continue
        elif "Original Data:" in line:
            current_section = "original_data"
            continue
        elif "Fitting Curve:" in line:
            current_section = "fitting_curve"
            continue
        elif "Date of Export:" in line:
            try:
                date_str = line.split("Date of Export:")[1].strip()
                data["export_date"] = date_str
            except:
                pass
            continue

        # Skip empty lines and headers
        if not line or "Concentration" in line or "Simulated" in line:
            continue

        # Parse based on current section
        if current_section == "input":
            if f"{mapping['constant']} (M):" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data[mapping["constant"]] = float(
                        re.sub(r"[^\d.eE+-]", "", value_str)
                    )
                except:
                    pass
            elif "h0 (M):" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["h0"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass
            elif "Kd (M^-1):" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["Kd"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass
            elif "Id lower bound" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["Id_lower"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass
            elif "Id upper bound" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["Id_upper"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass
            elif "I0 lower bound" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["I0_lower"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass
            elif "I0 upper bound" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["I0_upper"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass

        elif current_section == "median_params":
            if (
                f"{mapping['k_param']} (M^-1):" in line
                or f"K_g (M^-1):" in line
                or f"K_d (M^-1):" in line
            ):
                try:
                    value_str = line.split(":")[1].strip()
                    data["median_params"][mapping["k_param"]] = float(
                        re.sub(r"[^\d.eE+-]", "", value_str)
                    )
                except:
                    pass
            elif "I_0:" in line or "I0:" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["median_params"]["I0"] = float(
                        re.sub(r"[^\d.eE+-]", "", value_str)
                    )
                except:
                    pass
            elif "I_d" in line or "Id" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["median_params"]["Id"] = float(
                        re.sub(r"[^\d.eE+-]", "", value_str)
                    )
                except:
                    pass
            elif "I_hd" in line or "Ihd" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["median_params"]["Ihd"] = float(
                        re.sub(r"[^\d.eE+-]", "", value_str)
                    )
                except:
                    pass
            elif "RMSE" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["rmse"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass
            elif "R²" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["r_squared"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass

        elif current_section == "acceptable_params":
            # Skip header line with parameter names
            if mapping["k_param"] in line and "I0" in line:
                continue
            try:
                parts = line.split("\t")
                if len(parts) >= 6:
                    params = [
                        float(parts[1]),
                        float(parts[0]),
                        float(parts[2]),
                        float(parts[3]),
                    ]  # [I0, K, Id, Ihd]
                    rmse = float(parts[4])
                    r2 = float(parts[5])
                    data["acceptable_params"].append((params, rmse, r2))
            except:
                pass

        elif current_section == "std_devs":
            if f"{mapping['k_param']} Std Dev" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["std_devs"][mapping["k_param"]] = float(
                        re.sub(r"[^\d.eE+-]", "", value_str)
                    )
                except:
                    pass
            elif "I_0 Std Dev" in line or "I0 Std Dev" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["std_devs"]["I0"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass
            elif "I_d Std Dev" in line or "Id Std Dev" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["std_devs"]["Id"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass
            elif "I_hd Std Dev" in line or "Ihd Std Dev" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["std_devs"]["Ihd"] = float(
                        re.sub(r"[^\d.eE+-]", "", value_str)
                    )
                except:
                    pass

        elif current_section == "original_data":
            try:
                parts = line.split("\t")
                if len(parts) >= 2:
                    conc = float(parts[0])
                    signal = float(parts[1])
                    data["concentrations"].append(conc)
                    data["signals"].append(signal)
            except:
                pass

        elif current_section == "fitting_curve":
            try:
                parts = line.split("\t")
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    data["fitting_curve_x"].append(x)
                    data["fitting_curve_y"].append(y)
            except:
                pass

    # Convert lists to numpy arrays
    data["concentrations"] = np.array(data["concentrations"])
    data["signals"] = np.array(data["signals"])
    data["fitting_curve_x"] = np.array(data["fitting_curve_x"])
    data["fitting_curve_y"] = np.array(data["fitting_curve_y"])

    return data


def export_merge_results(
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
    assay_type="dba_HtoD",
):
    """
    Export averaged merge results to a text file.

    Parameters:
    avg_concentrations: Averaged concentration data
    avg_signals: Averaged signal data
    avg_fitting_curve_x, avg_fitting_curve_y: Fitted curve data
    avg_params: Averaged fitting parameters [I0, K, Id, Ihd]
    stdev_params: Standard deviations of parameters
    rmse: Root mean square error
    r_squared: Coefficient of determination
    results_dir: Directory to save results
    input_values: Dictionary of input parameters
    retained_replicas_info: Information about retained replicas
    assay_type: Type of assay ('dba_HtoD', 'dba_DtoH', 'gda', 'ida')
    """
    # Determine parameter names based on assay type
    param_names = {
        "dba_HtoD": {"K": "Kd", "K_unit": "M^-1"},
        "dba_DtoH": {"K": "Kd", "K_unit": "M^-1"},
        "gda": {"K": "Kg", "K_unit": "M^-1"},
        "ida": {"K": "Kg", "K_unit": "M^-1"},
    }.get(assay_type, {"K": "K", "K_unit": "M^-1"})

    averaged_data_file = os.path.join(results_dir, "averaged_fit_results.txt")
    with open(averaged_data_file, "w") as f:
        f.write("Input:\n")
        for key, value in input_values.items():
            f.write(f"{key}: {value}\n")

        f.write("\nRetained Replicas:\n")
        f.write(
            f"Replica\t{param_names['K']} ({param_names['K_unit']})\tI0\tId (signal/M)\tIhd (signal/M)\tRMSE\tR²\n"
        )
        for replica_info in retained_replicas_info:
            original_index, params, fit_rmse, fit_r2 = replica_info
            f.write(
                f"{original_index}\t{params[1] * 1e6:.2e}\t{params[0]:.2e}\t{params[2] * 1e6:.2e}\t{params[3] * 1e6:.2e}\t{fit_rmse:.3f}\t{fit_r2:.3f}\n"
            )

        f.write("\nOutput:\nAveraged Parameters:\n")
        f.write(
            f"{param_names['K']}: {avg_params[1]:.2e} {param_names['K_unit']} (STDEV: {stdev_params[1]:.2e})\n"
        )
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



