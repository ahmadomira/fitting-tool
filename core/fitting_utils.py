import os
from datetime import datetime

import numpy as np


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
    r"""Tries to load boundaries from previous fit results. If unsuccessful, sets default boundaries. Returns bounds in µM⁻¹."""
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
                    .strip()
                )
                I0_lower = 0.5 * average_I0
                I0_upper = 2.0 * average_I0
        except Exception as e:
            raise ValueError(f"Error loading dye-alone results from file: {e}")
    else:
        Id_lower, Id_upper = 1e3, 1e18
        I0_lower, I0_upper = 0, np.inf

    # Convert bounds to µM⁻¹ for fitting
    Id_lower /= 1e6
    Id_upper /= 1e6

    # TODO: ask frank about this Ihd
    Ihd_lower = 0.001
    Ihd_upper = 1e12

    return Id_lower, Id_upper, I0_lower, I0_upper, Ihd_lower, Ihd_upper


def save_replica_file(
    results_save_dir,
    filtered_results,
    input_params,
    median_params,
    fitting_params,
    assay,
):
    analytes = {
        "dba_HtoD": {"constant": "d0", "variable": "h0", "K_d": "K_d"},
        "dba_DtoH": {"constant": "h0", "variable": "d0", "K_d": "K_d"},
        "ida": {"constant": "d0", "variable": "g0"},
        "gda": {"constant": "g0", "variable": "d0"},
    }[assay]

    K_text = analytes.get("K_d", "K_g")

    constant_analyte_in_M, h0_in_M, Kd_in_M, Id_lower, Id_upper, I0_lower, I0_upper = (
        input_params
    )
    I0, k, I_d, I_hd, rmse, r_squared = median_params
    (
        variable_analyte_values,
        Signal_observed,
        fitting_curve_x,
        fitting_curve_y,
        replica_index,
    ) = fitting_params

    replica_filename = f"fit_results_replica_{replica_index}.txt"
    replica_file = os.path.join(results_save_dir, replica_filename)
    with open(replica_file, "w") as f:
        f.write("Input:\n")
        f.write(f"{analytes['constant']} (M): {constant_analyte_in_M:.6e}\n")
        if h0_in_M:
            f.write(f"h0 (M): {h0_in_M:.6e}\n")
        if Kd_in_M:
            f.write(f"Kd (M^-1): {Kd_in_M:.6e}\n")
        f.write(f"Id lower bound (signal/M): {Id_lower * 1e6:.3e}\n")
        f.write(f"Id upper bound (signal/M): {Id_upper * 1e6:.3e}\n")
        f.write(f"I0 lower bound: {I0_lower:.3e}\n")
        f.write(f"I0 upper bound: {I0_upper:.3e}\n")

        f.write("\nOutput:\nMedian parameters:\n")
        f.write(f"{K_text} (M^-1): {k * 1e6:.2e}\n")
        f.write(f"I_0: {I0:.2e}\n")
        f.write(f"I_d (signal/M): {I_d * 1e6:.2e}\n")
        f.write(f"I_hd (signal/M): {I_hd * 1e6:.2e}\n")
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"R²: {r_squared:.3f}\n")

        f.write("\nAcceptable Fit Parameters:\n")
        f.write(f"{K_text} (M^-1)\tI0\tId (signal/M)\tIhd (signal/M)\tRMSE\tR²\n")
        for params, fit_rmse, fit_r2 in filtered_results:
            f.write(
                f"{params[1] * 1e6:.2e}\t{params[0]:.2e}\t{params[2] * 1e6:.2e}\t{params[3] * 1e6:.2e}\t{fit_rmse:.3f}\t{fit_r2:.3f}\n"
            )
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
        f.write(f"{K_text} Std Dev (M^-1): {Kg_std:.2e}\n")
        f.write(f"I_0 Std Dev: {I0_std:.2e}\n")
        f.write(f"I_d Std Dev (signal/M): {Id_std:.2e}\n")
        f.write(f"I_hd Std Dev (signal/M): {Ihd_std:.2e}\n")

        f.write(f"\nOriginal Data:\nConcentration {analytes['variable']} (M)\tSignal\n")
        for titration_step, signal in zip(
            variable_analyte_values / 1e6, Signal_observed
        ):
            f.write(f"{titration_step:.6e}\t{signal:.6e}\n")

        f.write("\nFitting Curve:\n")
        f.write("Simulated Concentration (M)\tSimulated Signal\n")
        for x_sim, y_sim in zip(np.array(fitting_curve_x) / 1e6, fitting_curve_y):
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

    # Define assay-specific parameter mappings
    assay_mappings = {
        "dba_HtoD": {"constant": "d0", "variable": "h0", "k_param": "Kd"},
        "dba_DtoH": {"constant": "h0", "variable": "d0", "k_param": "Kd"},
        "ida": {"constant": "d0", "variable": "g0", "k_param": "Kg"},
        "gda": {"constant": "g0", "variable": "d0", "k_param": "Kg"},
    }

    if assay not in assay_mappings:
        raise ValueError(f"Unknown assay type: {assay}")

    mapping = assay_mappings[assay]

    # Initialize data structure
    data = {
        # Input parameters
        mapping["constant"]: None,
        "h0": None if assay in ["ida", "gda"] else None,
        "Kd": None if assay in ["ida", "gda"] else None,
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
            elif "h0 (M):" in line and assay in ["dba_HtoD", "dba_DtoH"]:
                try:
                    value_str = line.split(":")[1].strip()
                    data["h0"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass
            elif "Kd (M^-1):" in line and assay in ["dba_HtoD", "dba_DtoH"]:
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
            elif "I_d (signal/M):" in line or "Id (signal/M):" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["median_params"]["Id"] = float(
                        re.sub(r"[^\d.eE+-]", "", value_str)
                    )
                except:
                    pass
            elif "I_hd (signal/M):" in line or "Ihd (signal/M):" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["median_params"]["Ihd"] = float(
                        re.sub(r"[^\d.eE+-]", "", value_str)
                    )
                except:
                    pass
            elif "RMSE:" in line:
                try:
                    value_str = line.split(":")[1].strip()
                    data["rmse"] = float(re.sub(r"[^\d.eE+-]", "", value_str))
                except:
                    pass
            elif "R²:" in line:
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


def run_fitting_routine():
    pass
