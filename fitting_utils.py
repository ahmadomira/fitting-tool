
import numpy as np
from scipy.optimize import brentq, minimize
import os
from datetime import datetime

def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.readlines()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading file: {e}")
    except IOError as e:
        raise IOError(f"Error loading file: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

def split_replica(data):
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
            raise ValueError("No valid data found in the input file to split into replicas.")
        return replicas

def compute_signal_dba(params, x_titrations, y_fixed):
    r"""This function can compute the signal for Dye-to-Host and Host-to-Dye binding assays, depending on the values of x_titrations and y_fixed."""
    I0, Kd, Id, Ihd = params
    Signal_values = []
    for x in x_titrations:
        delta = x - y_fixed
        a = Kd
        b = Kd * delta + 1
        c = -y_fixed
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            Signal_values.append(np.nan)
            continue

        sqrt_discriminant = np.sqrt(discriminant)
        y1 = (-b + sqrt_discriminant) / (2 * a)
        y2 = (-b - sqrt_discriminant) / (2 * a)

        y = y1 if y1 >= 0 else y2 if y2 >= 0 else np.nan
        if np.isnan(y):
            Signal_values.append(np.nan)
            continue

        x_calc = y + delta
        hd = Kd * y * x_calc
        Signal = I0 + Id * y + Ihd * hd
        Signal_values.append(Signal)

    return np.array(Signal_values)

def compute_signal_gda(params, d0_values, Kd, h0, g0):
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

def compute_signal_ida(params, g0_values, Kd, h0, d0):
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
