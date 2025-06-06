import numpy as np
from scipy.optimize import brentq

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
                h = h0 - h_d
                Signal = I0 + Id * d_free + Ihd * h_d
                Signal_values.append(Signal)
            except Exception:
                Signal_values.append(np.nan)
        return np.array(Signal_values)