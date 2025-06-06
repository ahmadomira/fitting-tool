import numpy as np

from forward_model import compute_signal_dba, compute_signal_gda, compute_signal_ida

# params: [I0 (M^-1), Kg (M^-1), Id (M^-1), Ihd (M^-1)]

## GDA 
params_gda = [8.77e+01, 7.79e+09, 1.00e+03, 7.77e+09]
d0_values = np.array([0.0, 2.985e-05, 5.941e-05, 8.867e-05, 1.1765e-04, 1.4634e-04, 1.7476e-04, 2.029e-04, 2.3077e-04, 2.5837e-04, 2.8571e-04, 3.128e-04, 3.3962e-04, 3.662e-04, 3.9252e-04, 4.186e-04, 4.4444e-04, 4.7005e-04, 4.9541e-04, 5.2055e-04, 5.4545e-04, 5.7014e-04, 5.9459e-04, 6.1883e-04])

g0 = 6e-06  # (M)
h0 = 4.3e-06  # (M)
Kd = 1.68e+07  # (M^-1)

signal_values_gda = compute_signal_gda(params_gda, d0_values, Kd, h0, g0)

## IDA 
params_ida = [1.02e+04, 3.13e+08, 7.69e+06, 6.37e+10]

# g0_values in (M)
g0_values = np.array([0.0, 1e-06, 2e-06, 2.5e-06, 3e-06, 3.5e-06, 4e-06, 4.5e-06, 5e-06, 6e-06, 8e-06, 9e-06])
d0 = 6e-06      # in (M)
h0 = 4.3e-06    # in (M)
Kd = 1.68e+07   # in (M^-1)

signal_values_ida = compute_signal_ida(params_ida, g0_values, Kd, h0, d0)

## DBA
params_dba = [1.06e+03, 2.17e+03, 1.69e+08, 4.69e+09]
h0_values = np.array([0.0, 3e-05, 6e-05, 1e-04, 1.5e-04, 2.25e-04, 3e-04, 4e-04, 5e-04, 6e-04, 7e-04, 8.4e-04])

d0 = 6e-06      # in (M)

signal_values_dba = compute_signal_dba(params_dba, g0_values, d0)

from plot_utils import create_plots
import matplotlib.pyplot as plt

fig, ax = create_plots()

ax.plot(g0_values, signal_values_ida, label="Signal", linestyle='none')

plt.show()