import numpy as np
import matplotlib.pyplot as plt

from forward_model import compute_signal_ida
from plot_utils import create_plots

plot_config = {
    # defaults 
    'x_label': r'Concentration $\rm{[\mu M]}$',
    'y_label': r'Signal $\rm{[AU]}$',     
    'annotation_config': {
        'xy': (0.8, 0.04),
        'ha': 'left',
        'va': 'bottom'
        },
    'legend_config': {
        'loc': 'upper left',
        'bbox_to_anchor': (0.02, 0.98)
    },
    'K_g' : 'K_{a (Guest)}',
    
    'ida' : {
        'x_label': r'$G_0$ $\rm{[\mu M]}$',
        'annotation_config': {
            'xy': (0.97, 0.95),
            'ha': 'right',
            'va': 'top'
            },
        'legend_config': {
            'loc': 'lower left',
            'bbox_to_anchor': (0.02, 0.02)
        }
        
    },

}
config = plot_config.get('ida')

ka_guest_values = 10 ** np.linspace(8.5, 10, 10, endpoint=False)

fig, ax = create_plots(x_label=config['x_label'])
K_text = config.get('K_d', plot_config['K_g'])

for ka_guest in ka_guest_values:
    
    # params: [I0 (M^-1), Kg (M^-1), Id (M^-1), Ihd (M^-1)]
    params_ida = [1.02e+04, ka_guest, 7.69e+07, 6.37e+11]

    I_0, k, I_d, I_hd = params_ida

    # g0_values in (M)
    # g0_values = np.array([0.0, 1e-06, 2e-06, 2.5e-06, 3e-06, 3.5e-06, 4e-06, 4.5e-06, 5e-06, 6e-06, 8e-06, 9e-06])
    g0_values = np.linspace(0e-6, 4e-06, 12)
    d0 = 7e-06      # in (M)
    h0 = 1e-06    # in (M)
    Kd = 1.68e+8   # in (M^-1)

    signal_values_ida = compute_signal_ida(params_ida, g0_values, Kd, h0, d0)

    # Convert scientific notation to LaTeX format
    def sci_to_latex(value):
        return f"{value:.2e}".replace('e', r' \times 10^{') + '}'

    param_text = (  f"$D_0$: {d0 * 1e6:.0f} $µM$\n"
                    f"$H_0$: {h0 * 1e6:.0f} $µM$\n"
                    f"$K_{{a(Dye)}}$: ${sci_to_latex(Kd)}$\n"
                    f"$I_0$: ${sci_to_latex(I_0)}$\n"
                    f"$I_d$: ${sci_to_latex(I_d)}$ $M^{{-1}}$\n"
                    f"$I_{{hd}}$: ${sci_to_latex(I_hd)}$ $M^{{-1}}$\n"
                )

    annot_config = config.get('annotation_config', plot_config['annotation_config'])
    ax.annotate(param_text, xy=annot_config['xy'], xycoords='axes fraction', fontsize=10,
                ha=annot_config['ha'], va=annot_config['va'], bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey", alpha=0.5), multialignment='left')

    ax.plot(g0_values * 1e6, signal_values_ida, label=f"${sci_to_latex(k)}$ $M^{{-1}}$")
    
    legend_config = config.get('legend_config', plot_config['legend_config'])
    legend = ax.legend(loc=legend_config['loc'], bbox_to_anchor=legend_config['bbox_to_anchor'])
    legend.set_title(f"${K_text}$")
    ax.set_title(f'IDA Signal Curves for Different Binding Affinity Values')
plt.show()
