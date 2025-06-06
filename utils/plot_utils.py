import os
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

font_size = 11
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'figure.dpi': 120,
    'figure.titlesize': font_size,
    'figure.titleweight': 'bold',
    'figure.autolayout': True,

    'lines.marker': 'o',
    'lines.linewidth': 1.5,

    'font.weight': 'light',
    'font.size': font_size,

    'axes.titlesize': font_size,
    'axes.titleweight': 'bold',
    'axes.labelsize': font_size - 2,
    'axes.labelweight': 'light',
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'both',
    'axes.formatter.use_mathtext': True,
    'axes.formatter.limits': (-2, 3),

    'xtick.direction': 'in',
    'xtick.minor.visible': True,
    'xtick.major.size': 5,
    'xtick.minor.size': 3,
    'xtick.labelsize': font_size - 2,

    'ytick.direction': 'in',
    'ytick.minor.visible': True,
    'ytick.major.size': 5,
    'ytick.minor.size': 3,
    'ytick.labelsize': font_size - 2,

    'legend.loc': 'best',
    'legend.frameon': True,
    'legend.framealpha': 0.5,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.fontsize': font_size - 4,

    'grid.color': 'gray',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7,

    'savefig.bbox': 'tight',
    'savefig.format': 'png',
})

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
    
    # specific configurations
    'dba_HtoD' : {
        'x_label' : r'$H_0$ $\rm{[\mu M]}$',
        'K_d' : 'K_d' 
    },
    
    'dba_DtoH' : {
        'x_label' : r'$D_0$ $\rm{[\mu M]}$',
        'K_d' : 'K_d' 
    },
    
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
    
    'gda' : {
        'x_label': r'$D_0$ $\rm{[\mu M]}$',
    }
}
    
def create_plots(x_label=r'Concentration $\rm{[\mu M]}$', 
                 y_label=r'Intensity $\rm{[AU]}$', 
                 suptitle='', plot_title='', 
                 *args, **kwargs):
    
    fig, ax = plt.subplots(*args, **kwargs)
    
    ax.grid(which='major', linestyle=':', linewidth='0.5', color='gray')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
    
    if suptitle != '':
        fig.suptitle(f'{suptitle}')
    
    if plot_title != '':
        ax.set_title(f'{plot_title}')
    
    ax.set_xlabel(f'{x_label}')
    ax.set_ylabel(f'{y_label}')
    
    return fig, ax

def format_value(value):
    return f"{value:.0f}" if value > 10 else f"{value:.3f}"

def plot_fitting_results(fitting_params, median_params, assay):
    x_values, Signal_observed, fitting_curve_x, fitting_curve_y, replica_index = fitting_params
    I_0, k, I_d, I_hd, rmse, r_squared = median_params
    plot_title = f'Replica {replica_index}'
    
    # TODO: implement a "dynamic" unit and scale for the x-axis
    config = plot_config.get(assay)
    x_label = config.get('x_label', plot_config['x_label'])
    y_label = config.get('y_label', plot_config['y_label'])
    
    fig, ax = create_plots(x_label=x_label, y_label=y_label)

    ax.plot(x_values, Signal_observed, 'o', label='Observed Signal')
    ax.plot(fitting_curve_x, fitting_curve_y, '--', color='blue', alpha=0.6, label='Simulated Fitting Curve')
    
    #plot_title = f'Observed vs. Simulated Fitting Curve ({plot_title})'
    
    #ax.set_title(plot_title)
    
    K_text = config.get('K_d', plot_config['K_g'])
    param_text = (f"${K_text}$: {k * 1e6:.2e} $M^{{-1}}$\n"
                  f"$I_0$: {I_0:.2e}\n"
                  f"$I_d$: {I_d * 1e6:.2e} $M^{{-1}}$\n"
                  f"$I_{{hd}}$: {I_hd * 1e6:.2e} $M^{{-1}}$\n"
                #   f"$RMSE$: {format_value(rmse)}\n"
                #   f"$R^2$: {r_squared:.3f}"
                )

    annot_config = config.get('annotation_config', plot_config['annotation_config'])
    ax.annotate(param_text, xy=annot_config['xy'], xycoords='axes fraction', fontsize=10,
                ha=annot_config['ha'], va=annot_config['va'], bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey", alpha=0.5), multialignment='left')
    
    legend_config = config.get('legend_config', plot_config['legend_config'])
    ax.legend(loc=legend_config['loc'], bbox_to_anchor=legend_config['bbox_to_anchor'])

    # the label is used in save_plot as the filename for saving the plot
    fig.set_label(f"fit_plot_replica_{replica_index}")
    return fig

def save_plot(fig, plots_dir):
    filename = f"{fig.get_label()}.png"
    plot_file = os.path.join(plots_dir, filename)
    fig.savefig(plot_file, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")