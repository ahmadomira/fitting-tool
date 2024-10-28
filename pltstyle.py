import matplotlib.pyplot as plt
font_size = 11
plt.rcParams.update({
    'figure.figsize': (10, 7),
    'figure.dpi': 150,
    'figure.titlesize': font_size,
    'figure.titleweight': 'bold',
    'figure.autolayout': True,

    'lines.marker': 'o',
    'lines.linewidth': 1.5,

    'font.weight': 'light',
    'font.size': font_size,

    'axes.titlesize': font_size,
    'axes.titleweight': 'light',
    'axes.labelsize': font_size - 2,
    'axes.labelweight': 'light',
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'both',
    'axes.formatter.use_mathtext': True,
    'axes.formatter.limits': (-2, 2),

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