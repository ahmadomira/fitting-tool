import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

font_size = 11
plt.rcParams.update(
    {
        "figure.figsize": (8, 6),
        "figure.dpi": 120,
        "figure.titlesize": font_size,
        "figure.titleweight": "bold",
        "figure.autolayout": True,
        "lines.marker": "o",
        "lines.linewidth": 1.5,
        "font.weight": "light",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.titleweight": "light",
        "axes.labelsize": font_size,
        "axes.labelweight": "light",
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.grid.which": "both",
        "axes.formatter.use_mathtext": True,
        "axes.formatter.limits": (-2, 3),
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.major.size": 5,
        "xtick.minor.size": 3,
        "xtick.labelsize": font_size - 1,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.major.size": 5,
        "ytick.minor.size": 3,
        "ytick.labelsize": font_size - 1,
        "legend.loc": "best",
        "legend.frameon": True,
        "legend.framealpha": 0.5,
        "legend.fancybox": True,
        "legend.shadow": False,
        "legend.fontsize": font_size - 2,
        "grid.color": "gray",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "savefig.bbox": "tight",
        "savefig.format": "png",
    }
)
