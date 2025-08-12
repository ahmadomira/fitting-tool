import os

import matplotlib
import numpy as np

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from core.fitting_utils import to_M

font_size = 11
plt.rcParams.update(
    {
        "figure.figsize": (8, 6),
        "figure.dpi": 120,
        "figure.titlesize": font_size,
        "axes.titlesize": font_size,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size - 2,
        "figure.titleweight": "bold",
        "figure.autolayout": True,
        "lines.marker": "o",
        "lines.linewidth": 1.5,
        "font.weight": "light",
        "axes.titleweight": "light",
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
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.major.size": 5,
        "ytick.minor.size": 3,
        "legend.loc": "best",
        "legend.frameon": True,
        "legend.framealpha": 0.5,
        "legend.fancybox": True,
        "legend.shadow": False,
        "grid.color": "gray",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "savefig.bbox": "tight",
        "savefig.format": "png"
    }
)

plot_config = {
    # defaults
    "x_label": r"Concentration $\rm{[\mu M]}$",
    "y_label": r"Signal $\rm{[AU]}$",
    "K_g": "K_{a(G)}",
    # specific configurations
    "dba_HtoD": {"x_label": r"$H_0$ $\rm{[\mu M]}$", "K_d": "K_{a(D)}"},
    "dba_DtoH": {"x_label": r"$D_0$ $\rm{[\mu M]}$", "K_d": "K_{a(D)}"},
    "ida": {
        "x_label": r"$G_0$ $\rm{[\mu M]}$",
    },
    "gda": {
        "x_label": r"$D_0$ $\rm{[\mu M]}$",
    },
}


def scientific_notation(val, pos=0):
    if val == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(val))))
    coeff = val / 10**exponent
    return r"{:.2f} \times 10^{{{}}}".format(coeff, exponent)


def create_plots(
    x_label=r"Concentration $\rm{[\mu M]}$",
    y_label=r"Intensity $\rm{[AU]}$",
    suptitle="",
    plot_title="",
    *args,
    **kwargs,
):

    fig, ax = plt.subplots(*args, **kwargs)

    ax.grid(which="major", linestyle=":", linewidth="0.5", color="gray")
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="lightgray")

    if suptitle != "":
        fig.suptitle(f"{suptitle}")

    if plot_title != "":
        ax.set_title(f"{plot_title}")

    ax.set_xlabel(f"{x_label}")
    ax.set_ylabel(f"{y_label}")

    return fig, ax


def format_value(value):
    return f"{value:.0f}" if value > 10 else f"{value:.3f}"


def plot_fitting_results(
    fitting_params, median_params_uM, assay, custom_x_label=None, custom_plot_title=None
):
    x_values_uM, Signal_observed, fitting_curve_x_uM, fitting_curve_y, replica_index = (
        fitting_params
    )

    I_0, k, I_d, I_hd, rmse, r_squared = median_params_uM  # in µM⁻¹ from fitting step

    # Use custom plot title if provided, otherwise use default
    if custom_plot_title:
        full_plot_title = f"{custom_plot_title} - Replica {replica_index}"
    else:
        full_plot_title = (
            f"Observed vs. Simulated Fitting Curve (Replica {replica_index})"
        )

    config = plot_config.get(assay)
    # Use custom x_label if provided, otherwise use automatic selection
    if custom_x_label:
        x_label = custom_x_label + r" $\rm{[\mu M]}$"
    else:
        x_label = config.get("x_label", plot_config["x_label"])
    y_label = config.get("y_label", plot_config["y_label"])

    fig, ax = create_plots(x_label=x_label, y_label=y_label)

    ax.plot(x_values_uM, Signal_observed, "o", label="Observed Signal")
    ax.plot(
        fitting_curve_x_uM,
        fitting_curve_y,
        "--",
        color="blue",
        alpha=0.6,
        label="Simulated Fitting Curve",
    )

    ax.set_title(full_plot_title)

    K_text = config.get("K_d", plot_config["K_g"])
    param_text = (
        # k, Id, Ihd are in µM⁻¹
        f"${K_text}$: {k / to_M(1.0):.2e} $M^{{-1}}$\n"
        f"$I_0$: {I_0:.2e}\n"
        f"$I_d$: {I_d / to_M(1.0):.2e} $M^{{-1}}$\n"
        f"$I_{{hd}}$: {I_hd / to_M(1.0):.2e} $M^{{-1}}$\n"
        # f"$RMSE$: {format_value(rmse)}\n"
        f"$R^2$: {r_squared:.3f}"
    )

    # Find optimal positions for both legend and annotation to avoid data overlap
    legend, annotation = place_legend_and_annotation_safely(ax, param_text)

    # the label is used in save_plot as the filename for saving the plot
    fig.set_label(f"fit_plot_replica_{replica_index}")

    return fig


def save_plot(fig, plots_dir):
    filename = f"{fig.get_label()}.png"
    plot_file = os.path.join(plots_dir, filename)
    fig.savefig(plot_file, bbox_inches="tight", dpi=300)
    print(f"Plot saved to {plot_file}")


def place_legend_and_annotation_safely(
    ax, annotation_text, annotation_fontsize=font_size - 2, min_distance=20
):
    """
    Optimized placement to avoid overlapping data points.
    Uses a faster approach with fewer matplotlib operations.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    # Force a single draw to ensure all elements are rendered
    fig.canvas.draw()

    # Collect all data points in display coordinates
    x_all, y_all = [], []
    for line in ax.get_lines():
        x_all.extend(line.get_xdata(orig=False))
        y_all.extend(line.get_ydata(orig=False))

    if x_all:
        data_disp = ax.transData.transform(np.column_stack((x_all, y_all)))
    else:
        data_disp = np.empty((0, 2))

    # Helper function to calculate minimum distance from bbox to data points
    def min_distance_to_data(bbox, data_points):
        if len(data_points) == 0:
            return float("inf")

        # Vectorized distance calculation for better performance
        px, py = data_points[:, 0], data_points[:, 1]
        xmin, ymin, xmax, ymax = bbox.x0, bbox.y0, bbox.x1, bbox.y1

        # Calculate distance from points to rectangle
        dx = np.maximum(0, np.maximum(xmin - px, px - xmax))
        dy = np.maximum(0, np.maximum(ymin - py, py - ymax))
        distances = np.hypot(dx, dy)

        return np.min(distances)

    # Reduced candidates for better performance
    legend_locations = ["upper right", "upper left", "lower left", "lower right"]
    annotation_candidates = [
        (0.03, 0.97, "left", "top"),
        (0.97, 0.97, "right", "top"),
        (0.03, 0.03, "left", "bottom"),
        (0.97, 0.03, "right", "bottom"),
    ]

    best_score = -1
    best_legend_loc = "upper right"
    best_ann_pos = (0.97, 0.03)
    best_ann_align = ("right", "bottom")

    # Pre-calculate legend bounding boxes (without drawing each time)
    legend_boxes = {}
    for legend_loc in legend_locations:
        temp_legend = ax.legend(loc=legend_loc, fancybox=True, framealpha=0.5)
        fig.canvas.draw()  # Single draw per legend
        legend_boxes[legend_loc] = {
            "bbox": temp_legend.get_window_extent(renderer),
            "dist": min_distance_to_data(
                temp_legend.get_window_extent(renderer), data_disp
            ),
        }
        temp_legend.remove()

    # Now test annotation positions efficiently
    for legend_loc, legend_info in legend_boxes.items():
        legend_bbox = legend_info["bbox"]
        legend_data_dist = legend_info["dist"]

        for x, y, ha, va in annotation_candidates:
            # Create temporary annotation to get its bbox
            temp_ann = ax.annotate(
                annotation_text,
                xy=(x, y),
                xycoords="axes fraction",
                ha=ha,
                va=va,
                fontsize=annotation_fontsize,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="black",
                    facecolor="lightgrey",
                    alpha=0.5,
                ),
                multialignment="left",
            )
            fig.canvas.draw()  # Single draw per annotation
            ann_bbox = temp_ann.get_window_extent(renderer)
            temp_ann.remove()

            # Skip if annotation overlaps with legend
            if ann_bbox.overlaps(legend_bbox):
                continue

            # Calculate annotation's distance to data
            ann_data_dist = min_distance_to_data(ann_bbox, data_disp)
            combo_score = min(legend_data_dist, ann_data_dist)

            # Check if this is good enough and better than previous
            if (
                legend_data_dist >= min_distance
                and ann_data_dist >= min_distance
                and combo_score > best_score
            ):
                best_score = combo_score
                best_legend_loc = legend_loc
                best_ann_pos = (x, y)
                best_ann_align = (ha, va)

                # If we found a really good one, use it immediately
                if combo_score > 50:
                    break

        # Early exit if we found a great combination
        if best_score > 50:
            break

    # Create the final legend and annotation
    final_legend = ax.legend(loc=best_legend_loc, fancybox=True, framealpha=0.5)
    x, y = best_ann_pos
    ha, va = best_ann_align

    final_annotation = ax.annotate(
        annotation_text,
        xy=(x, y),
        xycoords="axes fraction",
        ha=ha,
        va=va,
        fontsize=annotation_fontsize,
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="black",
            facecolor="lightgrey",
            alpha=0.5,
        ),
        multialignment="left",
    )

    return final_legend, final_annotation
