"""
DBA Merge Fits logic - refactored to use the unified merge framework.
"""

from core.fitting.base_merge import run_merge_fits


def run_dba_merge_fits(
    results_dir,
    outlier_relative_threshold,
    rmse_threshold_factor,
    kd_threshold_factor,
    save_plots,
    display_plots,
    save_results,
    results_save_dir,
    custom_plot_title=None,
    assay_type="dba_HtoD",
    custom_x_label=None,
):
    """
    Run DBA merge fits with configurable assay type.

    Parameters:
    assay_type (str): Either 'dba_HtoD' or 'dba_DtoH' to specify the assay direction
    """
    return run_merge_fits(
        results_dir=results_dir,
        assay_type=assay_type,
        outlier_relative_threshold=outlier_relative_threshold,
        rmse_threshold_factor=rmse_threshold_factor,
        k_threshold_factor=kd_threshold_factor,  # Map kd_threshold_factor to k_threshold_factor
        save_plots=save_plots,
        display_plots=display_plots,
        save_results=save_results,
        results_save_dir=results_save_dir,
        plot_title=custom_plot_title,
        custom_x_label=custom_x_label,
    )
