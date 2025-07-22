"""
IDA merge fitting logic - refactored to use the unified merge framework.
"""

from core.fitting.base_merge import run_merge_fits


def run_ida_merge_fits(
    results_dir,
    outlier_relative_threshold,
    rmse_threshold_factor,
    kg_threshold_factor,
    save_plots,
    display_plots,
    save_results,
    results_save_dir,
    plot_title,
):
    """
    Run IDA merge fits using the unified merge framework.
    """
    return run_merge_fits(
        results_dir=results_dir,
        assay_type="ida",
        outlier_relative_threshold=outlier_relative_threshold,
        rmse_threshold_factor=rmse_threshold_factor,
        k_threshold_factor=kg_threshold_factor,  # Map kg_threshold_factor to k_threshold_factor
        save_plots=save_plots,
        display_plots=display_plots,
        save_results=save_results,
        results_save_dir=results_save_dir,
        plot_title=plot_title,
    )
