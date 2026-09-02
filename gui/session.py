"""Session-level helpers: JSON/TXT/CSV export and import of fit results, plot image export."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from core.pipeline.fit_pipeline import FitResult, ParameterSummary, summarize_parameters
from core.units import Quantity
from gui.plotting.labels import fmt_unit_pretty

if TYPE_CHECKING:
    from gui.fitting_session import FittingSession


def export_results(results: list[FitResult], path: str | Path) -> None:
    """Export a list of FitResult objects to a JSON file.

    Parameters
    ----------
    results : list[FitResult]
        Fit results to export.
    path : str or Path
        Output file path (should have ``.json`` extension).
    """
    path = Path(path)
    data = [r.to_dict() for r in results]
    path.write_text(json.dumps(data, indent=2))


def import_results(path: str | Path) -> list[FitResult]:
    """Import fit results from a JSON file created by :func:`export_results`.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    list[FitResult]
        Reconstructed FitResult objects.
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = [raw]
    return [FitResult.from_dict(d) for d in raw]


def _row_label(spec: ParameterSummary) -> str:
    """Plain-text row label; log twins are named ``log10(<key>)``."""
    return f'log10({spec.key})' if spec.is_log else spec.key


def _row_unit(spec: ParameterSummary) -> str:
    """Unit shown for a row.

    A log₁₀ value is dimensionless, but the number is only meaningful against
    the unit its parameter was measured in — so the row names that reference
    rather than dropping it.
    """
    if not spec.unit:
        return '—'
    pretty = fmt_unit_pretty(spec.unit)
    return f'log10({pretty})' if spec.is_log else pretty


def _fixed_width_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    """Render *rows* as space-padded columns sized to their widest entry.

    The first column is left-aligned (names), the numeric middle columns are
    right-aligned, and the trailing Units column reads better left-aligned.
    """
    if not rows:
        return []
    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]

    def render(cells: tuple[str, ...]) -> str:
        out = [f'  {cells[0]:<{widths[0]}}']
        out += [f'{cells[i]:>{widths[i]}}' for i in range(1, len(cells) - 1)]
        out.append(f'{cells[-1]:<{widths[-1]}}')
        return '   '.join(out).rstrip()

    header = render(headers)
    body = [render(row) for row in rows]
    rule = '  ' + '-' * (max(len(line) for line in (header, *body)) - 2)
    return [header, rule, *body]


def export_results_csv(results: list[FitResult], path: str | Path) -> None:
    """Export fit results as a tidy CSV — one row per reported parameter.

    Complements the TXT report: same numbers, but machine-readable so the
    values can go straight into a spreadsheet or a plotting script. Magnitudes
    are in the stored base units, named per row in the ``unit`` column. Pool
    statistics are left blank when the result carries no fit pool.

    Parameters
    ----------
    results : list[FitResult]
        Fit results to export.
    path : str or Path
        Output file path (should have ``.csv`` extension).
    """
    path = Path(path)
    columns = (
        'result_index',
        'assay_type',
        'source_file',
        'parameter',
        'unit',
        'estimate',
        'min',
        'max',
        'median',
        'mad',
        'mean',
        'std',
        'p16',
        'p84',
        'n_passing',
        'n_total',
        'rmse',
        'r_squared',
    )
    stat_keys = ('min', 'max', 'median', 'mad', 'mean', 'std', 'p16', 'p84')

    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, result in enumerate(results):
            for spec in summarize_parameters(result):
                stats = spec.stats or {}
                writer.writerow(
                    [
                        index,
                        result.assay_type,
                        result.source_file or '',
                        _row_label(spec),
                        # Log rows are dimensionless; the row they log is in
                        # the same file, so its unit is one lookup away.
                        '' if spec.is_log else spec.unit,
                        f'{spec.estimate:.6e}',
                        *(f'{stats[k]:.6e}' if k in stats else '' for k in stat_keys),
                        result.n_passing,
                        result.n_total,
                        f'{result.rmse:.6e}',
                        f'{result.r_squared:.6f}',
                    ]
                )


def export_results_txt(results: list[FitResult], path: str | Path) -> None:
    """Export fit results as a human-readable text report.

    Parameters
    ----------
    results : list[FitResult]
        Fit results to export (typically a single-element list).
    path : str or Path
        Output file path (should have ``.txt`` extension).
    """
    path = Path(path)
    lines: list[str] = []

    lines.append('=' * 60)
    lines.append('FIT RESULTS REPORT')
    lines.append('=' * 60)
    lines.append(f'Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}')
    lines.append('')

    for i, result in enumerate(results):
        if len(results) > 1:
            lines.append(f'--- Result {i + 1} of {len(results)} ---')
            lines.append('')

        # Header
        lines.append(f'Assay type:  {result.assay_type}')
        lines.append(f'Model:       {result.model_name}')
        if result.source_file:
            lines.append(f'Source file: {result.source_file}')
        lines.append(f'Timestamp:   {result.timestamp}')
        lines.append('')

        # Parameters — the headline block, then the pool it came from.
        specs = summarize_parameters(result)
        pooled = any(spec.stats is not None for spec in specs)

        lines.append('FITTED PARAMETERS')
        lines.append('-' * 60)
        lines.append(f'Estimate = the single best fit (highest R²) of {result.n_passing} accepted fits.')
        if pooled:
            lines.append(f'Range    = (min, max) across those {result.n_passing} fits.')
        else:
            lines.append('Range    = unavailable: no fit pool is stored for this result.')
        lines.append('')
        lines.extend(
            _fixed_width_table(
                ('Parameter', 'Estimate', 'Range (min, max)', 'Units'),
                [
                    (
                        _row_label(spec),
                        f'{spec.estimate:.4g}',
                        f'({spec.stats["min"]:.4g}, {spec.stats["max"]:.4g})' if spec.stats else '—',
                        _row_unit(spec),
                    )
                    for spec in specs
                ],
            )
        )
        lines.append('')

        if pooled:
            lines.append(f'SPREAD ACROSS THE ACCEPTED-FIT POOL (n = {result.n_passing})')
            lines.append('-' * 60)
            lines.extend(
                _fixed_width_table(
                    ('Parameter', 'Median ± MAD', 'Mean ± SD', '68% Range [p16, p84]', 'Units'),
                    [
                        (
                            _row_label(spec),
                            f'{spec.stats["median"]:.4g} ± {spec.stats["mad"]:.4g}',
                            f'{spec.stats["mean"]:.4g} ± {spec.stats["std"]:.4g}',
                            f'[{spec.stats["p16"]:.4g}, {spec.stats["p84"]:.4g}]',
                            _row_unit(spec),
                        )
                        for spec in specs
                        if spec.stats is not None
                    ],
                )
            )
            lines.append('')

        # Fit quality
        lines.append('FIT QUALITY')
        lines.append('-' * 60)
        lines.append(f'  RMSE:          {result.rmse:.4g} {fmt_unit_pretty("au")}')
        lines.append(f'  R²:            {result.r_squared:.6f}')
        lines.append(f'  Fits passing:  {result.n_passing} / {result.n_total}')
        lines.append('')

        # Conditions
        if result.conditions:
            lines.append('CONDITIONS')
            lines.append('-' * 60)
            for cond_key, cond_val in result.conditions.items():
                if isinstance(cond_val, Quantity):
                    lines.append(f'  {cond_key}: {cond_val:.4g~P}')
                else:
                    lines.append(f'  {cond_key}: {cond_val}')
            lines.append('')

        if i < len(results) - 1:
            lines.append('')

    lines.append('=' * 60)

    path.write_text('\n'.join(lines), encoding='utf-8')


# ---------------------------------------------------------------------------
# Multi-artefact export
# ---------------------------------------------------------------------------


@dataclass
class ExportableArtefact:
    """One thing the user can export in a batch.

    Attributes
    ----------
    key
        Stable identifier persisted under ``QSettings`` (e.g. ``raw_txt``).
    label
        Human-readable label shown in the dialog.
    suffix
        Filename suffix appended to the base, including the extension.
    available
        Whether the precondition for this artefact is currently met.
    unavailable_reason
        Tooltip text shown when ``available`` is False.
    writer
        Callable accepting a destination Path. Called only when the user
        selects this artefact and the precondition is met.
    """

    key: str
    label: str
    suffix: str
    available: bool
    unavailable_reason: str = ''
    writer: Callable[[Path], None] = field(default=lambda _p: None)


def build_artefacts(session: 'FittingSession') -> list[ExportableArtefact]:
    """Build the list of artefacts exportable from a FittingSession.

    Preconditions:
    - Raw data / fit curve plot: a measurement set is loaded.
    - Distribution plot: a fit has been run (uses fit_results).
    - Fit results (JSON/TXT/CSV): a fit has been run.
    - Style template: always available (style widget is always present).
    """
    state = session._state
    ms_loaded = state.measurement_set is not None
    has_results = bool(state.fit_results)

    no_data_msg = 'Load a measurement file first.'
    no_fit_msg = 'Run a fit first.'

    from core.io.formats.measurement_writer import (
        write_measurements_csv,
        write_measurements_txt,
    )
    from gui.plotting.plot_style import save_style_json

    def _write_raw_txt(path: Path) -> None:
        write_measurements_txt(state.measurement_set, str(path))

    def _write_raw_csv(path: Path) -> None:
        write_measurements_csv(state.measurement_set, str(path))

    def _write_results_json(path: Path) -> None:
        export_results(state.fit_results, path)

    def _write_results_txt(path: Path) -> None:
        export_results_txt(state.fit_results, path)

    def _write_results_csv(path: Path) -> None:
        export_results_csv(state.fit_results, path)

    def _write_fit_png(path: Path) -> None:
        session._plot_widget.export_image(str(path))

    def _write_fit_svg(path: Path) -> None:
        session._plot_widget.export_image(str(path))

    def _write_distributions(path: Path) -> None:
        cfg = session._distributions_export_config()
        session._distribution_widget.save_plot(
            keys=cfg.keys,
            rows=cfg.rows,
            cols=cfg.cols,
            width_in=cfg.width_in,
            dpi=cfg.dpi,
            path=str(path),
        )

    def _write_style(path: Path) -> None:
        style = session._style_widget.widget.current_style()
        save_style_json(style, path)

    return [
        ExportableArtefact(
            key='raw_txt',
            label='Raw data (TXT)',
            suffix='_data.txt',
            available=ms_loaded,
            unavailable_reason=no_data_msg,
            writer=_write_raw_txt,
        ),
        ExportableArtefact(
            key='raw_csv',
            label='Raw data (CSV)',
            suffix='_data.csv',
            available=ms_loaded,
            unavailable_reason=no_data_msg,
            writer=_write_raw_csv,
        ),
        ExportableArtefact(
            key='results_json',
            label='Fit results (JSON)',
            suffix='_results.json',
            available=has_results,
            unavailable_reason=no_fit_msg,
            writer=_write_results_json,
        ),
        ExportableArtefact(
            key='results_txt',
            label='Fit results (TXT report)',
            suffix='_results.txt',
            available=has_results,
            unavailable_reason=no_fit_msg,
            writer=_write_results_txt,
        ),
        ExportableArtefact(
            key='results_csv',
            label='Fit results (CSV table)',
            suffix='_results.csv',
            available=has_results,
            unavailable_reason=no_fit_msg,
            writer=_write_results_csv,
        ),
        ExportableArtefact(
            key='fit_png',
            label='Fit curve plot (PNG)',
            suffix='_fit.png',
            available=ms_loaded,
            unavailable_reason=no_data_msg,
            writer=_write_fit_png,
        ),
        ExportableArtefact(
            key='fit_svg',
            label='Fit curve plot (SVG)',
            suffix='_fit.svg',
            available=ms_loaded,
            unavailable_reason=no_data_msg,
            writer=_write_fit_svg,
        ),
        ExportableArtefact(
            key='distributions_png',
            label='Distributions plot (PNG)',
            suffix='_distributions.png',
            available=has_results,
            unavailable_reason=no_fit_msg,
            writer=_write_distributions,
        ),
        ExportableArtefact(
            key='style_json',
            label='Style template (JSON)',
            suffix='_style.json',
            available=True,
            writer=_write_style,
        ),
    ]


def export_batch(
    arts: list[ExportableArtefact],
    folder: Path,
    base: str,
) -> list[tuple[str, Path, Exception | None]]:
    """Run a batch of artefact writers and report per-item outcomes.

    Returns a list of ``(label, path, error_or_None)`` tuples in the same
    order as ``arts``. Errors are caught so a single failure does not
    abort the rest of the batch — the caller surfaces them.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    outcomes: list[tuple[str, Path, Exception | None]] = []
    for art in arts:
        dest = folder / f'{base}{art.suffix}'
        try:
            art.writer(dest)
            outcomes.append((art.label, dest, None))
        except Exception as exc:
            outcomes.append((art.label, dest, exc))
    return outcomes
