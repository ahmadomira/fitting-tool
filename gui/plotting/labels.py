"""HTML display labels for parameter names.

Used by FitSummaryWidget and plot annotations to render math-style names
in Qt widgets that support HTML (QLabel, QFormLayout, pg.TextItem).

Value and unit formatting is handled by Pint's built-in formatters:
  - ``f"{Q_(value, unit):.3g~H}"`` for HTML values+units
  - ``f"{Q_(value, 'dimensionless'):.3g~P}"`` for pretty Unicode values
  - ``f"{ureg.Unit(unit_str):~H}"`` for HTML unit display

Reciprocal units (e.g. 1/M) are post-processed into negative-exponent
notation (M⁻¹ / M<sup>−1</sup>) because Pint 0.25 does not have a
format modifier for that.
"""

from __future__ import annotations

from core.units import ureg


def _canonical_unit(unit_str: str) -> str:
    """Map the signal unit's display alias ``a.u.`` to its parseable token ``au``.

    Pint parses the dotted ``a.u.`` as ``year · atomic_mass_unit`` (the dots are
    multiplications), so any unit string reaching the formatters below must use
    ``au``. The registry stores the y-axis unit as ``a.u.`` for raw display; this
    guard keeps that alias from garbling if it is ever formatted here.
    """
    return 'au' if unit_str.strip() in ('a.u.', 'a.u') else unit_str


PARAM_LABELS: dict[str, str] = {
    'Ka_guest': '<b>K<sub>a(G)</sub></b>',
    'Ka_dye': '<b>K<sub>a(D)</sub></b>',
    'I0': '<b>I<sub>0</sub></b>',
    'I_dye_free': '<b>I<sub>D</sub></b>',
    'I_dye_bound': '<b>I<sub>HD</sub></b>',
    'slope': '<b>slope</b>',
    'intercept': '<b>intercept</b>',
    # Stepwise 1:2 / 2:1 host–guest parameters
    'Ka_HG': '<b>K<sub>a(HG)</sub></b>',
    'Ka_HG2': '<b>K<sub>a(HG₂)</sub></b>',
    'Ka_H2G': '<b>K<sub>a(H₂G)</sub></b>',
    'I_G': '<b>I<sub>G</sub></b>',
    'I_H': '<b>I<sub>H</sub></b>',
    'I_HG': '<b>I<sub>HG</sub></b>',
    'I_HG2': '<b>I<sub>HG₂</sub></b>',
    'I_H2G': '<b>I<sub>H₂G</sub></b>',
}


def fmt_param(name: str) -> str:
    """Return HTML display label for a parameter name, fallback to raw name."""
    return PARAM_LABELS.get(name, name)


# Plain-text (no HTML) labels for widgets that cannot render rich text,
# e.g. QCheckBox toggle labels. Uses Unicode subscripts where unambiguous.
PARAM_LABELS_PLAIN: dict[str, str] = {
    'Ka_guest': 'Ka(G)',
    'Ka_dye': 'Ka(D)',
    'I0': 'I₀',
    'I_dye_free': 'I(D)',
    'I_dye_bound': 'I(HD)',
    'slope': 'slope',
    'intercept': 'intercept',
    'Ka_HG': 'Ka(HG)',
    'Ka_HG2': 'Ka(HG₂)',
    'Ka_H2G': 'Ka(H₂G)',
    'I_G': 'I(G)',
    'I_H': 'I(H)',
    'I_HG': 'I(HG)',
    'I_HG2': 'I(HG₂)',
    'I_H2G': 'I(H₂G)',
}


def fmt_param_plain(name: str) -> str:
    """Return a plain-text (no HTML) parameter label, fallback to raw name."""
    return PARAM_LABELS_PLAIN.get(name, name)


# Bracketed concentration labels for equilibrium species, keyed by the species
# names returned by ``BaseAssay.species()``.  Unicode subscripts so they render
# in plain-text widgets (legend entries, hover readouts) as well as rich text.
SPECIES_LABELS: dict[str, str] = {
    'H': '[H]',
    'D': '[D]',
    'G': '[G]',
    'HD': '[HD]',
    'HG': '[HG]',
    'HG2': '[HG₂]',
    'H2G': '[H₂G]',
}


def fmt_species(key: str, *, brackets: bool = True) -> str:
    """Return a concentration label for a species, e.g. ``HG2`` → ``[HG₂]``.

    ``brackets=False`` drops them (``HG₂``) — used for the compact plot legend,
    where the concentration-bracket notation isn't needed.
    """
    label = SPECIES_LABELS.get(key, f'[{key}]')
    return label if brackets else label.strip('[]')


def fmt_unit_html(unit_str: str) -> str:
    """Format a unit string as abbreviated HTML with negative exponents.

    Pint's ``~H`` formatter renders reciprocal units as fractions (e.g.
    ``1/M``).  This function post-processes the output to use superscript
    negative exponents instead (e.g. ``M<sup>−1</sup>``).
    """
    if not unit_str:
        return ''
    formatted = f'{ureg.Unit(_canonical_unit(unit_str)):~H}'
    if formatted.startswith('1/'):
        base = formatted[2:].strip()
        return f'{base}<sup>−1</sup>'
    return formatted


def fmt_unit_pretty(unit_str: str) -> str:
    """Format a unit string as abbreviated Unicode with negative exponents.

    Like :func:`fmt_unit_html` but returns plain Unicode (e.g. ``M⁻¹``)
    suitable for text files.
    """
    if not unit_str:
        return ''
    formatted = f'{ureg.Unit(_canonical_unit(unit_str)):~P}'
    if formatted.startswith('1/'):
        base = formatted[2:].strip()
        return f'{base}⁻¹'
    return formatted
