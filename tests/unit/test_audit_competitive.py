"""Competitive-binding checks using conservation and manufactured equilibria."""

import numpy as np
import pytest

from core.models.equilibrium import (
    competitive_species_point,
    gda_signal,
    gda_species,
    ida_signal,
    ida_species,
)


@pytest.mark.parametrize('scale', [1.0, 1e-6, 1e-12, 1e-18, 1e-24])
def test_manufactured_equilibrium_is_invariant_to_concentration_scale(scale):
    # Choose free H,D,G = 1,2,3; mass action gives HD,HG = 1,6.
    # Totals 8,3,9 and inverse-scaled Ka preserve every occupancy.
    sp = competitive_species_point(2 / scale, 0.5 / scale, 8 * scale, 3 * scale, 9 * scale)
    for key, expected in {'H': 1, 'D': 2, 'G': 3, 'HD': 1, 'HG': 6}.items():
        assert sp[key] / scale == pytest.approx(expected, rel=2e-12, abs=0)


def test_zero_host_returns_unbound_ligands():
    sp = competitive_species_point(2e6, 5e5, 0, 3e-6, 9e-6)
    assert sp == {'H': 0, 'D': 3e-6, 'G': 9e-6, 'HD': 0, 'HG': 0}


def test_finite_extreme_affinity_does_not_overflow_bracket_evaluation():
    # Saturated dye leaves 5 host units; h=.5, HG=4.5 exactly close that balance.
    sp = competitive_species_point(2, 1e308, 8, 3, 9)
    for key, expected in {'H': 0.5, 'D': 6e-308, 'G': 4.5, 'HD': 3, 'HG': 4.5}.items():
        assert sp[key] == pytest.approx(expected, rel=2e-12, abs=0)


@pytest.mark.parametrize('field', range(5))
@pytest.mark.parametrize('invalid', [-1e-8, np.nan, np.inf, -np.inf])
def test_nonphysical_equilibrium_inputs_return_nan_species(field, invalid):
    args = [2e6, 5e5, 8e-6, 3e-6, 9e-6]
    args[field] = invalid
    assert all(np.isnan(value) for value in competitive_species_point(*args).values())


@pytest.mark.parametrize('model', ['gda', 'ida'])
def test_integer_titrant_does_not_truncate_fluorescence(model):
    # Same exact point H=1,D=2,G=3,HD=1,HG=6 with B=7.25.
    if model == 'gda':
        observed = gda_signal(7.25, 2, 2, 5, 0.5, 8, np.array([3]), 9)
    else:
        observed = ida_signal(7.25, 2, 2, 5, 0.5, 8, 3, np.array([9]))
    assert observed[0] == pytest.approx(16.25)


@pytest.mark.parametrize('a,b', [(0, 2), (0.5, 0), (0.5, 0.5), (1e-8, 1e8), (1e8, 1e-8)])
def test_species_obey_all_balances_and_both_mass_action_relations(a, b):
    sp = competitive_species_point(b, a, 8e-6, 3e-6, 9e-6)
    h, d, g, hd, hg = (sp[key] for key in ['H', 'D', 'G', 'HD', 'HG'])
    assert h + hd + hg == pytest.approx(8e-6, rel=2e-12, abs=0)
    assert d + hd == pytest.approx(3e-6, rel=2e-12, abs=0)
    assert g + hg == pytest.approx(9e-6, rel=2e-12, abs=0)
    assert hd == pytest.approx(a * h * d, rel=2e-12, abs=0)
    assert hg == pytest.approx(b * h * g, rel=2e-12, abs=0)
    assert 0 <= hd <= min(8e-6, 3e-6)
    assert 0 <= hg <= min(8e-6, 9e-6)


def test_ida_displacement_direction_changes_with_dye_brightness():
    titrant = np.array([0.0, 1.0, 9.0, 30.0, 100.0])
    sp = ida_species(2, 0.5, 8, 3, titrant)
    assert np.all(np.diff(sp['HD']) < 0)
    bright = ida_signal(7, 2, 2, 5, 0.5, 8, 3, titrant)
    quenched = ida_signal(7, 2, 5, 2, 0.5, 8, 3, titrant)
    assert np.all(np.diff(bright) < 0)
    assert np.all(np.diff(quenched) > 0)


def test_gda_increases_both_dye_species_while_displacing_guest():
    titrant = np.array([0.0, 1.0, 3.0, 10.0, 30.0])
    sp = gda_species(2, 0.5, 8, titrant, 9)
    assert np.all(np.diff(sp['HD']) > 0)
    assert np.all(np.diff(sp['D']) > 0)
    assert np.all(np.diff(sp['HG']) < 0)
    observed = gda_signal(7, 2, 2, 5, 0.5, 8, titrant, 9)
    np.testing.assert_allclose(observed, [7, 10.1403562793, 16, 34.1359086308, 79.466843214], rtol=2e-11)


def test_ida_signal_parameters_have_exact_fixed_dye_ambiguity():
    titrant = np.array([0.0, 1.0, 9.0, 30.0, 100.0])
    reference = ida_signal(7, 2, 2, 5, 0.5, 8, 3, titrant)
    shifted = ida_signal(4, 2, 3, 6, 0.5, 8, 3, titrant)
    np.testing.assert_allclose(shifted, reference, rtol=2e-12)
