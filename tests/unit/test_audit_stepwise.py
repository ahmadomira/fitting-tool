"""Stepwise mass-action checks using independently specified equilibrium species."""

import numpy as np
import pytest

from core.models.equilibrium import h2g_signal, h2g_species, hg2_signal, hg2_species


@pytest.mark.parametrize(
    'species,ternary,host,guest,expected',
    [
        (hg2_species, 'HG2', 1e-4, 1.8e-4, (4e-5, 1e-4, 4e-5, 2e-5)),
        (h2g_species, 'H2G', 1e-4, 13 / 120000, (5e-5, 1 / 15000, 1 / 30000, 1 / 120000)),
    ],
)
@pytest.mark.parametrize('scale', [1.0, 1e-8, 1e-14, 1e8])
def test_manufactured_stepwise_species_are_concentration_scale_invariant(
    species, ternary, host, guest, expected, scale
):
    """Changing concentration units/scales cannot change occupancy fractions."""
    sp = species(1e4 / scale, 5e3 / scale, host * scale, np.array([guest * scale]))
    actual = np.array([sp[key][0] for key in ('H', 'G', 'HG', ternary)]) / scale
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize(
    'species,ternary,host,guest',
    [
        (hg2_species, 'HG2', 3e-200, 4e-200),
        (h2g_species, 'H2G', 4e-200, 3e-200),
    ],
)
def test_finite_dimensionless_occupancies_do_not_overflow_cumulative_constant(species, ternary, host, guest):
    # Constructed with h=g=HG=ternary=1e-200 M and K1=K2=1e200 M^-1.
    with np.errstate(over='raise', invalid='raise'):
        sp = species(1e200, 1e200, host, np.array([guest]))
    np.testing.assert_allclose([sp[key][0] / 1e-200 for key in ('H', 'G', 'HG', ternary)], 1, rtol=2e-12)


@pytest.mark.parametrize('species', [hg2_species, h2g_species])
@pytest.mark.parametrize('bad', [-1.0, np.nan, np.inf])
@pytest.mark.parametrize('field', ['K1', 'K2', 'host', 'guest'])
def test_nonphysical_stepwise_inputs_return_nan(species, bad, field):
    args = dict(K1=1e4, K2=5e3, host=1e-4, guest=1e-4)
    args[field] = bad
    sp = species(args['K1'], args['K2'], args['host'], np.array([args['guest']]))
    assert all(np.isnan(value).all() for value in sp.values())


@pytest.mark.parametrize('species,ternary', [(hg2_species, 'HG2'), (h2g_species, 'H2G')])
@pytest.mark.parametrize('K1,K2,host,guest', [(1e4, 5e3, 0, 1e-4), (1e4, 5e3, 1e-4, 0), (0, 5e3, 1e-4, 2e-4)])
def test_zero_component_or_first_constant_leaves_free_components(species, ternary, K1, K2, host, guest):
    sp = species(K1, K2, host, np.array([guest]))
    np.testing.assert_allclose([sp[key][0] for key in ('H', 'G', 'HG', ternary)], [host, guest, 0, 0], atol=0)


@pytest.mark.parametrize('species,ternary', [(hg2_species, 'HG2'), (h2g_species, 'H2G')])
def test_zero_second_constant_matches_manufactured_one_to_one_case(species, ternary):
    sp = species(1e4, 0, 8e-5, np.array([1.4e-4]))
    np.testing.assert_allclose(
        [sp[key][0] for key in ('H', 'G', 'HG', ternary)], [4e-5, 1e-4, 4e-5, 0], rtol=2e-12, atol=0
    )


@pytest.mark.parametrize('species,ternary,n_host,n_guest', [(hg2_species, 'HG2', 1, 2), (h2g_species, 'H2G', 2, 1)])
def test_high_affinity_conservation_and_mass_action(species, ternary, n_host, n_guest):
    host = 1e-4
    guest = np.array([1e-15, 1e-9, 5e-5, 1e-4, 2e-4, 1e-2])
    sp = species(1e12, 1e12, host, guest)
    assert all(np.all(value >= 0) for value in sp.values())
    np.testing.assert_allclose(sp['H'] + sp['HG'] + n_host * sp[ternary], host, rtol=2e-11, atol=0)
    np.testing.assert_allclose(sp['G'] + sp['HG'] + n_guest * sp[ternary], guest, rtol=2e-11, atol=0)
    np.testing.assert_allclose(sp['HG'], 1e12 * sp['H'] * sp['G'], rtol=2e-11, atol=0)
    partner = sp['G'] if n_guest == 2 else sp['H']
    np.testing.assert_allclose(sp[ternary], 1e12 * sp['HG'] * partner, rtol=2e-11, atol=0)


@pytest.mark.parametrize(
    'signal,guest,expected,n_host', [(hg2_signal, 1.8e-4, 0.374, 1), (h2g_signal, 13 / 120000, 49 / 150, 2)]
)
def test_signal_uses_per_complex_coefficients_and_preserves_host_gauge(signal, guest, expected, n_host):
    values = signal(0.25, 1e4, 5e3, 200, 100, 1000, 3000, 1e-4, np.array([guest]))
    shifted = signal(0.249, 1e4, 5e3, 200, 110, 1010, 3000 + n_host * 10, 1e-4, np.array([guest]))
    np.testing.assert_allclose(values, expected, rtol=2e-12)
    np.testing.assert_allclose(shifted, values, rtol=2e-12)


def test_excess_guest_distinguishes_two_titration_orientations():
    hg2 = hg2_species(1e4, 5e3, 1e-4, np.array([1e4]))
    h2g = h2g_species(1e4, 5e3, 1e-4, np.array([1e4]))
    assert hg2['HG2'][0] / 1e-4 > 0.99999
    assert h2g['HG'][0] / 1e-4 > 0.99999
    assert h2g['H2G'][0] / 1e-4 < 1e-7


@pytest.mark.parametrize('species,signal,ternary', [(hg2_species, hg2_signal, 'HG2'), (h2g_species, h2g_signal, 'H2G')])
@pytest.mark.parametrize('exponent', [100, 200])
def test_underflowing_occupancy_retains_representable_species(species, signal, ternary, exponent):
    # Independent free-species construction, rounded to finite float totals:
    # core=10**p, ligand=10**-p, CL=1e-300, CL2=10**-p.
    # An occupancy may underflow even though its concentration is representable.
    core = 10.0**exponent
    ligand = 10.0**-exponent
    host, guest = (core, 3 * ligand) if ternary == 'HG2' else (3 * ligand, core)
    expected_h, expected_g = (core, ligand) if ternary == 'HG2' else (ligand, core)
    sp = species(1e-300, 1e300, host, np.array([guest]))
    np.testing.assert_allclose(sp['H'] / expected_h, 1, rtol=2e-12)
    np.testing.assert_allclose(sp['G'] / expected_g, 1, rtol=2e-12)
    np.testing.assert_allclose(sp['HG'] / 1e-300, 1, rtol=2e-12)
    np.testing.assert_allclose(sp[ternary] / ligand, 1, rtol=2e-12)
    bound = sp['HG'] + 2 * sp[ternary]
    free = sp['G'] if ternary == 'HG2' else sp['H']
    np.testing.assert_allclose((free + bound) / (3 * ligand), 1, rtol=2e-12)
    actual_signal = signal(0, 1e-300, 1e300, 0, 0, 0, core, host, np.array([guest]))
    np.testing.assert_allclose(actual_signal, 1, rtol=2e-12)
