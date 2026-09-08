"""Direct-binding checks using high-precision roots, exact species, and physical limits."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from core.assays.dba import DBAAssay
from core.assays.dye_alone import DyeAloneAssay
from core.models.equilibrium import dba_signal, dba_species
from core.models.linear import linear_signal
from core.units import Q_


def _decimal_species(ka, h_total, d_total):
    """High-precision complex-root oracle; independent of free-species solve."""
    with localcontext() as ctx:
        ctx.prec = 180
        k, h, d = map(lambda v: Decimal(str(v)), (ka, h_total, d_total))
        if not k or not h or not d:
            return np.array([float(h), float(d), 0.0])
        s = h + d + 1 / k
        c = 2 * h * d / (s + (s * s - 4 * h * d).sqrt())
        return np.array([float(h - c), float(d - c), float(c)])


@pytest.mark.parametrize('mode', ['HtoD', 'DtoH'])
@pytest.mark.parametrize(
    'ka,x,fixed',
    [
        (0.0, 2e-6, 3e-6),
        (1e-20, 2e-6, 3e-6),
        (1e12, 1e-9, 1e-3),
        (1e20, 2e-6, 3e-6),
        (1e20, 3e-6, 2e-6),
        (1e120, 2e-6, 3e-6),
        (1e120, 2e-6, 2e-6),
        (5e5, 2e-6, 3e-6),
        (5e5, 0.0, 3e-6),
        (5e5, 2e-6, 0.0),
    ],
)
def test_direct_species_against_high_precision_mass_balance(ka, x, fixed, mode):
    h_total, d_total = (x, fixed) if mode == 'HtoD' else (fixed, x)
    expected = _decimal_species(ka, h_total, d_total)
    with np.errstate(all='raise'):
        species = dba_species(ka, np.array([x]), fixed, mode=mode)
    actual = np.array([species[name][0] for name in ('H', 'D', 'HD')])
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
    assert np.all(actual >= 0)
    assert actual[0] + actual[2] == pytest.approx(h_total, rel=2e-15, abs=0)
    assert actual[1] + actual[2] == pytest.approx(d_total, rel=2e-15, abs=0)


@pytest.mark.parametrize(
    'ka,concentration,free',
    [
        (1e300, 1e300, 1.0),
        (1e300, 1e-300, (3 - np.sqrt(5)) / 2),
    ],
)
def test_direct_extreme_finite_scales_remain_physical(ka, concentration, free):
    species = dba_species(ka, np.array([concentration]), concentration, mode='HtoD')
    assert all(np.all(np.isfinite(v)) and np.all(v >= 0) for v in species.values())
    if concentration == 1e300:
        assert species['H'][0] == pytest.approx(free, rel=1e-14)
    else:
        # K*C=1 gives bound fraction (3-sqrt(5))/2.
        assert species['HD'][0] / concentration == pytest.approx(free, rel=1e-14)


@pytest.mark.parametrize('ka,fixed', [(-1.0, 1e-6), (np.nan, 1e-6), (np.inf, 1e-6), (1e6, -1e-6), (1e6, np.inf)])
def test_direct_invalid_global_domain_returns_nan(ka, fixed):
    species = dba_species(ka, np.array([0.0, 1e-6]), fixed, mode='DtoH')
    assert all(np.all(np.isnan(v)) for v in species.values())


def test_direct_invalid_points_do_not_poison_valid_points():
    species = dba_species(5e5, np.array([-1e-6, np.nan, np.inf, 2e-6]), 3e-6, mode='HtoD')
    for values in species.values():
        assert np.all(np.isnan(values[:3]))
        assert np.isfinite(values[3])


@pytest.mark.parametrize('mode', ['HtoD', 'DtoH'])
def test_direct_hand_signal_and_unit_conversion(mode):
    # h=1 uM,d=2 uM,c=1 uM satisfies Ka=0.5/uM and totals2/3 uM.
    titrant, fixed = (2.0, 3.0) if mode == 'HtoD' else (3.0, 2.0)
    assay = DBAAssay(x_data=Q_([titrant], 'uM'), y_data=Q_([17.0], 'au'), fixed_conc=Q_(fixed, 'uM'), mode=mode)
    assert assay.forward_model(np.array([5e5, 5.0, 2e6, 8e6])).magnitude[0] == pytest.approx(17.0)
    # Equal species brightness removes all binding information.
    expected = 5.0 + 2e6 * 3e-6
    assert assay.forward_model(np.array([5e5, 5.0, 2e6, 2e6])).magnitude[0] == pytest.approx(expected)


def test_direct_quenching_and_dye_excess_tail():
    x = np.array([0.0, 2e-6, 2e-3])
    quenched = dba_signal(5.0, 5e5, 8e6, 2e6, x, 3e-6, mode='HtoD')
    assert np.all(np.diff(quenched) < 0)
    x = np.array([2e-3, 4e-3])
    signal = dba_signal(5.0, 5e5, 2e6, 8e6, x, 2e-6, mode='DtoH')
    assert np.diff(signal)[0] / np.diff(x)[0] == pytest.approx(2e6, rel=1e-5)


@pytest.mark.parametrize('scale', [1e-200, 1.0, 1e200])
@pytest.mark.parametrize('mode', ['HtoD', 'DtoH'])
def test_direct_dimensional_rescaling_of_exact_case(scale, mode):
    # H=1,D=2,HD=1 uM remains the solution when all totals are scaled
    # and association/response coefficients are inversely scaled.
    x, fixed = (2e-6, 3e-6) if mode == 'HtoD' else (3e-6, 2e-6)
    sp = dba_species(5e5 / scale, np.array([x * scale]), fixed * scale, mode=mode)
    np.testing.assert_allclose(
        [sp[name][0] / scale for name in ('H', 'D', 'HD')], [1e-6, 2e-6, 1e-6], rtol=2e-14, atol=0
    )
    signal = dba_signal(5.0, 5e5 / scale, 2e6 / scale, 8e6 / scale, np.array([x * scale]), fixed * scale, mode=mode)
    assert signal[0] == pytest.approx(17.0, rel=2e-14)


def test_dye_calibration_hand_case_and_override():
    np.testing.assert_allclose(linear_signal(2e6, 5.0, np.array([0.0, 1e-6, 3e-6])), [5.0, 7.0, 11.0])
    assay = DyeAloneAssay(x_data=Q_([0.0, 1.0], 'uM'), y_data=Q_([5.0, 7.0], 'au'))
    np.testing.assert_allclose(assay.forward_model(np.array([2e6, 5.0]), x=np.array([3e-6])).magnitude, [11.0])
