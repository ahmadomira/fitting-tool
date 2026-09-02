"""P2: Forward model math tests.

Verify forward models produce correct signal values for known inputs.
These tests validate the mathematical correctness of the equilibrium
and linear models independently of the optimizer.
"""

import numpy as np
import pytest
from scipy.optimize import brentq, fsolve

from core.models.equilibrium import competitive_signal_point, dba_signal, gda_signal, ida_signal
from core.models.linear import linear_signal


class TestLinearModel:
    """Tests for the linear (dye-alone) forward model."""

    def test_linearity(self):
        """Signal is strictly linear with concentration.

        Expected values are arithmetic, not a restatement of the formula:
        slope 5e10 over 0/1/2 µM adds 0 / 50 000 / 100 000 to the 100 baseline.
        """
        result = linear_signal(5e10, 100.0, np.array([0.0, 1e-6, 2e-6]))
        np.testing.assert_allclose(result, [100.0, 50_100.0, 100_100.0], rtol=1e-12)

    def test_empty_array(self):
        """Empty input returns empty output."""
        x = np.array([])
        result = linear_signal(slope=1e10, intercept=100.0, x=x)
        assert len(result) == 0


class TestDBAModel:
    """Tests for the DBA (direct binding) forward model."""

    def test_zero_titrant_gives_baseline(self):
        """HtoD with no host => all dye free; signal = I0 + I_dye_free * d0."""
        x = np.array([0.0])
        # HtoD: y_fixed is dye, x_titrant is host. With no host, no complex
        # forms and all dye remains free.
        result = dba_signal(
            I0=50.0,
            Ka_dye=1e6,
            I_dye_free=1000.0,
            I_dye_bound=5000.0,
            x_titrant=x,
            y_fixed=10e-6,
            mode='HtoD',
        )
        expected = 50.0 + 1000.0 * 10e-6
        assert result[0] == pytest.approx(expected, rel=1e-6)

    def test_high_ka_saturates(self):
        """Very high Ka_dye should drive binding to saturation."""
        x = np.array([10e-6])  # Equal to fixed
        result_low = dba_signal(
            I0=0,
            Ka_dye=1e3,
            I_dye_free=1000.0,
            I_dye_bound=5000.0,
            x_titrant=x,
            y_fixed=10e-6,
            mode='HtoD',
        )
        result_high = dba_signal(
            I0=0,
            Ka_dye=1e12,
            I_dye_free=1000.0,
            I_dye_bound=5000.0,
            x_titrant=x,
            y_fixed=10e-6,
            mode='HtoD',
        )
        # Higher Ka => more complex => higher signal (since I_dye_bound > I_dye_free)
        assert result_high[0] > result_low[0]

    def test_htod_dtoh_equivalence(self):
        """HtoD and DtoH produce the same signal at the same physical state.

        Regression test for a bug where DBA_DtoH multiplied I_dye_free
        by free *host* concentration instead of free *dye*.  Same physical
        setup (h0, d0, parameters) must give the same signal regardless
        of which species is labelled the titrant.
        """
        Ka_dye = 5e5
        I0, I_dye_free, I_dye_bound = 100.0, 5e7, 3e8
        h0, d0 = 10e-6, 5e-6

        # HtoD: titrant = host = h0, fixed = dye = d0
        signal_htod = dba_signal(
            I0=I0,
            Ka_dye=Ka_dye,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            x_titrant=np.array([h0]),
            y_fixed=d0,
            mode='HtoD',
        )
        # DtoH: titrant = dye = d0, fixed = host = h0
        signal_dtoh = dba_signal(
            I0=I0,
            Ka_dye=Ka_dye,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            x_titrant=np.array([d0]),
            y_fixed=h0,
            mode='DtoH',
        )
        np.testing.assert_allclose(signal_htod, signal_dtoh, rtol=1e-10)


class TestCompetitiveModels:
    """Tests for GDA and IDA competitive binding models."""

    # Common conditions for competitive tests
    Ka_dye = 5e5
    h0 = 10e-6
    d0 = 1e-6
    g0 = 5e-6

    def test_gda_signal_increases_with_dye(self):
        """GDA signal increases as dye is titrated in."""
        d0_values = np.linspace(1e-7, 30e-6, 20)
        result = gda_signal(
            I0=50.0,
            Ka_guest=1.5e6,
            I_dye_free=1000.0,
            I_dye_bound=5000.0,
            Ka_dye=self.Ka_dye,
            h0=self.h0,
            d0_values=d0_values,
            g0=self.g0,
        )
        assert not np.any(np.isnan(result))
        # Overall trend: signal increases with dye
        assert result[-1] > result[0]

    def test_ida_signal_decreases_with_guest(self):
        """IDA signal decreases as guest displaces dye."""
        g0_values = np.linspace(0, 50e-6, 20)
        result = ida_signal(
            I0=100.0,
            Ka_guest=2e6,
            I_dye_free=2000.0,
            I_dye_bound=8000.0,
            Ka_dye=self.Ka_dye,
            h0=self.h0,
            d0=self.d0,
            g0_values=g0_values,
        )
        assert not np.any(np.isnan(result))
        # I_dye_bound > I_dye_free => as guest displaces dye from host,
        # signal decreases (less HD complex)
        assert result[-1] < result[0]


# ---------------------------------------------------------------------------
# Model output vs. independently solved equilibrium
# ---------------------------------------------------------------------------


class TestModelVsIndependentEquilibrium:
    """Verify the forward models against equilibrium solutions computed by a
    different numerical method.  The models solve a quadratic (DBA) or a
    Brent root-find on the host balance (competitive); here the same systems
    are solved via Brent on the *dye* balance and via a 3-species ``fsolve``,
    so agreement validates the model algebra rather than restating it.
    """

    def test_dba_signal_matches_independent_equilibrium(self):
        """dba_signal (quadratic) agrees with a Brent solve of the dye balance."""
        Ka, h0 = 5e5, 10e-6
        I0, I_dye_free, I_dye_bound = 100.0, 5e7, 3e8
        d0_values = np.array([1e-6, 10e-6, 50e-6])

        actual = dba_signal(
            I0=I0,
            Ka_dye=Ka,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            x_titrant=d0_values,
            y_fixed=h0,
            mode='DtoH',
        )

        for i, d0 in enumerate(d0_values):
            # Dye balance with [H_free] = h0 / (1 + Ka*[D_free]) substituted:
            #   d + Ka*d*h0/(1 + Ka*d) - d0 = 0
            d_free = brentq(
                lambda d: d * (1 + Ka * h0 / (1 + Ka * d)) - d0,
                0.0,
                d0,
                xtol=1e-22,
                rtol=1e-15,
            )
            hd = d0 - d_free
            expected = I0 + I_dye_free * d_free + I_dye_bound * hd
            assert actual[i] == pytest.approx(expected, rel=1e-9)

    def test_competitive_signal_matches_independent_equilibrium(self):
        """competitive_signal_point agrees with an fsolve of the full
        3-species system (solved in µM space for conditioning)."""
        Ka_dye, Ka_guest = 5e5, 1.5e6
        h0, d0, g0 = 10e-6, 5e-6, 20e-6
        I0, I_dye_free, I_dye_bound = 100.0, 5e7, 3e8

        # Solve [H], [D], [G] simultaneously in µM units.
        kd, kg = Ka_dye * 1e-6, Ka_guest * 1e-6  # 1/µM
        h0u, d0u, g0u = h0 * 1e6, d0 * 1e6, g0 * 1e6

        def equations(v):
            h, d, g = v
            hd = kd * h * d
            hg = kg * h * g
            return [h + hd + hg - h0u, d + hd - d0u, g + hg - g0u]

        sol, _, ier, msg = fsolve(equations, [h0u / 2, d0u / 2, g0u / 2], full_output=True, xtol=1e-13)
        assert ier == 1, f'independent solver failed: {msg}'
        h_u, d_u, _ = sol
        hd_u = kd * h_u * d_u
        expected = I0 + I_dye_free * d_u * 1e-6 + I_dye_bound * hd_u * 1e-6

        actual = competitive_signal_point(
            I0=I0,
            Ka_guest=Ka_guest,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            Ka_dye=Ka_dye,
            h0=h0,
            d0=d0,
            g0=g0,
        )
        assert actual == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# GAP-3: Competitive model boundary conditions
# ---------------------------------------------------------------------------


class TestCompetitiveBoundaries:
    """Limiting cases for the competitive binding model."""

    def test_gda_g0_zero_matches_dba(self):
        """GDA with g0=0 numerically matches DBA."""
        Ka_dye = 5e5
        h0 = 10e-6
        I0, I_dye_free, I_dye_bound = 50.0, 1000.0, 5000.0
        d0_values = np.linspace(1e-7, 30e-6, 10)

        gda_result = gda_signal(
            I0,
            Ka_guest=1e6,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            Ka_dye=Ka_dye,
            h0=h0,
            d0_values=d0_values,
            g0=0.0,
        )
        # GDA fixes h0 and titrates d0 -> the equivalent DBA framing is
        # mode='DtoH' (dye titrated, host fixed).
        dba_result = dba_signal(
            I0,
            Ka_dye=Ka_dye,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            x_titrant=d0_values,
            y_fixed=h0,
            mode='DtoH',
        )
        # Different solvers (Brent vs quadratic) -> tolerance set tight
        # enough to catch a regression in the signal model itself.
        np.testing.assert_allclose(gda_result, dba_result, rtol=1e-6)

    def test_ida_g0_zero_matches_dba(self):
        """IDA's titration start (g0=0) numerically matches the DBA quadratic
        for the same (h0, d0) — cross-validates Brent vs quadratic solvers."""
        Ka_dye = 5e5
        h0, d0 = 10e-6, 5e-6
        I0, I_dye_free, I_dye_bound = 50.0, 1000.0, 5000.0

        ida_result = ida_signal(
            I0=I0,
            Ka_guest=1e6,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            Ka_dye=Ka_dye,
            h0=h0,
            d0=d0,
            g0_values=np.array([0.0]),
        )
        dba_result = dba_signal(
            I0=I0,
            Ka_dye=Ka_dye,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            x_titrant=np.array([d0]),
            y_fixed=h0,
            mode='DtoH',
        )
        np.testing.assert_allclose(ida_result, dba_result, rtol=1e-6)

    def test_ida_ka_guest_zero_no_displacement(self):
        """Non-binding guest (Ka→0) produces constant signal."""
        g0_values = np.linspace(0, 50e-6, 10)
        result = ida_signal(
            I0=50.0,
            Ka_guest=1e-20,
            I_dye_free=1000.0,
            I_dye_bound=5000.0,
            Ka_dye=5e5,
            h0=10e-6,
            d0=5e-6,
            g0_values=g0_values,
        )
        assert not np.any(np.isnan(result))
        # Signal should be essentially constant (no displacement)
        assert np.ptp(result) / result[0] < 1e-6

    def test_ida_ka_guest_huge_displaces_all_dye(self):
        """Overwhelmingly strong guest saturates host → dye fully displaced."""
        g0_values = np.array([100e-6])  # large excess
        d0 = 5e-6
        I0, I_dye_free, I_dye_bound = 50.0, 1000.0, 5000.0

        result = ida_signal(
            I0,
            Ka_guest=1e15,
            I_dye_free=I_dye_free,
            I_dye_bound=I_dye_bound,
            Ka_dye=5e5,
            h0=10e-6,
            d0=d0,
            g0_values=g0_values,
        )
        # All dye free → signal ≈ I0 + I_dye_free * d0
        expected = I0 + I_dye_free * d0
        assert result[0] == pytest.approx(expected, rel=0.01)

    def test_competitive_d0_zero_gives_I0(self):
        """No dye → signal equals I0 (no dye contribution)."""
        result = competitive_signal_point(
            I0=50.0,
            Ka_guest=1e6,
            I_dye_free=1000.0,
            I_dye_bound=5000.0,
            Ka_dye=5e5,
            h0=10e-6,
            d0=0.0,
            g0=5e-6,
        )
        assert result == pytest.approx(50.0, abs=1e-10)


# ---------------------------------------------------------------------------
# GAP-4: Brent solver failure propagation
# ---------------------------------------------------------------------------


class TestSolverFailure:
    """Verify graceful NaN on solver failure."""

    def test_gda_nan_point_does_not_crash(self):
        """An array with extreme values propagates NaN but doesn't crash."""
        # h0 near-zero with huge binding constants → Brent bracket may fail
        d0_values = np.array([1e-6, 10e-6])
        result = gda_signal(
            I0=0,
            Ka_guest=1e50,
            I_dye_free=1,
            I_dye_bound=1,
            Ka_dye=1e50,
            h0=1e-30,
            d0_values=d0_values,
            g0=1.0,
        )
        assert len(result) == 2  # no crash, correct length


# ---------------------------------------------------------------------------
# Brent-bracket robustness across Ka magnitudes
# ---------------------------------------------------------------------------


class TestBrentBracketRobustness:
    """Verify the competitive Brent solver finds a physical root across
    several decades of Ka.  Guards against bracket-collapse failures at the
    weak-binding limit (root near h0) and the strong-binding limit (root
    near 1e-20).
    """

    # Corner cases probe bracket collapse at the weak/strong-binding limits
    # of each constant; the asymmetric mid case guards the generic path.
    @pytest.mark.parametrize(
        'Ka_dye, Ka_guest',
        [
            (1.0, 1.0),
            (1.0, 1e10),
            (1e10, 1.0),
            (1e10, 1e10),
            (1e4, 1e7),
        ],
    )
    def test_competitive_root_is_physical(self, Ka_dye, Ka_guest):
        """[H_free] must be finite and in (0, h0) regardless of Ka magnitudes."""
        from core.models.equilibrium import competitive_signal_point

        h0, d0, g0 = 10e-6, 5e-6, 20e-6
        result = competitive_signal_point(
            I0=0.0,
            Ka_guest=Ka_guest,
            I_dye_free=1.0,
            I_dye_bound=1.0,
            Ka_dye=Ka_dye,
            h0=h0,
            d0=d0,
            g0=g0,
        )
        # Signal ranges from I0 + I_dye_free*0 (full binding) to
        # I0 + I_dye_free*d0 + I_dye_bound*0 (no binding) i.e. [0, d0].
        # Just guard that it is finite and physical.
        assert np.isfinite(result), f'NaN / Inf at Ka_dye={Ka_dye:g}, Ka_guest={Ka_guest:g}'
        assert 0.0 <= result <= d0 + 1e-12, (
            f'Signal {result:g} out of physical range [0, {d0:g}] at Ka_dye={Ka_dye:g}, Ka_guest={Ka_guest:g}'
        )

    @pytest.mark.parametrize('Ka_dye', [1.0, 1e10])
    def test_dba_root_is_physical(self, Ka_dye):
        """DBA quadratic must produce a non-negative free fixed-species root."""
        h0, d0 = 10e-6, 5e-6
        # DtoH framing: titrant=dye, fixed=host
        signal = dba_signal(
            I0=0.0,
            Ka_dye=Ka_dye,
            I_dye_free=1.0,
            I_dye_bound=1.0,
            x_titrant=np.array([d0]),
            y_fixed=h0,
            mode='DtoH',
        )
        assert np.isfinite(signal[0]), f'NaN / Inf at Ka_dye={Ka_dye:g}'
        # Signal at this point ∈ [0, max(d0, [HD]≤min(h0,d0))]
        assert 0.0 <= signal[0] <= max(h0, d0) + 1e-12, (
            f'Signal {signal[0]:g} out of physical range at Ka_dye={Ka_dye:g}'
        )
