"""Unit tests for the ensemble-collapse module.

Pins the single-source collapse behaviour: representative selection and the
per-parameter summary every reported number is derived from.
"""

import numpy as np
import pytest

from core.optimizer import ensemble

# ---------------------------------------------------------------------------
# Representative selection
# ---------------------------------------------------------------------------


class TestSelectRepresentativeIndex:
    def test_picks_highest_r_squared(self):
        quality = {'rmse': np.array([0.03, 0.01, 0.02]), 'r_squared': np.array([0.98, 0.999, 0.99])}
        assert ensemble.select_representative_index(quality) == 1

    def test_highest_r_squared_is_lowest_rmse(self):
        """On a fixed dataset the two criteria agree by construction."""
        quality = {'rmse': np.array([0.5, 0.1, 0.3]), 'r_squared': np.array([0.90, 0.999, 0.95])}
        idx = ensemble.select_representative_index(quality)
        assert idx == int(np.argmin(quality['rmse']))

    def test_r_squared_ties_broken_by_lowest_rmse(self):
        """When R² ties (e.g. all 0 for constant y, ss_tot==0), pick lowest RMSE."""
        quality = {'rmse': np.array([0.9, 0.4, 0.7]), 'r_squared': np.array([0.0, 0.0, 0.0])}
        assert ensemble.select_representative_index(quality) == 1


# ---------------------------------------------------------------------------
# collapse
# ---------------------------------------------------------------------------


class TestCollapse:
    def test_pools_and_aligns_by_index(self):
        pm = np.array([[10.0, 100.0], [20.0, 50.0], [30.0, 75.0]])
        rmse = np.array([0.02, 0.01, 0.03])
        r2 = np.array([0.99, 0.999, 0.98])
        res = ensemble.collapse(pm, rmse, r2, ['Ka', 'I0'])

        np.testing.assert_array_equal(res.parameter_samples['Ka'], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(res.parameter_samples['I0'], [100.0, 50.0, 75.0])
        np.testing.assert_array_equal(res.quality_samples['rmse'], rmse)
        np.testing.assert_array_equal(res.quality_samples['r_squared'], r2)

    def test_representative_is_the_best_real_row(self):
        pm = np.array([[10.0, 100.0], [20.0, 50.0], [30.0, 75.0]])
        rmse = np.array([0.02, 0.01, 0.03])
        r2 = np.array([0.99, 0.999, 0.98])
        res = ensemble.collapse(pm, rmse, r2, ['Ka', 'I0'])

        assert res.representative_index == 1
        # Representative is an actual row of the pool, not a per-column aggregate.
        np.testing.assert_array_equal(res.representative_params, [20.0, 50.0])


# ---------------------------------------------------------------------------
# describe / describe_log10 (Ka-space vs log-space statistics)
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_range_matches_numpy(self):
        s = np.array([3.0, 1.0, 2.0, 5.0])
        d = ensemble.describe(s)
        assert d['min'] == np.min(s)
        assert d['max'] == np.max(s)

    def test_centres_and_spreads_are_hand_computable(self):
        # [1, 2, 3]: median=2, |dev|=[1,0,1] → MAD=1; mean=2, sample SD (ddof=1)=1.
        d = ensemble.describe(np.array([1.0, 2.0, 3.0]))
        assert d['median'] == pytest.approx(2.0)
        assert d['mad'] == pytest.approx(1.0)
        assert d['mean'] == pytest.approx(2.0)
        assert d['std'] == pytest.approx(1.0)

    def test_single_sample_has_zero_spread(self):
        """One sample → no dispersion defined; report 0, never NaN."""
        d = ensemble.describe(np.array([7.0]))
        assert d['median'] == pytest.approx(7.0)
        assert d['mean'] == pytest.approx(7.0)
        assert d['mad'] == 0.0
        assert d['std'] == 0.0
        assert d['min'] == d['max'] == pytest.approx(7.0)

    def test_p16_p84_are_the_16th_84th_percentiles(self):
        s = np.array([3.0, 1.0, 2.0, 5.0, 8.0, 4.0])
        d = ensemble.describe(s)
        p16, p84 = np.percentile(s, [16, 84])
        assert d['p16'] == pytest.approx(p16)
        assert d['p84'] == pytest.approx(p84)
        assert d['p16'] <= d['median'] <= d['p84']

    def test_describe_log10_transforms_first_not_log_of_spread(self):
        """log₁₀ stats must come from log₁₀(pool). The centre commutes (median),
        but the spread does NOT — log of a Ka MAD/std is wrong."""
        s = np.array([10.0, 100.0, 1000.0, 50.0, 500.0])  # odd length: exact median
        d = ensemble.describe(s)
        dl = ensemble.describe_log10(s)

        # transform-first == summarising the log-transformed pool (independent path)
        expected = ensemble.describe(np.log10(s))
        for k in dl:
            assert dl[k] == pytest.approx(expected[k])

        # median commutes with log; MAD and mean do not
        assert dl['median'] == pytest.approx(np.log10(d['median']))
        assert dl['mad'] != pytest.approx(np.log10(d['mad']))
        assert dl['mean'] != pytest.approx(np.log10(d['mean']))  # Jensen bias

    def test_describe_log10_rejects_non_positive(self):
        with pytest.raises(ValueError, match='positive'):
            ensemble.describe_log10(np.array([1.0, -2.0, 3.0]))
        with pytest.raises(ValueError):
            ensemble.describe_log10(np.array([]))
