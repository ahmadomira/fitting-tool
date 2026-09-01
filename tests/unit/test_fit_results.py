"""Tests for FitResult serialization and traceability."""

import json

import numpy as np
import pytest

from core.pipeline.fit_pipeline import FitResult
from core.units import Q_, Quantity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_fit_result(**overrides) -> FitResult:
    """Build a minimal FitResult for testing."""
    defaults = dict(
        parameters={
            'Ka_guest': Q_(1.5e6, '1/M'),
            'I0': Q_(0.0, 'au'),
            'I_dye_free': Q_(5e7, 'au/M'),
            'I_dye_bound': Q_(3e8, 'au/M'),
        },
        rmse=42.0,
        r_squared=0.998,
        n_passing=80,
        n_total=100,
        x_fit=Q_(np.linspace(0, 50e-6, 30), 'M'),
        y_fit=Q_(np.linspace(100, 3000, 30), 'au'),
        assay_type='GDA',
        model_name='equilibrium_4param',
        conditions={'Ka_dye': Q_(5e5, '1/M'), 'h0': Q_(10e-6, 'M'), 'g0': Q_(20e-6, 'M')},
        fit_config={'n_trials': 100, 'rmse_threshold_factor': 1.5},
        measurement_set_id='abc123',
        source_file='data/GDA_system.txt',
        parameter_samples={
            'Ka_guest': np.array([1.4e6, 1.5e6, 1.6e6]),
            'I0': np.array([-1.0, 0.0, 1.0]),
            'I_dye_free': np.array([4.9e7, 5.0e7, 5.1e7]),
            'I_dye_bound': np.array([2.9e8, 3.0e8, 3.1e8]),
        },
        quality_samples={
            'rmse': np.array([45.0, 42.0, 48.0]),
            'r_squared': np.array([0.997, 0.998, 0.996]),
        },
        representative_index=1,
    )
    defaults.update(overrides)
    return FitResult(**defaults)


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


class TestFitResultProperties:
    """Core properties and defaults."""

    def test_unique_ids(self):
        r1 = _sample_fit_result()
        r2 = _sample_fit_result()
        assert r1.id != r2.id


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    """to_dict / from_dict round-trip."""

    def test_to_dict_json_safe(self):
        """All values must be JSON-serializable."""
        d = _sample_fit_result().to_dict()
        json.dumps(d)  # must not raise

    def test_round_trip(self):
        """from_dict(to_dict(r)) reproduces r faithfully."""
        original = _sample_fit_result()
        d = original.to_dict()
        restored = FitResult.from_dict(d)

        # Parameters round-trip as Quantities
        for k in original.parameters:
            assert restored.parameters[k].magnitude == pytest.approx(original.parameters[k].magnitude)
        assert restored.rmse == original.rmse
        assert restored.r_squared == original.r_squared
        assert restored.n_passing == original.n_passing
        assert restored.n_total == original.n_total
        assert restored.assay_type == original.assay_type
        assert restored.model_name == original.model_name
        # Conditions round-trip as Quantities with their units preserved.
        for k, v in original.conditions.items():
            rv = restored.conditions[k]
            assert isinstance(rv, Quantity), f'condition {k!r} lost its Quantity type'
            assert rv.magnitude == pytest.approx(v.magnitude)
            assert rv.units == v.units
        assert restored.fit_config == original.fit_config
        assert restored.measurement_set_id == original.measurement_set_id
        assert restored.source_file == original.source_file
        assert restored.id == original.id
        assert restored.timestamp == original.timestamp
        np.testing.assert_array_almost_equal(restored.x_fit.magnitude, original.x_fit.magnitude)
        np.testing.assert_array_almost_equal(restored.y_fit.magnitude, original.y_fit.magnitude)
        # Ensemble fields round-trip too
        for k in original.parameter_samples:
            np.testing.assert_array_almost_equal(restored.parameter_samples[k], original.parameter_samples[k])
        for k in original.quality_samples:
            np.testing.assert_array_almost_equal(restored.quality_samples[k], original.quality_samples[k])
        assert restored.representative_index == original.representative_index

    def test_from_dict_missing_optional_fields(self):
        """from_dict handles missing optional keys gracefully."""
        minimal = {
            'parameters': {'slope': 1.0},
            'rmse': 0.5,
            'r_squared': 0.99,
            'n_passing': 1,
            'n_total': 1,
            'x_fit': [0, 1, 2],
            'y_fit': [0, 1, 2],
            'assay_type': 'DYE_ALONE',
            'model_name': 'linear',
        }
        r = FitResult.from_dict(minimal)
        assert r.conditions == {}
        assert r.measurement_set_id is None
        assert r.source_file is None
        assert isinstance(r.id, str)

    def test_x_fit_y_fit_are_ndarray_after_from_dict(self):
        d = _sample_fit_result().to_dict()
        r = FitResult.from_dict(d)
        assert isinstance(r.x_fit, Quantity)
        assert isinstance(r.y_fit, Quantity)

    def test_x_y_fit_units_round_trip(self):
        """x_fit/y_fit carry their own unit tokens; older files without them fall
        back to the M/au convention (L10)."""
        r = _sample_fit_result()
        restored = FitResult.from_dict(r.to_dict())
        assert restored.x_fit.units == r.x_fit.units
        assert restored.y_fit.units == r.y_fit.units

        legacy = r.to_dict()
        legacy.pop('x_fit_unit')
        legacy.pop('y_fit_unit')
        loaded = FitResult.from_dict(legacy)
        assert loaded.x_fit.units == Q_(1, 'M').units
        assert loaded.y_fit.units == Q_(1, 'au').units

    def test_unknown_assay_round_trips_via_quantity_units(self):
        """to_dict emits unit tokens from the parameter Quantities (not a registry
        lookup), so a result whose assay_type is unknown to the registry still
        round-trips with units — the write side is self-describing."""
        r = _sample_fit_result(assay_type='LEGACY_UNKNOWN')
        d = r.to_dict()
        assert d['parameter_units']['Ka_guest'] == str(Q_(1, '1/M').units)
        restored = FitResult.from_dict(d)
        assert restored.parameters['Ka_guest'].units == Q_(1, '1/M').units
        assert restored.parameters['I_dye_free'].units == Q_(1, 'au/M').units

    def test_config_custom_bounds_keep_unit_token(self):
        """Serialized custom_bounds provenance keeps its unit token so a bound in
        1/M isn't stored as a unit-ambiguous bare number (L9)."""
        from core.pipeline.fit_pipeline import FitConfig, _config_to_dict

        cfg = FitConfig(custom_bounds={'Ka_guest': (Q_(1e5, '1/M'), Q_(1e9, '1/M'))})
        lo, hi, unit = _config_to_dict(cfg)['custom_bounds']['Ka_guest']
        assert (lo, hi) == (1e5, 1e9)
        assert unit == str(Q_(1, '1/M').units)


class TestEnsembleMutators:
    """Fail-fast contracts on the public pipeline mutation helpers."""

    def test_select_representative_rejects_out_of_range_index(self):
        from core.pipeline.fit_pipeline import select_representative

        r = _sample_fit_result()  # pool size 3
        with pytest.raises(ValueError, match='out of range'):
            select_representative(r, None, 99)  # index validated before the assay is used

    def test_select_representative_rejects_empty_pool(self):
        from dataclasses import replace

        from core.pipeline.fit_pipeline import select_representative

        r = replace(_sample_fit_result(), parameter_samples={})
        with pytest.raises(ValueError, match='no parameter_samples'):
            select_representative(r, None, 0)  # empty dict -> clear error, not StopIteration


class TestFromDictNormalisation:
    """Malformed/legacy imports are sanitised so they can't crash the GUI later."""

    def test_out_of_range_representative_index_dropped(self):
        d = _sample_fit_result().to_dict()  # pool size 3
        d['representative_index'] = 999
        assert FitResult.from_dict(d).representative_index is None

    def test_from_dict_raises_on_missing_units_unknown_assay(self):
        """A legacy dict with no parameter_units AND an unknown assay type must
        fail loud, not silently load a real unit (1/M) as dimensionless (L1)."""
        d = _sample_fit_result().to_dict()
        d.pop('parameter_units', None)
        d['assay_type'] = 'LEGACY_UNKNOWN'  # not resolvable via the registry
        with pytest.raises(ValueError, match='no unit for parameter'):
            FitResult.from_dict(d)
