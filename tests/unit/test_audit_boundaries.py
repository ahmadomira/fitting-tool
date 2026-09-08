"""Independent data, concentration, and parameter-unit invariants."""

import numpy as np
import pandas as pd
import pytest

from core.data_processing.measurement_set import MeasurementSet
from core.units import Q_


def test_distinct_nanomolar_replica_grids_are_rejected():
    # 1 nM and 5 nM are different analytical concentrations, not roundoff.
    df = pd.DataFrame(
        {'replica': ['a', 'a', 'b', 'b'], 'concentration': [0.0, 1e-9, 0.0, 5e-9], 'signal': [1, 2, 1, 3]}
    )
    with pytest.raises(ValueError, match='different concentration grid'):
        MeasurementSet.from_dataframe(df)


def test_calibration_bounds_preserve_per_micromolar_response():
    from PyQt6.QtWidgets import QApplication

    from core.assays import AssayType
    from gui.widgets.bounds_panel import BoundsPanel

    _app = QApplication.instance() or QApplication([])
    panel = BoundsPanel()
    panel.set_assay_type(AssayType.DBA_HtoD)
    # One au per micromolar is exactly one million au per molar.
    panel.apply_dye_alone_bounds({'I_dye_free': (Q_(1, 'au/uM'), Q_(2, 'au/uM'))})
    bounds = panel.current_bounds()['I_dye_free']
    np.testing.assert_allclose([q.to('au/M').magnitude for q in bounds], [1e6, 2e6], rtol=1e-14)
    panel.close()


@pytest.mark.parametrize(
    'x,y,reason',
    [
        ([0, -1e-9], [1, 2], 'nonnegative'),
        ([0, np.nan], [1, 2], 'finite'),
        ([0, np.inf], [1, 2], 'finite'),
        ([0, 1e-9], [1, np.nan], 'finite'),
        ([0, 1e-9], [1, np.inf], 'finite'),
        ([[0, 1e-9]], [[1, 2]], 'one-dimensional'),
    ],
)
def test_assay_rejects_invalid_observation_vectors(x, y, reason):
    from core.assays import DyeAloneAssay

    with pytest.raises(ValueError, match=reason):
        DyeAloneAssay(x_data=Q_(x, 'M'), y_data=Q_(y, 'au'))


def test_assay_accepts_finite_background_subtracted_signal():
    from core.assays import DyeAloneAssay

    assay = DyeAloneAssay(x_data=Q_([0, 1], 'uM'), y_data=Q_([-2, -1], 'au'))
    np.testing.assert_array_equal(assay.y_data.magnitude, [-2, -1])


@pytest.mark.parametrize('invalid', [np.nan, np.inf, [1.0]])
@pytest.mark.parametrize(
    'name,key,conditions',
    [
        ('DBAAssay', 'fixed_conc', {'fixed_conc': Q_(1, 'uM')}),
        ('GDAAssay', 'Ka_dye', {'Ka_dye': Q_(1e6, '1/M'), 'h0': Q_(1, 'uM'), 'g0': Q_(1, 'uM')}),
        ('GDAAssay', 'h0', {'Ka_dye': Q_(1e6, '1/M'), 'h0': Q_(1, 'uM'), 'g0': Q_(1, 'uM')}),
        ('GDAAssay', 'g0', {'Ka_dye': Q_(1e6, '1/M'), 'h0': Q_(1, 'uM'), 'g0': Q_(1, 'uM')}),
        ('IDAAssay', 'Ka_dye', {'Ka_dye': Q_(1e6, '1/M'), 'h0': Q_(1, 'uM'), 'd0': Q_(1, 'uM')}),
        ('IDAAssay', 'h0', {'Ka_dye': Q_(1e6, '1/M'), 'h0': Q_(1, 'uM'), 'd0': Q_(1, 'uM')}),
        ('IDAAssay', 'd0', {'Ka_dye': Q_(1e6, '1/M'), 'h0': Q_(1, 'uM'), 'd0': Q_(1, 'uM')}),
        ('HG2Assay', 'h0', {'h0': Q_(1, 'uM')}),
        ('H2GAssay', 'h0', {'h0': Q_(1, 'uM')}),
    ],
)
def test_fixed_conditions_require_finite_scalar_quantities(name, key, conditions, invalid):
    import core.assays as assays

    conditions = {**conditions, key: Q_(invalid, conditions[key].units)}
    with pytest.raises(ValueError, match='finite scalar'):
        getattr(assays, name)(x_data=Q_([0, 1e-6], 'M'), y_data=Q_([1, 2], 'au'), **conditions)


def test_simulation_condition_provenance_uses_canonical_units():
    from core.assays import IDAAssay
    from core.simulation import simulate_dataset

    dataset = simulate_dataset(
        IDAAssay,
        {'h0': Q_(2, 'uM'), 'd0': Q_(3, 'uM'), 'Ka_dye': Q_(0.5, '1/uM')},
        {'Ka_guest': 1e5, 'I0': 5, 'I_dye_free': 2e6, 'I_dye_bound': 8e6},
        np.array([0, 1e-6]),
    )
    assert dataset.metadata['simulation']['conditions'] == pytest.approx({'h0': 2e-6, 'd0': 3e-6, 'Ka_dye': 5e5})


@pytest.mark.parametrize('route', ['import', 'unit_change'])
def test_explicit_simulation_vector_preserves_concentration_precision(route):
    from PyQt6.QtWidgets import QApplication

    from gui.simulation.controls import TitrantInput

    _app = QApplication.instance() or QApplication([])
    vector = np.array([0.0, 1.2345678901234567e-6, 8.765432109876543e-6])
    control = TitrantInput()
    if route == 'import':
        control.set_explicit(vector)
    else:
        # User-supplied µM digits must survive changing the display to nM.
        control._vector.setText('0, 1.2345678901234567, 8.765432109876543')
        control._custom.setChecked(True)
        control._vector_unit.setCurrentText('nM')
    restored = TitrantInput()
    restored.load_state(control.state())
    _, values = restored.spec()
    np.testing.assert_allclose(values['values'], vector, rtol=3e-15, atol=0)
    control.close()
    restored.close()


@pytest.mark.parametrize('noise', [-0.1, np.nan, np.inf])
def test_simulation_rejects_invalid_noise_fraction(noise):
    from core.assays import DyeAloneAssay
    from core.simulation import simulate_dataset

    with pytest.raises(ValueError, match='noise_frac.*finite.*nonnegative'):
        simulate_dataset(DyeAloneAssay, {}, {'slope': 1, 'intercept': 0}, np.array([0, 1]), noise_frac=noise)


@pytest.mark.parametrize('replicas', [0, -1, 1.5, np.nan, np.inf, True])
def test_simulation_rejects_invalid_replica_count(replicas):
    from core.assays import DyeAloneAssay
    from core.simulation import simulate_dataset

    with pytest.raises(ValueError, match='n_replicas.*positive integer'):
        simulate_dataset(DyeAloneAssay, {}, {'slope': 1, 'intercept': 0}, np.array([0, 1]), n_replicas=replicas)
