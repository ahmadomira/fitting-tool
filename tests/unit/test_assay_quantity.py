"""Tests for assay constructors accepting pint Quantity conditions."""

import numpy as np
import pint
import pytest

from core.assays.dba import DBAAssay
from core.assays.gda import GDAAssay
from core.assays.ida import IDAAssay
from core.units import Q_

# Shared dummy data
_x = Q_(np.linspace(1e-7, 50e-6, 10), 'M')
_y = Q_(np.ones(10), 'au')


class TestGDAQuantityConditions:
    """GDA assay accepts Quantity conditions with dimensional validation."""

    def test_quantity_conditions_accepted(self):
        assay = GDAAssay(
            x_data=_x,
            y_data=_y,
            Ka_dye=Q_(5e5, '1/M'),
            h0=Q_(10, 'µM'),
            g0=Q_(20, 'µM'),
        )
        assert assay.Ka_dye.magnitude == pytest.approx(5e5)
        assert assay.h0.magnitude == pytest.approx(10e-6)
        assert assay.g0.magnitude == pytest.approx(20e-6)

    def test_bare_float_conditions_rejected(self):
        with pytest.raises(TypeError, match='Ka_dye must be a pint Quantity'):
            GDAAssay(
                x_data=_x,
                y_data=_y,
                Ka_dye=5e5,
                h0=10e-6,
                g0=20e-6,
            )

    def test_wrong_dimensionality_Ka_raises(self):
        with pytest.raises(pint.DimensionalityError):
            GDAAssay(
                x_data=_x,
                y_data=_y,
                Ka_dye=Q_(10, 'µM'),  # concentration, not 1/concentration
                h0=Q_(10e-6, 'M'),
                g0=Q_(20e-6, 'M'),
            )

    def test_wrong_dimensionality_h0_raises(self):
        with pytest.raises(pint.DimensionalityError):
            GDAAssay(
                x_data=_x,
                y_data=_y,
                Ka_dye=Q_(5e5, '1/M'),
                h0=Q_(10, '1/M'),  # 1/M is not a concentration
                g0=Q_(20e-6, 'M'),
            )


class TestIDAQuantityConditions:
    """IDA assay accepts Quantity conditions with dimensional validation."""

    def test_quantity_conditions_accepted(self):
        assay = IDAAssay(
            x_data=_x,
            y_data=_y,
            Ka_dye=Q_(5e5, 'M^-1'),
            h0=Q_(10, 'µM'),
            d0=Q_(5, 'µM'),
        )
        assert assay.Ka_dye.magnitude == pytest.approx(5e5)
        assert assay.h0.magnitude == pytest.approx(10e-6)
        assert assay.d0.magnitude == pytest.approx(5e-6)

    def test_bare_float_conditions_rejected(self):
        with pytest.raises(TypeError, match='Ka_dye must be a pint Quantity'):
            IDAAssay(x_data=_x, y_data=_y, Ka_dye=5e5, h0=10e-6, d0=1e-6)

    def test_wrong_dimensionality_d0_raises(self):
        with pytest.raises(pint.DimensionalityError):
            IDAAssay(
                x_data=_x,
                y_data=_y,
                Ka_dye=Q_(5e5, '1/M'),
                h0=Q_(10e-6, 'M'),
                d0=Q_(5, '1/M'),  # wrong: should be concentration
            )


class TestDBAQuantityConditions:
    """DBA assay accepts Quantity fixed_conc with dimensional validation."""

    def test_quantity_fixed_conc_accepted(self):
        assay = DBAAssay(
            x_data=_x,
            y_data=_y,
            fixed_conc=Q_(10, 'µM'),
            mode='DtoH',
        )
        assert assay.fixed_conc.magnitude == pytest.approx(10e-6)

    def test_bare_float_fixed_conc_rejected(self):
        with pytest.raises(TypeError, match='fixed_conc must be a pint Quantity'):
            DBAAssay(x_data=_x, y_data=_y, fixed_conc=10e-6, mode='DtoH')

    def test_wrong_dimensionality_fixed_conc_raises(self):
        with pytest.raises(pint.DimensionalityError):
            DBAAssay(
                x_data=_x,
                y_data=_y,
                fixed_conc=Q_(10, '1/M'),  # wrong: should be concentration
                mode='DtoH',
            )
