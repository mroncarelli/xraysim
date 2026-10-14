import numpy as np
import pytest
from pyxhydro.specutils import absorption as spabs


def test_table_energy_ascending():
    """
    The energy column in the energy table must be in ascending order
    """
    for index in range(1, len(spabs.energyArr)):
        assert spabs.energyArr[index] > spabs.energyArr[index - 1]


def test_table_sigma_abs_gt_0():
    """
    Values of sigma_abs in the table must be strictly positive
    """
    assert (spabs.sigmaAbsTable > 0.).all()


def test_abundance_tables_names_must_be_valid():
    """
    The abundance tables' names must be valid Xspec table names
    """
    for abund in spabs.abundArr:
        assert abund in ['angr', 'aspl', 'feld', 'aneb', 'grsa', 'wilm', 'lodd', 'lpgp', 'lpgs', 'felc']


def test_table_values():
    """
    Values for input energy present in the table must be the corresponding table values
    """
    for abund in spabs.abundArr:
        table_values = spabs.sigmaAbsTable[:, np.where(spabs.abundArr == abund)[0][0]]
        assert spabs.sigma_abs_galactic(spabs.energyArr, abund=abund) == pytest.approx(table_values)


def test_1kev():
    """
    The cross-section at 1 keV must be 2.532892 cm^2 (works only for tbabs + angr)
    :return:
    """
    assert spabs.sigma_abs_galactic(1) == pytest.approx(2.532892)
