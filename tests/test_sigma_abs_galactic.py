import pytest
from xraysim.specutils import absorption as spabs


def test_table_energy_ascending():
    """
    The energy column in the energy table must be in ascending order
    """
    for index in range(1, len(spabs.energy_table)):
        assert spabs.energy_table[index] > spabs.energy_table[index - 1]


def test_table_sigma_abs_gt_0():
    """
    Values of sigma_abs in the table must be strictly positive
    """
    assert all(spabs.sigma_abs_table > 0.)


def test_table_values():
    """
    Values for input energy present in the table must be the corresponding table values
    """
    for index, ene in enumerate(spabs.energy_table):
        assert spabs.sigma_abs_galactic(ene) == pytest.approx(spabs.sigma_abs_table[index])


def test_1kev():
    """
    The cross-section at 1 keV must be 2.422534 cm^2 (works only for wabs)
    :return:
    """
    assert spabs.sigma_abs_galactic(1.) == pytest.approx(2.422534)
