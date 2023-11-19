from xraysim.specutils import absorption as spabs
import numpy as np


def test_values_between_0_and_1():
    """
    f_abs must be between 0 and 1
    """
    nh = 0.1
    energy = np.linspace(spabs.energy_table.min(), spabs.energy_table.max(), 1000, endpoint=True)
    vals = spabs.f_abs_galactic(energy, nh)
    assert all(vals >= 0.)
    assert all(vals <= 1.)


def test_decreases_with_nh():
    """
    Increasing nh must give larger absorption and, therefore, lower f_abs_galactic, at all energies
    """
    nh = np.linspace(0., 1., 101, endpoint=True)
    energy = np.linspace(spabs.energy_table.min(), spabs.energy_table.max(), 100, endpoint=True)
    f_abs0 = spabs.f_abs_galactic(energy, nh[0])
    for index in range(1, len(nh)):
        f_abs1 = spabs.f_abs_galactic(energy, nh[index])
        assert all(f_abs0/f_abs1 >= 1.)
        f_abs0 = f_abs1
