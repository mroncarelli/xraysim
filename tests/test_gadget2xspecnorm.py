import pytest
from pkg.gadgetutils.phys_const import Xp, Msun2g, m_p, kpc2cm, x_e0, pi
from pkg.gadgetutils import convert
import numpy as np

gadget_unit_normalization = 1.e-14 * Xp ** 2 * x_e0 / (4. * pi) * (1.e10 * Msun2g / m_p) ** 2 / kpc2cm ** 5


def test_unit_value():
    """
    A fully ionised gas element with mass = 10^10 M_Sun mass, density = 10^10 M_Sun / kpc^3 and d_c = 1 kpc (i.e.
    all values equal to 1 in Gadget units) must have a Xspec normalization equal to 269033750046.39606 (obtained by
    multiplying conversion factors as follows).
    """
    assert convert.gadgget2xspecnorm(1., 1., 1., 1.) == pytest.approx(gadget_unit_normalization)


def test_mass_dependence():
    """
    Xspec normalization should scale linearly with mass
    """
    v = 7.5
    assert convert.gadgget2xspecnorm(v, 1., 1., 1.) / convert.gadgget2xspecnorm(1., 1., 1., 1.) == pytest.approx(v)


def test_density_dependence():
    """
    Xspec normalization should scale linearly with density, if mass is preserved
    """
    v = 42.
    assert convert.gadgget2xspecnorm(1., v, 1., 1.) / convert.gadgget2xspecnorm(1., 1., 1., 1.) == pytest.approx(v)


def test_distance_dependence():
    """
    Xspec normalization should scale inversely with the square of the comoving distance
    """
    v = 123.
    assert convert.gadgget2xspecnorm(1., 1., v, 1.) / convert.gadgget2xspecnorm(1., 1., 1., 1.) == pytest.approx(
        1. / v ** 2)


def test_hubble_constant_dependence():
    """
    Xspec normalization should scale like h^3
    """
    v = 0.7
    assert convert.gadgget2xspecnorm(1., 1., 1., v) / convert.gadgget2xspecnorm(1., 1., 1., 1.) == pytest.approx(v ** 3)


def test_xe_dependence():
    """
    Xspec normalization should scale linearly with ionization fraction
    """
    npart = 100
    mass = rho = d_c = np.full(npart, 1.)
    x_e = np.full(npart, 0.8)
    h = 0.67
    norm_reference = convert.gadgget2xspecnorm(mass, rho, d_c, h)
    norm_test = convert.gadgget2xspecnorm(mass, rho, d_c, h, ne=x_e)
    assert all(nt / nr == pytest.approx(x_e_i / x_e0) for nt, nr, x_e_i in zip(norm_test, norm_reference, x_e))
