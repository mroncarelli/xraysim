import pygadgetreader as pygr
import warnings
import numpy as np
from .phys_const import mu0, m_p, k_B, Xp, Yp, keV2K, c_light


def readtemperature(filename: str, units='K', f_cooling=None, suppress=None):
    """
    Reads the temperature of the gas particles from a Gadget file.
    :param filename: (str) The Gadget file.
    :param units: (str) Can be 'K' or 'keV', default 'K'
    :param f_cooling: if set to 0 assumes full ionization, if set to 1 the ionization fraction is read from the
    snapshot. Default: None, i.e. reads f_cooling from the simulation file.
    :param suppress: (int) Verbosity.
    :return: The vector of temperatures.
    """
    u = pygr.readsnap(filename, 'u', 'gas', units=0, suppress=suppress)  # Internal energy per unit mass [km^2 s^-2]
    if f_cooling is None:
        f_cooling = pygr.readhead(filename, 'f_cooling')
    if f_cooling == 0:
        temp = 2. / 3. * u * (1.e5 ** 2) * mu0 * m_p / k_B  # [K] (full ionization)
    else:
        ne = pygr.readsnap(filename, 'ne', 'gas', units=0, suppress=suppress)
        temp = 2. / 3. * u * (1.e5 ** 2) / ((1. + ne) * Xp + 0.25 * Yp) * m_p / k_B  # [K]

    if units.lower() == 'k':
        pass
    elif units.lower() == 'kev':
        temp /= keV2K
    else:
        warnings.warn("WARNING (readtemperature): Invalid value of keyword 'units': " + str(units) + ". Assuming 'K'.")

    return temp


def readvelocity(filename: str, units='km/s', redshift=None, suppress=None):
    """
    Reads the velocity of the gas particles from a Gadget file.
    :param filename: (str) The Gadget file.
    :param units: (str) Can be 'km/s', 'cm/s', 'cgs' or 'c', default 'km/s'.
    :param redshift: (float) If set it is used for the 1/sqrt(redshift) conversion, otherwise the redshift of the
    snapshot is considered. Default: None.
    :param suppress: (int) Verbosity.
    :return: The (ngas, 3) vector of velocities.
    """
    if redshift is None:
        redshift = pygr.readhead(filename, 'redshift')

    vel = pygr.readsnap(filename, 'vel', 'gas', units=0, suppress=suppress) / np.sqrt(1 + redshift)
    if units.lower() is None or units == 'km/s':
        pass
    elif units.lower() in ['cm/s', 'cgs']:
        vel *= 1.e5
    elif units.lower() == 'c':
        vel *= 1.e5 / c_light
    else:
        warnings.warn("WARNING (readvelocity): Invalid value of keyword 'units': " + str(units) + ". Assuming 'km/s'.")

    return vel
