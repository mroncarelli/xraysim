import pygadgetreader as pygr
import warnings
from .phys_const import mu0, m_p, k_B, Xp, Yp, keV2K


def readtemperature(filename: str, f_cooling=None, suppress=None, units=None):
    u = pygr.readsnap(filename, 'u', 'gas', units=0, suppress=suppress)
    if f_cooling is None:
        f_cooling = pygr.readhead(filename, 'f_cooling')
    if f_cooling == 0:
        temp = 2. / 3. * u * (1.e5 ** 2) * mu0 * m_p / k_B  # [K] (full ionization)
    else:
        ne = pygr.readsnap(filename, 'ne', 'gas', units=0, suppress=suppress)
        temp = 2. / 3. * u * (1.e5 ** 2) / ((1. + ne) * Xp + 0.25 * Yp) * m_p / k_B  # [K]

    if units is None or units == 'K':
        pass
    elif units == 'keV':
        temp /= keV2K
    else:
        warnings.warn("WARNING (readtemperature): Invalid value of keyword 'units': " + str(units) + ". Assuming 'K'.")

    return temp  # [K]
