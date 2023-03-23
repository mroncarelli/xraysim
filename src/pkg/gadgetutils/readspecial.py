import pygadgetreader as pygr

from .phys_const import mu0, m_p, k_B, Xp, Yp


def readtemperature(filename: str, f_cooling=None):
    u = pygr.readsnap(filename, 'u', 'gas', units=0)
    if f_cooling is None:
        f_cooling = pygr.readhead(filename, 'f_cooling')
    if f_cooling == 0:
        temp = 2. / 3. * u * (1.e5 ** 2) * mu0 * m_p / k_B  # [K] (full ionization)
    else:
        ne = pygr.readsnap(filename, 'ne', 'gas', units=0)
        temp = 2. / 3. * u * (1.e5 ** 2) / ((1. + ne) * Xp + 0.25 * Yp) * m_p / k_B  # [K]
    return temp
