from .phys_const import Xp, Msun2g, Mpc2cm, m_p, kpc2cm, x_e0, pi, c_light
import numpy as np


def mass2NH(mass, h, xp=None):
    if xp is None:
        xp = Xp
    return mass * 1e10 * Msun2g / h * xp / m_p  # [---]


def rho2nH(rho, h, xp=None):
    if xp is None:
        xp = Xp
    return rho * 1e10 * Msun2g / kpc2cm ** 3 * h ** 2 * xp / m_p  # [cm^-3]


def gadgget2xspecnorm(mass, rho, d_c, h, ne=None):
    """
    Calculates the Xspec normalization from Gadget units
    :param mass: particle mass [10^10 h^-1 M_Sun]
    :param rho: particle physical density [10^10 h^2 M_Sun kpc^-3]
    :param d_c: comoving distance [h^-1 kpc]. CAUTION: the redshift used to compute the comoving distance should
    eventually be corrected for peculiar velocities, if they are relevant
    :param h: Hubble parameter [100 km s^-1 Mpc^-1]
    :param ne: electron to hydrogen ratio, i.e. n_e / n_H [---]. If not set assumes full ionization
    :return: Xspec normalization factor [10^14 cm^-5], i.e. Int(n_e * n_H * dV) / (4 pi d_c^2) * 10^-14
    """

    conversion_factor = 1.e-14 * (1.e10 * Msun2g * Xp / m_p) ** 2 / (4. * pi * kpc2cm ** 5) * x_e0 * h ** 3

    if ne is not None:
        conversion_factor *= ne / x_e0  # considering ionization fraction

    result = conversion_factor * mass * rho / d_c ** 2  # [10^14 cm^-5]

    return result


def vpec2zobs(v_pec, z_h, units=None):
    """
    Converts a peculiar velocity into observed redshift, considering also a Hubble-flow redshift.
    :param v_pec: (float) peculiar velocity, in units of light-speed (unless argument units specifies otherwise)
    :param z_h: (float) Hubble flow redshift, i.e. the one corresponding to comoving distance
    :param units: (str) can be either 'cgs', 'mks', 'm/s', 'cm/s' or 'km/s'
    :return: the corresponding observed redshift
    """

    if units:
        if units in ['cgs', 'cm/s']:
            conv = c_light
        elif units in ['mks', 'm/s']:
            conv = c_light * 1e-2
        elif units == 'km/s':
            conv = c_light * 1e-2
        else:
            print("Invalid unit: ", units, "Must be one of 'cgs', 'mks', 'm/s', 'cm/s' or 'km/s' or None")
            raise ValueError
    else:
        conv = 1.

    return np.sqrt((1. + v_pec / conv) / (1. - v_pec / conv)) * (1. + z_h) - 1.