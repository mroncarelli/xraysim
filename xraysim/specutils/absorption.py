import os
import numpy as np
import copy as cp

absorption_table_path = os.path.join(os.path.dirname(__file__), "sigma_wabs.dat")
energy_table = np.loadtxt(absorption_table_path, skiprows=1, usecols=0, dtype=float)
sigma_abs_table = np.loadtxt(absorption_table_path, skiprows=1, usecols=1, dtype=float)


def sigma_abs_galactic(energy) -> float:
    """
    Computes the Galactic absorption cross-section for a given energy value according to Morrison and McCammon (1983,
    ApJ, 270, 119). Reproduces the wabs model in Xspec (https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node271.html).
    :param energy: (float array) energy value [keV], must be withing the range of the table sigma_wabs.dat
    :return: Galactic absorption cross-section [cm^2]
    """
    return np.interp(energy, energy_table, sigma_abs_table, left=None, right=None)


def f_abs_galactic(energy, nh: float):
    """
    Computes the Galactic absorption correction for a given value of energy and hydrogen surface density according to
    Morrison and McCammon (1983, ApJ, 270, 119). Mimics the wabs model in Xspec
    (https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node271.html).
    :param energy: energy: (float array) energy value [keV], must be withing the range of the table sigma_wabs.dat
    :param nh: hydrogen column density [10^22 cm^-2]
    :return: Galactic absorption correction, i.e. v_absorbed = f_abs_galactic * v_full
    """
    return np.exp(-nh * sigma_abs_galactic(energy))


def convert_nh(struct, nh: float, nh_input=None, preserve_input=True):
    """
    Converts a structure containing spectral data (in the 'data' key) changing the value of its hydrogen column
    density. Since the input data may already have been computed with another value of N_H, the info on the original
    value is considered with this priority: 1) the input structure value stored in the 'nh' key (if present), 2) the
    nh_input argument (if provided), 3) nh=0 (no absorption).
    :param struct: structure containing the spectral data in the 'data' key, and energy values in the 'energy' key
    (i.e. a spectral table from the 'tables' module or a spectral cube from sphprojection.mapping.make_cube). The
    spectra are assumed to be in the last/rightmost dimension of the 'data' array
    :param nh: hydrogen column density [10^22 cm^-2]
    :param nh_input: hydrogen column density in the input data [10^22 cm^-2] (not considered if the input structure
    contains the 'nh' key
    :param preserve_input: (bool) if set to True the input struct is preserved from modification, default True
    :return: a structure with the absorbed spectrum in the 'data' key. The 'nh' key is updated/added, all other fields
    are left untouched.
    """
    if ('data' not in struct) or ('energy' not in struct):
        print("Invalid argument in convert_nh: it must contain the 'data' and 'energy' keys.")
        raise ValueError

    energy = struct.get('energy')
    if len(energy) != struct['data'].shape[-1]:
        print("Invalid argument in convert_nh: incoherent shape between 'data' and 'energy' keys.")
        raise ValueError

    if 'nh' in struct:
        fabs0 = f_abs_galactic(energy, struct.get('nh'))
    elif nh_input is not None:
        fabs0 = f_abs_galactic(energy, nh_input)
    else:
        fabs0 = np.full(len(energy), 1.)  # assumes no absorption in the original data

    result = cp.deepcopy(struct) if preserve_input else struct
    result['data'] *= np.float32(f_abs_galactic(energy, nh) / fabs0)
    result['nh'] = nh

    return result
