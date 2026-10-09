import os
import numpy as np
import copy as cp
import warnings

absorptionDataFile = os.path.join(os.path.dirname(__file__), "sigma_tbabs.csv")
abundArr = np.loadtxt(absorptionDataFile, delimiter=',', max_rows=1, dtype=str)[1:]
for ind in range(len(abundArr)):
    abundArr[ind] = abundArr[ind].strip().lower()
energyArr = np.loadtxt(absorptionDataFile, delimiter=',', skiprows=1, usecols=0, dtype=float)
sigmaAbsTable = np.loadtxt(absorptionDataFile, delimiter=',', skiprows=1, dtype=float)[:, 1:]


def __get_abund(arg):
    return 'angr' if arg is None else arg.strip().lower()


def sigma_abs_galactic(energy, abund='angr') -> np.ndarray:
    """
    Computes the Galactic absorption cross-section for a given energy value according to Wilms, Allen & McCray (2000). 
    Reproduces the tbabs model of Xspec.
    :param energy: (float array) Energy value [keV], must be withing the range of the table sigma_tbabs.dat
    :patam abund: (str) Abundance table. Must be one of the Xspec tables. Default 'angr'
    :return: Galactic absorption cross-section [cm^2]
    """
    abund = __get_abund(abund)
    if abund not in abundArr:
        raise ValueError("Invalid abundance table. Must be one of the Xspec abundance tables: " + " ".join(abundArr))
    column = np.where(abundArr == abund)[0][0]
    return np.interp(energy, energyArr, sigmaAbsTable[:, column], left=None, right=None)


def f_abs_galactic(energy, nh: float, abund='angr') -> np.ndarray:
    """
    Computes the Galactic absorption correction for a given value of energy and hydrogen surface density according to
    Wilms, Allen & McCray (2000). Reproduces the tbabs model of Xspec
    :param energy: energy: (float array) energy value [keV], must be withing the range of the table sigma_tbabs.dat
    :param nh: (float) hydrogen column density [10^22 cm^-2]
    :patam abund: (str) Abundance table. Must be one of the Xspec tables. Default 'angr'
    :return: Galactic absorption correction, i.e. v_absorbed = f_abs_galactic * v_full
    """
    return np.exp(-nh * sigma_abs_galactic(energy, abund=abund))


def convert_nh(struct, nh: float, abund=None, nh_input=None, preserve_input=True):
    """
    Converts a structure containing spectral data (in the 'data' key) changing the value of its hydrogen column
    density. Since the input data may already have been computed with another value of N_H, the info on the original
    value is considered with this priority: 1) the nh_input argument (if provided), 2) the input structure value stored
    in the 'nh' key (if present), 3) nh=0 (no absorption). Same for the abund argument.
    :param struct: structure containing the spectral data in the 'data' key, and energy values in the 'energy' key
    (i.e. a spectral table from the 'tables' module or a spectral-map from sphprojection.mapping.specmap). The
    spectra are assumed to be in the last/rightmost dimension of the 'data' array
    :param nh: hydrogen column density [10^22 cm^-2]
    :patam abund: (str) Abundance table. Must be one of the Xspec tables. Default 'angr'
    :param nh_input: hydrogen column density in the input data [10^22 cm^-2] (not considered if the input structure
    contains the 'nh' key)
    :param preserve_input: (bool) if set to True the input struct is preserved from modification, default True
    :return: a structure with the absorbed spectrum in the 'data' key. The 'nh' key is updated/added, all other fields
    are left untouched.
    """
    if ('data' not in struct) or ('energy' not in struct):
        raise ValueError("Invalid argument: it must contain the 'data' and 'energy' keys.")

    energy = struct.get('energy')
    if len(energy) != struct['data'].shape[-1]:
        raise ValueError("Invalid argument: incoherent shape between 'data' and 'energy' keys.")

    abund_input = struct.get('abund')
    if abund_input is not None:
        abund_input = __get_abund(abund_input)
        if abund is None:
            abund = abund_input
        else:
            abund = __get_abund(abund_input)
            if abund != abund_input:
                warnings.warn("Abundance table assumed in the input differs from the one provided as argument: " +
                              abund_input + " != " + abund +" (the value will be substituted)")

    if nh_input is not None:
        nh_struct = struct.get('nh')
        if nh_struct != nh_input:
            warnings.warn("Hydrogen column density keyword is present in the input and differs from the one provided "
                          "as argument (the value will be substituted)")
        fabs0 = f_abs_galactic(energy, nh_input, abund=abund)
    elif 'nh' in struct:
        fabs0 = f_abs_galactic(energy, struct.get('nh'), abund=struct.get('abund'))
    else:
        fabs0 = np.full(len(energy), 1.)  # assumes no absorption in the original data

    result = cp.deepcopy(struct) if preserve_input else struct
    result['data'] *= np.float32(f_abs_galactic(energy, nh, abund=abund) / fabs0)
    result['nh'] = nh
    result['abund'] = abund

    return result
