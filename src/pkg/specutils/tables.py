from astropy.io import fits
import numpy as np


def nearest_index_sorted(array, value):
    """
    Returns the index whose value is closest to the input value. Assumes the array is sorted in ascending order.
    :param array: array (sorted ascending) to search into
    :param value: value to search
    :return: index of the closest value
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and np.abs(value - array[idx - 1]) < np.abs(value - array[idx]):
        idx += -1
    return idx


def largest_index_smaller(array, value):
    """
    Returns the largest index whose value is smaller than the input value. Assumes the array is sorted in ascending
    order.
    :param array: array (sorted ascending) to search into
    :param value: value: value to search
    :return: index of the largest value smaller than the input value
    """
    idx = len(array) - 1
    while idx > 0 and array[idx] >= value:
        idx += -1

    if array[idx] < value:
        return idx
    else:
        return None


def smallest_index_larger(array, value):
    """
    Returns the smallest index whose value is larger than the input value. Assumes the array is sorted in ascending
    order.
    :param array: array (sorted ascending) to search into
    :param value: value: value to search
    :return: index of the smallest value larger than the input value
    """
    idx = 0
    while idx < len(array) - 1 and array[idx] <= value:
        idx += 1

    if array[idx] > value:
        return idx
    else:
        return None


def reversed_fits_axis_order(inp) -> bool:
    """
    Determines whether a FITS file has been created following the column-major-order convention and, therefore,
    requires to transpose the input when 2d, 3d, ... data are present. WARNING: this is not complete! I could figure
    out that this is the case for FITS files created by the IDL routine MWRFITS, and how to identify these files. For
    FITS files created by other languages, applications, etc., this has to verified and implemented.
    :param inp: input file (FITS) or HDUList
    :return: (bool) True if it is in reversed axis order, False otherwise
    """

    input_type = type(inp)
    if input_type == str:
        hdulist = fits.open(inp)
        hdulist.close()
    elif input_type == fits.hdu.HDUList:
        hdulist = inp
    else:
        print("Invalid input in reversed_fits_axis_order: must be a string or HDUList")
        raise ValueError

    if hdulist[0].header.comments['SIMPLE'].lower().startswith('written by idl'):
        result = True
    else:
        result = False

    return result


def read_spectable(filename: str, z_cut=None, temperature_cut=None, energy_cut=None):
    """
    Reads a spectrum table from a file.
    :param filename: (str) input file (FITS)
    :param z_cut: (float 2) optional redshift interval where to cut the table [---]
    :param temperature_cut: (float 2) optional temperature interval where to cut the table [keV]
    :param energy_cut: (float 2) optional energy interval where to cut the table [keV]
    :return: a structure containing the spectrum table, in the 'data' key, with other information in the other keys.
    With standard Xspec parameters the units are [10^-14 photons s^-1 cm^3] or [10^-14 keV s^-1 cm^3] if the
    header of the file has the keyword FLAG_ENE = 1.
    """
    hdulist = fits.open(filename)
    spec = hdulist[0].data
    if reversed_fits_axis_order(hdulist):
        spec = spec.transpose()
    z = hdulist[1].data
    temperature = hdulist[2].data
    energy = hdulist[3].data

    # Sanity check
    assert spec.shape[0] == len(z) and spec.shape[1] == len(temperature) and spec.shape[2] == len(energy)

    if z_cut:
        try:
            z0, z1 = float(z_cut[0]), float(z_cut[1])
        except BaseException:
            print("Invalid tcut: ", z_cut, "Must be a 2d number vector")
            raise ValueError
        i0, i1 = largest_index_smaller(z, z0), smallest_index_larger(z, z1)
        if i0 is None:
            i0 = 0  # TODO: WARNING
        if i1 is None:
            i1 = len(z) - 1  # TODO: WARNING
        z = z[i0:i1 + 1]
        spec = spec[i0:i1 + 1, :, :]

    if temperature_cut:
        try:
            t0, t1 = float(temperature_cut[0]), float(temperature_cut[1])
        except BaseException:
            print("Invalid tcut: ", temperature_cut, "Must be a 2d number vector")
            raise ValueError
        i0, i1 = largest_index_smaller(temperature, t0), smallest_index_larger(temperature, t1)
        if i0 is None:
            i0 = 0  # TODO: WARNING
        if i1 is None:
            i1 = len(temperature) - 1  # TODO: WARNING
        if i0 == i1:
            i1 += 1
            if i1 == len(temperature):
                i0, i1 = len(temperature) - 2, len(temperature) - 1
        temperature = temperature[i0:i1 + 1]
        spec = spec[:, i0:i1 + 1, :]

    if energy_cut:
        try:
            e0, e1 = float(energy_cut[0]), float(energy_cut[1])
        except BaseException:
            print("Invalid tcut: ", energy_cut, "Must be a 2d number vector")
            raise ValueError
        i0, i1 = largest_index_smaller(energy, e0), smallest_index_larger(energy, e1)
        if i0 is None:
            i0 = 0  # TODO: WARNING
        if i1 is None:
            i1 = len(energy) - 1  # TODO: WARNING
        energy = energy[i0:i1 + 1]
        spec = spec[:, :, i0:i1 + 1]

    result = {
        'data': spec,
        'z': z,
        'temperature': temperature,
        'energy': energy,
        'units': hdulist[0].header['UNITS'],
        'flag_ene': hdulist[0].header['FLAG_ENE'] == 1,
        'model': hdulist[0].header['MODEL'],
        'param': hdulist[0].header['PARAM'],
        'temperature_units': hdulist[2].header['UNITS'],
        'energy_units': hdulist[3].header['UNITS']
    }
    if 'NH' in hdulist[0].header:
        result['nh'] = hdulist[0].header['NH']

    hdulist.close()
    return result


def calc_spec(spectable, z, temperature, no_z_interp=False, flag_ene=False):
    """
    Calculates a spectrum from a table for a given redshift and temperature
    :param spectable: structure containing the spectrum table
    :param z: redshift where to compute the spectrum [---]
    :param temperature: temperature where to compute the spectrum [keV]
    :param no_z_interp: (boolean) if set to True redshift interpolation is turned off (useful to avoid line-emission
     smearing in high resolution spectra)
    :param flag_ene: (boolean) if True the spectrum is calculated in energy, if False in photons (default False)
    :return: array containing the spectrum. With standard Xspec parameters the units are [10^-14 photons s^-1 cm^3] or
    [10^-14 keV s^-1 cm^3] if flag_ene is set to True.
    """

    data = spectable.get('data')  # [10^-14 photons s^-1 cm^3] or [10^-14 keV s^-1 cm^3]
    nene = data.shape[2]
    z_table = spectable.get('z')
    temperature_table = spectable.get('temperature')  # [keV]
    flag_ene_table = spectable.get('flag_ene')

    # Redshift (index 0)
    if no_z_interp:
        iz = nearest_index_sorted(z_table, z)
        data = data[iz, :, :]
    else:
        iz0 = largest_index_smaller(z_table, z)
        if iz0 is None:
            iz0 = 0  # TODO: WARNING
        elif iz0 == len(z_table) - 1:
            iz0 = len(z_table) - 2  # TODO: WARNING
        iz1 = iz0 + 1
        fz = (z - z_table[iz0]) / (z_table[iz1] - z_table[iz0])
        data = (1 - fz) * data[iz0, :, :] + fz * data[iz1, :, :]

    # Temperature (index 1)
    it0 = largest_index_smaller(temperature_table, temperature)
    if it0 is None:
        it0 = 0  # TODO: WARNING
    elif it0 == len(temperature_table) - 1:
        it0 = len(temperature_table) - 2  # TODO: WARNING
    it1 = it0 + 1
    ft = (np.log(temperature) - np.log(temperature_table[it0])) / (
            np.log(temperature_table[it1]) - np.log(temperature_table[it0]))
    valid = np.where(data[it0, :] * data[it0, :] > 0.)
    result = np.zeros(nene)
    result[valid] = np.exp((1 - ft) * np.log(data[it0, valid]) + ft * np.log(
        data[it1, valid]))  # [10^-14 photons s^-1 cm^3] or [10^-14 keV s^-1 cm^3]

    # Converting photons to energy or vice-versa if required
    if flag_ene != flag_ene_table:
        energy = spectable.get('energy')  # [keV]
        if flag_ene:
            for ind, ene in enumerate(energy):
                result[:, :, ind] *= ene  # [10^-14 keV s^-1 cm^3]
        else:
            for ind, ene in enumerate(energy):
                result[:, :, ind] /= ene  # [10^-14 photons s^-1 cm^3]

    return result  # [10^-14 photons s^-1 cm^3] or [10^-14 keV s^-1 cm^3]
