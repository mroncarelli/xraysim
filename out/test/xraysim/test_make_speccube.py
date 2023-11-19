import pytest
import numpy as np
import os
from astropy.io import fits

from xraysim.sphprojection.mapping import make_speccube, write_speccube
from xraysim.gadgetutils.phys_const import keV2K
from xraysim.specutils.tables import read_spectable, calc_spec

data_dir = os.environ.get('XRAYSIM') + '/tests/inp/'
reference_dir = os.environ.get('XRAYSIM') + '/tests/reference_files/'
snapshot_file = data_dir + 'snap_Gadget_sample'
spfile = data_dir + 'test_emission_table.fits'
reference_file = reference_dir + 'reference.speccube'
npix, size, redshift, center, proj, flag_ene, nsample, nh = 25, 1.05, 0.1, [2500., 2500.], 'z', False, 1, 0.01
nene = fits.open(spfile)[0].header.get('NENE')
test_file = data_dir + 'file_created_for_test.speccube'

spec_cube = make_speccube(snapshot_file, spfile, size=size, npix=npix, redshift=0.1, nh=nh, center=center, proj=proj)


def test_structure(inp=spec_cube):
    """
    The output dictionary must contain all the keywords that must be present in every output
    :param inp: the speccube dictionary to test
    """
    mandatory_keys_set = {'data', 'xrange', 'yrange', 'size', 'pixel_size', 'energy', 'energy_interval', 'units',
                          'coord_units', 'energy_units', 'simulation_file', 'spectral_table', 'proj', 'z_cos', 'd_c',
                          'flag_ene', 'smoothing', 'velocities'}
    for key in mandatory_keys_set:
        assert key in inp.keys()


def test_data_shape(inp=spec_cube):
    """
    The data in the output dictionary must be of the correct shape
    :param inp: the speccube dictionary to test
    """
    assert inp.get('data').shape == (npix, npix, nene)


def test_key_values(inp=spec_cube):
    """
    Some data in the output dictionary must correspond exactly to the input ones
    :param inp: the speccube dictionary to test
    """
    reference_dict = {'size': size, 'pixel_size': size / npix * 60., 'simulation_file': snapshot_file,
                      'spectral_table': spfile, 'proj': proj, 'z_cos': redshift, 'flag_ene': flag_ene,
                      'smoothing': 'ON', 'velocities': 'ON'}

    for key in reference_dict:
        value = reference_dict.get(key)
        if type(value) == float:
            assert inp.get(key) == pytest.approx(value)
        else:
            assert inp.get(key) == value


def test_energy(inp=spec_cube):
    """
    The energy array in the dictionary must correspond to the one of the spectral table (if no ecut is present)
    :param inp: the speccube dictionary to test
    """
    energy = inp.get('energy')
    energy_table = read_spectable(spfile).get('energy')
    assert len(energy) == len(energy_table)
    for ene, ene_table in zip(energy, energy_table):
        assert ene == pytest.approx(ene_table)


def test_isothermal_spectrum_with_temperature_from_table():
    """
    The spectrum of a spec_cube computed assuming isothermal gas with temperature taken directly from the table must
    have the same shape (i.e. non considering normalization) than the corresponding spectrum of the table.
    """
    sptable = read_spectable(spfile)
    z_table = sptable.get('z')
    temperature_table = sptable.get('temperature')  # [keV]
    iz, it = len(z_table) // 2, len(temperature_table) // 2
    z, temp_iso = z_table[iz], temperature_table[it] * keV2K  # [K]
    spec_reference = sptable.get('data')[iz, it, :]
    spec_reference /= spec_reference.mean()  # normalize to mean = 1
    spec_cube_iso = make_speccube(snapshot_file, spfile, size=size, npix=5, redshift=z, center=center, proj=proj,
                                  isothermal=temp_iso, novel=True, nsample=nsample).get('data')

    nene_speccube = spec_cube_iso.shape[2]
    spec_iso = np.ndarray(nene_speccube, dtype='float32')
    for iene in range(nene_speccube):
        spec_iso[iene] = spec_cube_iso[:, :, iene].sum()
    spec_iso /= spec_iso.mean()  # normalize to mean = 1

    for val, val_reference in zip(spec_iso, spec_reference):
        assert val / val_reference == pytest.approx(1., rel=1.e-5)


def test_isothermal_spectrum():
    """
    The spectrum of a spec_cube computed assuming isothermal gas must have the same shape (i.e. non considering
    normalization) than the corresponding spectrum of the table.
    """
    sptable = read_spectable(spfile)
    z_table = sptable.get('z')
    temperature_table = sptable.get('temperature')  # [keV]
    iz = len(z_table) // 2
    z = z_table[iz]
    temp_iso_kev = temperature_table[0] + 0.67 * (
                temperature_table[-1] - temperature_table[0])  # arbitrary value inside the table [keV]
    spec_reference = calc_spec(sptable, z, temp_iso_kev, no_z_interp=True)
    spec_reference /= spec_reference.mean()  # normalize to mean = 1
    temp_iso = temp_iso_kev * keV2K  # [K]
    spec_cube_iso = make_speccube(snapshot_file, spfile, size=size, npix=5, redshift=z, center=center, proj=proj,
                                  isothermal=temp_iso, novel=True, nsample=nsample).get('data')

    spec_iso = np.ndarray(nene, dtype='float32')
    for iene in range(nene):
        spec_iso[iene] = spec_cube_iso[:, :, iene].sum()
    spec_iso /= spec_iso.mean()  # normalize to mean = 1

    for val, val_reference in zip(spec_iso, spec_reference):
        assert val / val_reference == pytest.approx(1., rel=1.e-5)


def test_created_file_matches_reference(speccube_inp=spec_cube, reference=reference_file):
    """
    Writing the spec_cube to a fits file should produce a file with data identical to the reference one.
    """
    if os.path.isfile(test_file):
        os.remove(test_file)
    write_speccube(speccube_inp, test_file)
    hdulist = fits.open(test_file)
    os.remove(test_file)
    hdulist_reference = fits.open(reference)

# File must have Primary and 2 extensions
    assert len(hdulist) == len(hdulist_reference) == 3

    # Primary
    data = hdulist[0].data
    data_reference = hdulist_reference[0].data
    assert len(data.shape) == len(data_reference.shape) == 3
    assert data.shape == data_reference.shape
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                assert data[i, j, k] == pytest.approx(data_reference[i, j, k])

    # Extension 1
    data = hdulist[1].data
    data_reference = hdulist_reference[1].data
    assert len(data.shape) == len(data_reference.shape) == 1
    assert data.shape == data_reference.shape
    for i in range(data.shape[0]):
        assert data[i] == pytest.approx(data_reference[i])

    # Extension 2
    data = hdulist[2].data
    data_reference = hdulist_reference[2].data
    assert len(data.shape) == len(data_reference.shape) == 1
    assert data.shape == data_reference.shape
    for i in range(data.shape[0]):
        assert data[i] == pytest.approx(data_reference[i])
