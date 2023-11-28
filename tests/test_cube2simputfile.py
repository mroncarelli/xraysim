import pytest
import numpy as np
import os
from astropy.io import fits

from xraysim.specutils.sixte import cube2simputfile
from xraysim.sphprojection.mapping import make_speccube
from xraysim.specutils.tables import read_spectable, calc_spec
from xraysim.gadgetutils.phys_const import keV2K

environmentVariablesPathList = [os.environ.get('XRAYSIM'), os.environ.get('SIXTE_INSTRUMENTS')]
inputDir = os.environ.get('XRAYSIM') + '/tests/inp/'
referenceDir = os.environ.get('XRAYSIM') + '/tests/reference_files/'
snapshotFile = inputDir + 'snap_Gadget_sample'
spFile = inputDir + 'test_emission_table.fits'
referenceSimputFile = referenceDir + 'reference.simput'
npix, size, redshift, center, proj, flag_ene, tcut, nsample, nh = 25, 1.05, 0.1, [2500., 2500.], 'z', False, 1.e6, 1, 0.01
t_iso_keV = 6.3  # [keV]
t_iso = t_iso_keV * keV2K  # 73108018.313372612  # [K] (= 6.3 keV)
nene = fits.open(spFile)[0].header.get('NENE')
testSimputFile = inputDir + 'file_created_for_test.simput'

# Isothermal + no velocities
speccubeIsothermalNovel = make_speccube(snapshotFile, spFile, size=size, npix=npix, redshift=redshift, center=center,
                                        proj=proj, nsample=nsample, isothermal=t_iso, novel=True)

speccube = make_speccube(snapshotFile, spFile, size=size, npix=npix, redshift=redshift, center=center,
                         proj=proj, tcut=tcut, nh=nh, nsample=nsample)


def assert_string_in_header_matches_reference(string: str, string_reference: str):
    """
    Checks that a string in a header matches a reference string. If the string corresponds to a file name it may start
    with a path corresponding to an environment variable: in this case this part of the string is ignored, and it
    checks that the final parts of the strings match. Otherwise, it checks that the two sting are identical.
    :param string: (str) String to check.
    :param string_reference: (str) Reference string.
    :return: None
    """

    for envVar in environmentVariablesPathList:
        if string.startswith(envVar):
            assert string_reference.endswith(string.replace(envVar, ''))
            return None
    assert string == string_reference
    return None


def assert_header_has_all_keywords_and_values_of_reference(header: fits.header, header_reference: fits.header):
    """
    Checks that a header contains all the keys, with same values, than the reference one. Other keywords/values may
    be present and do not affect the result.
    :param header: (fits.header) Header to check.
    :param header_reference: (fits.header) Reference header.
    :return: None.
    """
    for key in header_reference.keys():
        val_reference = header_reference.get(key)
        assert key in header
        val = header.get(key)
        if type(val) == str:
            assert_string_in_header_matches_reference(val, val_reference)
        else:
            assert val == pytest.approx(val_reference)
    return None


def test_file_created(inp=speccubeIsothermalNovel, out=testSimputFile):
    """
    The output SIMPUT file must be correctly created
    :param inp: (dict) spectral cube structure
    :param out: (str) output SIMPUT file
    """
    if os.path.isfile(out):
        os.remove(out)
    cube2simputfile(inp, out)
    assert os.path.isfile(out)
    os.remove(out)


def test_primary_header_keywords(inp=speccubeIsothermalNovel, out=testSimputFile):
    """
    The header of the Primary of the output SIMPUT file must contain a series of keywords whose value depend on the
    input data
    :param inp: (dict) spectral cube structure
    :param out: (str) output SIMPUT file
    """
    if os.path.isfile(out):
        os.remove(out)
    center_map = (12., 34.)
    cube2simputfile(inp, out, pos=center_map)
    header = fits.open(out)[0].header
    os.remove(out)
    assert header.get('SIM_FILE') == inp.get('simulation_file')
    assert header.get('SP_FILE') == inp.get('spectral_table')
    assert header.get('PROJ') == inp.get('proj')
    assert header.get('Z_COS') == pytest.approx(inp.get('z_cos'))
    assert (header.get('D_C'), header.comments['D_C']) == pytest.approx((inp.get('d_c'), '[Mpc]'))
    assert header.get('NPIX') == inp.get('data').shape[0]
    assert header.get('NENE') == inp.get('data').shape[2]
    assert (header.get('ANG_PIX'), header.comments['ANG_PIX']) == pytest.approx((inp.get('pixel_size'), '[arcmin]'))
    assert (header.get('ANG_MAP'), header.comments['ANG_MAP']) == pytest.approx((inp.get('size'), '[deg]'))
    assert (header.get('RA_C'), header.comments['RA_C']) == pytest.approx((center_map[0], '[deg]'))
    assert (header.get('DEC_C'), header.comments['DEC_C']) == pytest.approx((center_map[1], '[deg]'))
    assert header.get('SMOOTH') == inp.get('smoothing')
    assert header.get('VPEC') == inp.get('velocities')


def test_isothermal_spectrum(inp=speccubeIsothermalNovel, out=testSimputFile):
    """
    All the spectra contained in the Extension 2 of the SIMPUT file must be isothermal with T equal to the value
    indicated by the spec_cube, with arbitrary normalization.
    :param inp: (dict) spectral cube structure
    :param out: (str) output SIMPUT file
    """

    # Creating SIMPUT file
    if os.path.isfile(out):
        os.remove(out)
    cube2simputfile(inp, out)
    hdulist = fits.open(testSimputFile)
    os.remove(out)
    header0 = hdulist[0].header

    # Reading the spectral table and the temperature from the header of the SIMPUT file just created
    sptable = read_spectable(header0.get('SP_FILE'))
    temp = header0.get('ISOTHERM') / keV2K  # [keV]

    # Getting energy and (normalized spectrum) from table and from
    z = speccubeIsothermalNovel.get('z_cos')
    energy_reference = speccubeIsothermalNovel.get('energy')  # [keV]
    spectrum_reference = calc_spec(sptable, z, temp, no_z_interp=True)
    spectrum_reference /= spectrum_reference.mean()  # normalize to mean = 1

    # Checking energy from the SIMPUT file
    for energy in hdulist[2].data['ENERGY']:
        for val, val_reference in zip(energy, energy_reference):
            assert val == pytest.approx(val_reference, rel=1.e-6)

    # Checking spectra from the SIMPUT file
    for spectrum in hdulist[2].data['FLUXDENSITY']:
        spectrum_norm = spectrum / spectrum.mean()  # normalize to mean = 1
        for val, val_reference in zip(spectrum_norm, spectrum_reference):
            assert val == pytest.approx(val_reference, rel=1.e-6)


def test_created_file_matches_reference(inp=speccube, out=testSimputFile, reference=referenceSimputFile):
    """
    Writing the spec_cube to a SIMPUT file should produce a file with data identical to the reference one.
    """
    if os.path.isfile(testSimputFile):
        os.remove(testSimputFile)
    cube2simputfile(inp, out)
    hdulist = fits.open(testSimputFile)
    os.remove(out)
    hdulist_reference = fits.open(reference)

    for hdu, hdu_reference in zip(hdulist, hdulist_reference):
        # Checking header keywords: created file must contain all keywords of reference file, with same value
        assert_header_has_all_keywords_and_values_of_reference(hdu.header, hdu_reference.header)
        # Checking that data match
        assert np.all(hdu.data == hdu_reference.data)

