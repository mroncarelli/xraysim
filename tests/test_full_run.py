import pytest
import numpy as np
import os
from astropy.io import fits

from src.pkg.sphprojection.mapping import make_speccube, write_speccube, read_speccube
from src.pkg.specutils.sixte import cube2simputfile, create_eventlist, make_pha

inputDir = os.environ.get('XRAYSIM') + '/tests/inp/'
referenceDir = os.environ.get('XRAYSIM') + '/tests/reference_files/'
snapshotFile = inputDir + 'snap_Gadget_sample'
spFile = inputDir + 'test_emission_table.fits'
referenceSpcubeFile = referenceDir + 'reference.speccube'
referenceSimputFile = referenceDir + 'reference.simput'
referenceEvtFile = referenceDir + 'reference.evt'
referencePhaFile = referenceDir + 'reference.pha'

spcubeFile = referenceDir + "spcube_file_created_for_test.spcube"
spcubeFile2 = referenceDir + "spcube_file_created_for_test_2.spcube"
simputFile = referenceDir + "simput_file_created_for_test.simput"
evtFile = referenceDir + "evt_file_created_for_test.evt"
phaFile = referenceDir + "pha_file_created_for_test.pha"


def header_has_all_keywords_and_values_of_reference(header: fits.header, header_reference: fits.header) -> bool:
    """
    Checks that a header contains all the keys, with same values, than the reference one. Other keywords/values may
    be present and do not affect the result.
    :param header: (fits.header) Header to check.
    :param header_reference: (fits.header) Reference header.
    :return: (bool) True if all key/values match, False otherwise.
    """
    result = True
    for key in header_reference.keys():
        # TODO: Create a method to check HISTORY
        # TODO: Check why COMMENT in pha file fails
        if key not in ['DATE', 'CREADATE', 'HISTORY', 'COMMENT']:
            result = result and header.get(key) == pytest.approx(header_reference.get(key))
    return result


def hdu_list_matches_reference(inp: fits.hdu.hdulist.HDUList, reference: fits.hdu.hdulist.HDUList) -> bool:
    """
    Checks that a speccube file matches a reference one
    :param inp: (HDUList) HDUList to check.
    :param reference: (HDUList) Reference HDUList.
    :return: (bool) True if the HDUList content matches the reference, False otherwise.
    """

    result = True
    for hdu, hdu_reference in zip(inp, reference):
        result = result and header_has_all_keywords_and_values_of_reference(hdu.header, hdu_reference.header)
        result = result and bool(np.all(hdu.data == hdu_reference.data))

    return result


def test_full_run():
    """
    A full run from Gadget snapshot to pha file, checking that each intermediate step produces a file compatible with
    reference one.
    """

    # Creating a speccube file from a calculated speccube
    speccube_calculated = make_speccube(snapshotFile, spFile, 1.05, 25, redshift=0.1, center=[2500., 2500.], proj='z',
                                        tcut=1.e6, nh=0.01, nsample=1)
    if os.path.isfile(spcubeFile):
        os.remove(spcubeFile)
    write_speccube(speccube_calculated, spcubeFile)
    assert os.path.isfile(spcubeFile)
    del speccube_calculated

    reference_speccube = fits.open(referenceSpcubeFile)

    # Checking that file content matches reference
    assert hdu_list_matches_reference(fits.open(spcubeFile), reference_speccube)

    # Creating a speccube file from the speccube read from the file
    speccube_read = read_speccube(spcubeFile)
    os.remove(spcubeFile)
    if os.path.isfile(spcubeFile2):
        os.remove(spcubeFile2)
    write_speccube(speccube_read, spcubeFile2)
    assert os.path.isfile(spcubeFile2)

    # Checking that file content matches reference
    assert hdu_list_matches_reference(fits.open(spcubeFile2), reference_speccube)

    # Creating a SIMPUT file from a speccube
    if os.path.isfile(simputFile):
        os.remove(simputFile)
    cube2simputfile(speccube_read, simputFile)
    del speccube_read

    # Checking that file content matches reference
    assert hdu_list_matches_reference(fits.open(simputFile), fits.open(referenceSimputFile))

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    create_eventlist(simputFile, 'xrism-resolve-test', 1.e5, evtFile, background=False, seed=42)
    os.remove(simputFile)

    # Checking that file content matches reference
    assert hdu_list_matches_reference(fits.open(evtFile), fits.open(referenceEvtFile))

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    make_pha(evtFile, phaFile)
    os.remove(evtFile)

    # Checking that file content matches reference
    assert hdu_list_matches_reference(fits.open(phaFile), fits.open(referencePhaFile))
    os.remove(phaFile)
