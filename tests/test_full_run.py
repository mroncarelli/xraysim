import pytest
import numpy as np
import os
from astropy.io import fits

from xraysim.sphprojection.mapping import make_speccube, write_speccube, read_speccube
from xraysim.specutils.sixte import cube2simputfile, create_eventlist, make_pha

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


def history_unpack(history: list) -> list:
    """
    Modifies a HISTORY list record by appending lines that correspond to the same record.
    :param history: (list of str) HISTORY record.
    :return: (list of str) Modified HISTORY record.
    """
    result = []
    for index, record in enumerate(history):
        if index > 0 and record.startswith('P') and record.split(' ')[0] == history[index - 1].split(' ')[0]:
            result[-1] += record[record.index(' ') + 1:]
        else:
            result.append(record)
    return result


def assert_header_has_all_keywords_and_values_of_reference(header: fits.header, header_reference: fits.header) -> None:
    """
    Checks that a header contains all the keys, with same values, than the reference one. Other keywords/values may
    be present and do not affect the result.
    :param header: (fits.header) Header to check.
    :param header_reference: (fits.header) Reference header.
    :return: None.
    """
    for key in header_reference.keys():
        if key in ['DATE', 'CREADATE', 'COMMENT']:
            pass
        elif key == 'HISTORY':
            history = history_unpack(header.get('HISTORY'))
            history_reference = history_unpack(header_reference.get('HISTORY'))
            skip_tags = ['START PARAMETER '] + \
                        [s.split(' ')[0] for s in history_reference if ' EvtFile = ' in s or ' Simput = ' in s]

            for history_record, history_record_reference in zip(history, history_reference):
                if any(history_record_reference.startswith(tag) for tag in skip_tags):
                    split_list = history_record.split(' ')
                    split_list_history = history_record_reference.split(' ')
                    assert split_list[0:2] == split_list_history[0:2]
                else:
                    assert history_record == history_record_reference
        else:
            assert header.get(key) == pytest.approx(header_reference.get(key))
    return None


def assert_generic_object_matches_reference(inp, reference) -> None:
    """
    Checks that a generic object matches reference. This function works for objects whose elements may be iterable, by
    calling itself recursively.
    :param inp: (obj) Object to check.
    :param reference: (obj) Reference object.
    :return: None
    """

    assert type(inp) == type(reference)

    # Checks if the input object is iterable
    if hasattr(inp, '__iter__'):
        assert len(inp) == len(reference)
        # Case of iterable with only one element (i.e. strings
        if len(inp) == 1:
            assert inp[0] == pytest.approx(reference[0])
        else:
            for inp_element, reference_element in zip(inp, reference):
                assert_generic_object_matches_reference(inp_element, reference_element)
    else:
        assert inp == pytest.approx(reference)


def assert_fits_rec_matches_reference(inp: fits.FITS_rec, reference: fits.FITS_rec) -> None:
    """
    Checks that a FITS_rec object (i.e. FITS table) matches reference.
    :param inp: (FITS_rec) FITS table to check.
    :param reference: (FITS_rec) Reference FITS table.
    :return: None
    """

    assert len(inp) == len(reference)
    for rec, rec_reference in zip(inp, reference):
        for obj, obj_reference in zip(rec, rec_reference):
            assert_generic_object_matches_reference(obj, obj_reference)

    return None


def assert_hdu_list_matches_reference(inp: fits.hdu.hdulist.HDUList, reference: fits.hdu.hdulist.HDUList) -> None:
    """
    Checks that a speccube file matches a reference one
    :param inp: (HDUList) HDUList to check.
    :param reference: (HDUList) Reference HDUList.
    :return: None
    """

    for hdu, hdu_reference in zip(inp, reference):
        assert_header_has_all_keywords_and_values_of_reference(hdu.header, hdu_reference.header)
        if type(hdu.data) == fits.FITS_rec:
            assert type(hdu.data) == fits.FITS_rec
            assert_fits_rec_matches_reference(hdu.data, hdu_reference.data)
        else:
            assert bool(np.all(hdu.data == pytest.approx(hdu_reference.data)))

    return None


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
    assert_hdu_list_matches_reference(fits.open(spcubeFile), reference_speccube)

    # Creating a speccube file from the speccube read from the file
    speccube_read = read_speccube(spcubeFile)
    os.remove(spcubeFile)
    if os.path.isfile(spcubeFile2):
        os.remove(spcubeFile2)
    sys_out_write_speccube = write_speccube(speccube_read, spcubeFile2)
    assert sys_out_write_speccube is None
    assert os.path.isfile(spcubeFile2)

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(spcubeFile2), reference_speccube)
    os.remove(spcubeFile2)

    # Creating a SIMPUT file from a speccube
    if os.path.isfile(simputFile):
        os.remove(simputFile)
    sys_out_write_cube2simputfile = cube2simputfile(speccube_read, simputFile)
    assert sys_out_write_cube2simputfile is None
    del speccube_read

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(simputFile), fits.open(referenceSimputFile))

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    sys_out_create_eventlist = create_eventlist(simputFile, 'xrism-resolve-test', 1.e5, evtFile, background=False,
                                                seed=42, verbosity=1)
    assert sys_out_create_eventlist == 0
    os.remove(simputFile)

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(evtFile), fits.open(referenceEvtFile))

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    sys_out_make_pha = make_pha(evtFile, phaFile)
    assert sys_out_make_pha == 0
    os.remove(evtFile)

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(phaFile), fits.open(referencePhaFile))
    os.remove(phaFile)
