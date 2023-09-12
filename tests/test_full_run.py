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
                    result = result and split_list[0:2] == split_list_history[0:2]
                else:
                    result = result and history_record == history_record_reference
        else:
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
