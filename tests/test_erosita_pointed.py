import os
import warnings

import pytest
from astropy.io import fits

from xraysim.specutils.sixte import create_eventlist, make_pha, version, erosita_ccd_eventfile
from .fitstestutils import assert_hdu_list_matches_reference

sixte_version = version()

inputDir = os.environ.get('XRAYSIM') + '/tests/inp/'
referenceDir = os.environ.get('XRAYSIM') + '/tests/reference_files/'
if sixte_version < (3,):
    referenceDir += 'sixte_v2/'
referenceSimputFile = referenceDir + 'reference.simput'
referenceEvtFile = referenceDir + 'reference_erosita_pointed.evt'
referencePhaFile = referenceDir + 'reference_erosita_pointed.pha'

simputFile = referenceDir + "reference.simput"
evtFile = referenceDir + "evt_file_erosita_pointed_created_for_test.evt"
evtFile_ccdList = []
for ccd in range(1, 8):
    evtFile_ccdList.append(erosita_ccd_eventfile(evtFile, ccd))
phaFile = referenceDir + "pha_file_erosita_pointed_created_for_test.pha"


# Introduced this option to address Issue #12. With the `standard` option the code does not test that the content of
# evtFile and phaFile match the reference as it may fail in some operative systems. With the `complete` option (pytest
# --eventlist complete) the contents are checked and the test fails if they don't match.
@pytest.fixture(scope="session")
def run_type(pytestconfig):
    return pytestconfig.getoption("eventlist").lower()


def test_erosita_pointed(run_type):
    """
    A run of a eROSITA pointed observation from SIMPUT file to pha file, checking that each intermediate step produces 
    a file compatible with reference one.
    """

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    sys_out = create_eventlist(referenceSimputFile, 'erosita', 1.e4, evtFile, background=False,
                               seed=42, verbosity=0)
    assert sys_out == [0, 0]

    # Removing CCD files
    for ccdFile in evtFile_ccdList:
        os.remove(ccdFile)

    if run_type == 'standard':
        # Checking only that the file was created
        assert os.path.isfile(evtFile)
        warnings.warn("Eventlist not checked. Run 'pytest --eventlist complete' to check it.")
    elif run_type == 'complete':
        # Checking that file content matches reference
        assert_hdu_list_matches_reference(fits.open(evtFile), fits.open(referenceEvtFile),
                                          key_skip=('DATE', 'CREADATE', 'COMMENT', 'CHECKSUM'),
                                          history_tag_skip=('START PARAMETER ', ' EvtFile = '))
    else:
        raise ValueError("ERROR in test_erosita_pointed.py: unknown option " + run_type)

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    make_pha(referenceEvtFile, phaFile)
    os.remove(evtFile)

    if run_type == 'standard':
        # Checking only that the file was created
        assert os.path.isfile(phaFile)
        warnings.warn("Pha file not checked. Run 'pytest --eventlist complete' to check it.")
    elif run_type == 'complete':
        # Checking that file content matches reference
        assert_hdu_list_matches_reference(fits.open(phaFile), fits.open(referencePhaFile),
                                          key_skip=('COMMENT'),
                                          history_tag_skip=('START PARAMETER ', ' Spectrum = '))
    else:
        raise ValueError("ERROR in test_erosita_pointed.py: unknown option " + run_type)

    os.remove(phaFile)
