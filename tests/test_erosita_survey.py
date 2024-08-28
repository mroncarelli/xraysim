import os
import warnings

import pytest
from astropy.io import fits

from xraysim.specutils.sixte import create_eventlist, make_pha
from .fitstestutils import assert_hdu_list_matches_reference

inputDir = os.environ.get('XRAYSIM') + '/tests/inp/'
referenceDir = os.environ.get('XRAYSIM') + '/tests/reference_files/'
referenceSimputFile = referenceDir + 'reference.simput'
referenceGTIFile = referenceDir + 'reference_erosita_survey.gti'
referenceEvtFile = referenceDir + 'reference_erosita_survey.evt'
referencePhaFile = referenceDir + 'reference_erosita_survey.pha'

simputFile = referenceDir + "reference.simput"
GTIFile = referenceDir + "evt_file_erosita_survey_created_for_test.gti"
evtFile = referenceDir + "evt_file_erosita_survey_created_for_test.evt"
evtFile_ccdList = []
for ccd in ('1', '2', '3', '4', '5', '6', '7'):
    evtFile_ccdList.append(os.path.splitext(evtFile)[0] + '_ccd' + ccd + '_evt.fits')
phaFile = referenceDir + "pha_file_erosita_survey_created_for_test.pha"


# Introduced this option to address Issue #12. With the `standard` option the code does not test that the content of
# evtFile and phaFile match the reference as it may fail in some operative systems. With the `complete` option (pytest
# --eventlist complete) the contents are checked and the test fails if they don't match.
@pytest.fixture(scope="session")
def run_type(pytestconfig):
    return pytestconfig.getoption("eventlist").lower()


def test_erosita_survey(run_type):
    """
    A run of a eROSITA survey observation from SIMPUT file to pha file, checking that each intermediate step produces
    a file compatible with reference one.
    """

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(GTIFile):
        os.remove(GTIFile)
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    sys_out = create_eventlist(simputFile, 'erass1', None, evtFile, background=False, seed=42, verbosity=0)
    assert sys_out == [0, 0, 0]

    # Checking GTI file
    assert_hdu_list_matches_reference(fits.open(GTIFile), fits.open(referenceGTIFile))
    os.remove(GTIFile)

    # Removing CCD files
    for ccdFile in evtFile_ccdList:
        os.remove(ccdFile)

    if run_type == 'standard':
        # Checking only that the file was created
        assert os.path.isfile(evtFile)
        warnings.warn("Eventlist not checked. Run pytest --eventlist complete to check it.")
    elif run_type == 'complete':
        # Checking that file content matches reference
        assert_hdu_list_matches_reference(fits.open(evtFile), fits.open(referenceEvtFile))
    else:
        raise ValueError("ERROR in test_erosita_survey.py: unknown option " + run_type)

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    make_pha(referenceEvtFile, phaFile)
    os.remove(evtFile)

    if run_type == 'standard':
        # Checking only that the file was created
        assert os.path.isfile(phaFile)
        warnings.warn("Pha file not checked. Run pytest --eventlist complete to check it.")
    elif run_type == 'complete':
        # Checking that file content matches reference
        assert_hdu_list_matches_reference(fits.open(phaFile), fits.open(referencePhaFile))
    else:
        raise ValueError("ERROR in test_erosita_survey.py: unknown option " + run_type)

    os.remove(phaFile)
