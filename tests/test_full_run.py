import warnings
import pytest

from pyxhydro.sixte import instruments, simput, sixtesim, makespec
from pyxhydro.sphprojection.mapping import specmap, write_specmap, read_specmap
from pyxhydro.specutils.specfit import *
from .fitstestutils import assert_hdu_list_matches_reference
from .specfittestutils import assert_specfit_has_coherent_properties
from .__shared import *

testInstrument = instruments.get(testInstrumentName)

spMapFile = referenceDir + "spmap_file_created_for_test.spmap"
spMapFile2 = referenceDir + "spmap_file_created_for_test_2.spmap"
simputFile = referenceDir + "simput_file_created_for_test.simput"
evtFile = referenceDir + "evt_file_created_for_test.evt"
phaFile = referenceDir + "pha_file_created_for_test.pha"
spfFile = referenceDir + "spf_file_created_for_test.spf"

# Introduced this option to address Issue #12. With the `standard` option the code does not test that the content of
# evtFile and phaFile match the reference as it may fail in some operative systems. With the `complete` option (pytest
# --eventlist complete) the contents are checked and the test fails if they don't match.
@pytest.fixture(scope="session")
def run_type(pytestconfig) -> str:
    return pytestconfig.getoption("eventlist").lower()


@pytest.mark.filterwarnings("ignore")
def test_full_run(run_type):
    """
    A full run from Gadget snapshot to spf file, checking that each intermediate step produces a file compatible with
    reference one.
    """

    # Creating a spectral map file from a calculated spectral map
    spmap_calculated = specmap(snapshotFile, referenceSpecTableFile, testInstrumentFOV, 25, redshift=0.1,
                                        center=snapshotCenter, proj='z', tcut=1e6, nh=0.01, nsample=1)
    if os.path.isfile(spMapFile):
        os.remove(spMapFile)
    write_specmap(spmap_calculated, spMapFile)
    assert os.path.isfile(spMapFile)
    del spmap_calculated

    reference_spmap = fits.open(referenceSpmapFile)

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(spMapFile), reference_spmap, tol=5e-5, warn_on_keys=True)

    # Creating a spmap file from the spectral-map read from the file
    spmap_read = read_specmap(spMapFile)
    os.remove(spMapFile)
    if os.path.isfile(spMapFile2):
        os.remove(spMapFile2)
    write_specmap(spmap_read, spMapFile2)
    assert os.path.isfile(spMapFile2)

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(spMapFile2), reference_spmap, tol=5e-5, warn_on_keys=True)
    os.remove(spMapFile2)

    # Creating a SIMPUT file from a spectral map
    if os.path.isfile(simputFile):
        os.remove(simputFile)
    simput(spmap_read, simputFile)
    del spmap_read

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(simputFile), fits.open(referenceSimputFile), tol=5e-5,
                                      warn_on_keys=True)

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    sys_out = sixtesim(referenceSimputFile, testInstrumentName, 1.e5, evtFile,
                       background=False, seed=42, verbose=0)
    assert sys_out == [0]
    os.remove(simputFile)

    if run_type == 'standard':
        # Checking only that the file was created
        assert os.path.isfile(evtFile)
        warnings.warn("Eventlist not checked. Run 'pytest --eventlist complete' to check it.")
    elif run_type == 'complete':
        # Checking that file content matches reference
        assert_hdu_list_matches_reference(fits.open(evtFile), fits.open(referenceEvtFile), tol=5e-5,
                                          key_skip=('DATE', 'CREADATE', 'COMMAND', 'COMMENT'),
                                          history_tag_skip=('START PARAMETER ', ' EvtFile = '),
                                          warn_on_keys=True)
    else:
        raise ValueError("ERROR in test_full_run.py: unknown option " + run_type)

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    makespec(referenceEvtFile, phaFile, rsppath=testInstrument.path)
    os.remove(evtFile)
    assert os.path.isfile(phaFile)

    if run_type == 'standard':
        warnings.warn("Pha file not checked. Run 'pytest --eventlist complete' to check it.")
    elif run_type == 'complete':
        # Checking that file content matches reference
        assert_hdu_list_matches_reference(fits.open(phaFile), fits.open(referencePhaFile), tol=5e-5,
                                          key_skip=('COMMAND', 'COMMENT'),
                                          history_tag_skip=('START PARAMETER ', ' Spectrum = '),
                                          warn_on_keys=True)
    else:
        raise ValueError("ERROR in test_full_run.py: unknown option " + run_type)

    # Getting rmf and arf from test file before removing it as its path may change with respect to the reference file
    rmf_file = fits.open(phaFile)[1].header.get('RESPFILE')
    arf_file = fits.open(phaFile)[1].header.get('ANCRFILE')
    os.remove(phaFile)

    # Fitting the spectrum in the pha file
    specfit = SpecFit(referencePhaFile, "tbabs(bapec)", respFile=rmf_file, arfFile=arf_file)
    assert_specfit_has_coherent_properties(specfit)
    startPars = (0.01, 6., 0.3, 0.1, 1000., 1e-3)
    fixedPars = (True, False, True, False, False, False)
    specfit.run(start=startPars, fixed=fixedPars, method='cstat', abund='aspl', erange=(3, 7), apecroot=(3, 0, 9))
    assert_specfit_has_coherent_properties(specfit)
    specfit.save(spfFile, overwrite=True)
    specfit.clear()
    assert os.path.isfile(spfFile)
    assert_hdu_list_matches_reference(fits.open(spfFile), fits.open(referenceSpfFile), tol=1e-4, warn_on_keys=True)
    os.remove(spfFile)


@pytest.fixture(scope="module", autouse=True)
def on_end_module():
    yield
    clear_file(spMapFile)
    clear_file(spMapFile2)
    clear_file(simputFile)
    clear_file(evtFile)
    clear_file(phaFile)
    clear_file(spfFile)
