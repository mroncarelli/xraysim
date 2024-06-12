import os
from astropy.io import fits

from xraysim.sphprojection.mapping import make_speccube, write_speccube, read_speccube
from xraysim.specutils.sixte import cube2simputfile, create_eventlist, make_pha
from .fitstestutils import assert_hdu_list_matches_reference

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
    write_speccube(speccube_read, spcubeFile2)
    assert os.path.isfile(spcubeFile2)

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(spcubeFile2), reference_speccube)

    # Creating a SIMPUT file from a speccube
    if os.path.isfile(simputFile):
        os.remove(simputFile)
    cube2simputfile(speccube_read, simputFile)
    del speccube_read

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(simputFile), fits.open(referenceSimputFile))

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    create_eventlist(simputFile, 'xrism-resolve-test', 1.e5, evtFile, background=False, seed=42, verbosity=0)
    os.remove(simputFile)

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(evtFile), fits.open(referenceEvtFile))

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    make_pha(evtFile, phaFile)
    os.remove(evtFile)

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(phaFile), fits.open(referencePhaFile))

    os.remove(phaFile)
