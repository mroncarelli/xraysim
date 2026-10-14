"""
This file contains tests designed to test the code in ideal cases, from spectra and map creation to spectral fitting.
The gas in the simulation is considered isothermal and, when present, velocities are assumed with Gaussian
distribution. The physical parameters are chosen randomly with a true random generator: this means that for every
run the physical parameters change and so do the starting values of the fit parameters. The reproducibility is assured
by the initial random seed that is shown in the error message in case of test failure: in order to reproduce the error
one must take not of the seed (i.e. 12345678) and call

pytest --seed 12345678
"""

from astropy import cosmology
from astropy.io import fits
import os
import numpy as np
import pytest
import xspec as xsp

from pyxhydro.pygadgetreader import readhead
from pyxhydro.gadgetutils.convert import vpec2zobs
from pyxhydro.gadgetutils.phys_const import keV2K, keV2erg, pi
from pyxhydro.sixte import simput, sixtesim, makespec
from pyxhydro.sphprojection.mapping import map2d, specmap
from pyxhydro.specutils.specfit import SpecFit
from pyxhydro.specutils.tables import apec_table

from .randomutils import TrueRandomGenerator, globalRandomSeed
from .specfittestutils import assert_fit_results_within_error
from .__shared import referenceDir, snapshotFile, clear_file, testInstrumentName, testInstrumentFOV, snapshotCenter


# Emission table parameters
abund = "aspl"  # Xspec abundance table
nene = 6000
e_min, e_max = 2., 8.  # Energy range [keV]
d_ene = (e_max - e_min) / nene  # Energy bin [keV]
energy = np.linspace(e_min, e_max, num=nene, endpoint=False) + 0.5 * d_ene  # [keV]
dz = 1e-3
nSigmaVel = 5.  # Used to define the redshift range in units of the velocity dispersion defined below

# Map parameters
npix = 50

# Setting physical parameters of the tests with a true random generator
nHMin, nHMax = 0, 0.05
tMin, tMax = 2, 9
zMin, zMax = 0, 1.5
metalMin, metalMax = 0.2, 1
sigmaVMin, sigmaVMax = 50, 500

TRG = TrueRandomGenerator(globalRandomSeed)
errMsg = "Random seed: " + str(TRG.initialSeed)  # Assertion error message if test fails
nH = TRG.uniform(nHMin, nHMax)  # Hydrogen column density [10^22 cm-2]
temp = TRG.uniform(tMin, tMax)  # Gas temperature [keV]
z = int(TRG.uniform(zMin, zMax) / dz) * dz  # redshift [---] (multiple of dz so that its values is in the table)
metal = TRG.uniform(metalMin, metalMax)  # metallicity [Solar]
sigma_v = TRG.uniform(sigmaVMin, sigmaVMax)  # velocity dispersion [km/s]

# Temperature ranges in the table
nt, tMinTable, tMaxTable = 3, temp / 1.5, temp * 1.5

# Calculating z_min and z_max for the spectral table: they correspond to z ± nSigma * sigma_v, rounded with dz
zMinTable = np.floor(vpec2zobs(-nSigmaVel * sigma_v, z, 'km/s') / dz) * dz
zMaxTable = np.ceil(vpec2zobs(nSigmaVel * sigma_v, z, 'km/s') / dz) * dz
nz = int(np.round((zMaxTable - zMinTable) / dz)) + 1
if nz < 3:
    zMinTable -= dz
    zMaxTable += dz
    nz = int(np.round((zMaxTable - zMinTable) / dz)) + 1

# Creating the spectral table
specTable = apec_table(nz, zMinTable, zMaxTable, nt, tMinTable, tMaxTable, nene, e_min, e_max, metal, abund=abund,
                       apecroot='3.0.9')

# Calculating normalization
cosmo = cosmology.FlatLambdaCDM(H0=100., Om0=0.3)
gadget2deg = cosmo.arcsec_per_kpc_comoving(z).to_value() / 3600.  # 1 deg / 1 h^-1 kpc (comoving)
d_C = 1e3 * cosmo.comoving_distance(z).to_value()  # [h^-1 kpc] comoving
mapSize = testInstrumentFOV / gadget2deg # [h^-1 kpc] (comoving)
h_Hubble = readhead(snapshotFile, 'hubble')
map_str = map2d(snapshotFile, 'nenH', 1, center=snapshotCenter, size=mapSize, struct=True, tcut=1e6)
InenHdl = map_str['map'][0, 0]  # [h^3 cn^-5] (comoving)
norm = InenHdl * 1e-14 * h_Hubble ** 3 * (1 + z) ** 3 * mapSize ** 2 / (4 * pi * d_C ** 2) # [10^14 cm^-5] (physical)

# Fit parameters settings
fitParsV0 = (nH, temp, metal, z, 0, norm)
fitPars = (fitParsV0[0], fitParsV0[1], fitParsV0[2], fitParsV0[3], sigma_v, fitParsV0[5])
startPV0 = (nH, TRG.uniform(low=tMin, high=tMax), TRG.uniform(low=metalMin, high=metalMax), z, 0, 1)
startP = (startPV0[0], startPV0[1], startPV0[2], startPV0[3], TRG.uniform(low=sigmaVMin, high=sigmaVMax), startPV0[5])

# Test files
spFile = referenceDir + "sp_file_created_for_ideal_test.simput"
simputFile = referenceDir + "simput_file_created_for_ideal_test.simput"
evtFile = referenceDir + "evt_file_created_for_ideal_test.evt"
phaFile = referenceDir + "pha_file_created_for_ideal_test.pha"


def tbabs_bapec(nh: float, t: float, met: float, redshift: float, broad: float, nrm: float) -> np.ndarray:
    """
    Returns an absorbed (tbabs) bapec spectrum.
    """
    xsp.Xset.chatter = 0
    xsp.Xset.addModelString("APECTHERMAL", "yes")
    xsp.Xset.abund = abund
    xsp.AllModels.setEnergies(str(e_min) + " " + str(e_max) + " " + str(nene) + " lin")

    pars = {1: nh, 2: t}
    for ind in range(28):
        pars[5 + ind] = met
    pars[33] = redshift
    pars[34] = broad
    pars[35] = nrm
    model = xsp.Model('tbabs(bvvapec)', 'test_ideal_run', sourceNum=0)
    model.setPars(pars)
    result = np.array(model.values(0))  # [photons s^-1 cm^-2] (already multiplied by norm)
    xsp.AllModels.setEnergies("reset")
    xsp.AllModels -= model.name
    return result


def test_isothermal_no_velocities():
    """
    An ideal run from Gadget snapshot, assuming isothermal gas with no velocities.
    """

    # Computing reference spectrum
    sp_ref = tbabs_bapec(nH, temp, metal, z, 0, norm)  # [photons s^-1 cm^-2] (already multiplied by norm)


    # Creating the spectral map from the snapshot assuming isothermal gas with Gaussian velocity distribution
    sp_map = specmap(snapshotFile, specTable, testInstrumentFOV, npix, z, center=snapshotCenter, proj='z',
                             tcut=1e6, isothermal=temp * keV2K, nh=nH, novel=True)

    # Checking that the integrated spectrum matches with the reference one
    assert sp_map['energy'] == pytest.approx(energy, rel=1e-6), errMsg  # [keV]

    sp = sp_map['data'].sum(axis=(0, 1)) * d_ene * sp_map['pixel_size'] ** 2  # [photons s^-1 cm^2]
    assert sp.sum() == pytest.approx(sp_ref.sum(), rel=1e-4), errMsg
    assert sp == pytest.approx(sp_ref, rel=1e-3), errMsg  # [photons s^-1 cm^2]

    # Creating the SIMPUT file
    if os.path.isfile(simputFile):
        os.remove(simputFile)
    simput(sp_map, simputFile)
    del sp_map

    # Extracting data from Simput file
    hdu_list = fits.open(simputFile)
    flux = hdu_list[1].data['FLUX']  # [erg/s/cm**2]
    fld = hdu_list[2].data['FLUXDENSITY']  # [photons/s/cm**2/keV]
    assert fld.shape[0] == npix**2, errMsg
    assert fld.shape[1] == nene, errMsg
    sp_simput = np.zeros(nene)
    for ind in range(npix**2):
        assert hdu_list[2].data['ENERGY'][ind, :] == pytest.approx(energy, rel=1e-6), errMsg
        flux_ = np.sum(fld[ind, :] * energy) * d_ene * keV2erg  # [erg s^-1 cm^-2]
        sp_simput += flux[ind] / flux_ * fld[ind] * d_ene  # [photons s^-1 cm^-2]

    assert sp_simput.sum() == pytest.approx(sp_ref.sum(), rel=1e-4), errMsg
    assert sp_simput == pytest.approx(sp_ref, rel=1e-3), errMsg

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    sys_out = sixtesim(simputFile, testInstrumentName, 1e5, evtFile, background=False,
                       seed=42, verbose=0)
    assert sys_out == [0], errMsg
    os.remove(simputFile)

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    makespec(evtFile, phaFile)
    os.remove(evtFile)
    assert os.path.isfile(phaFile), errMsg

    # Fitting the spectrum in the pha file with parameters starting with the right values
    spfit_right_start = SpecFit(phaFile, "tbabs(bapec)")

    fixed_pars = (True, False, False, False, True, False)
    spfit_right_start.run(start=fitParsV0, fixed=fixed_pars, method='cstat', abund=abund, erange=(e_min, e_max),
                          apecroot=(3, 0, 9))

    assert_fit_results_within_error(spfit_right_start, fitParsV0, sigma_tol=2, rel=5e-3, msg=errMsg)
    spfit_right_start.clear()
    del spfit_right_start

    # Fitting the spectrum in the pha file starting with wrong parameters
    spfit_wrong_start = SpecFit(phaFile, "tbabs(bapec)")
    os.remove(phaFile)
    fixed_pars = (True, False, False, False, True, False)
    spfit_wrong_start.run(start=startPV0, fixed=fixed_pars, method='cstat', abund=abund, erange=(e_min, e_max),
                          apecroot=(3, 0, 9))

    assert_fit_results_within_error(spfit_wrong_start, fitParsV0, sigma_tol=3, rel=5e-3, msg=errMsg)
    spfit_wrong_start.clear()


def test_isothermal_gaussian_velocities():
    """
    An ideal run from Gadget snapshot, assuming isothermal gas with Gaussian velocities.
    """

    # Computing reference spectrum
    sp_ref = tbabs_bapec(nH, temp, metal, z, sigma_v, norm)  # [photons s^-1 cm^-2] (already multiplied by norm)

    # Creating the spectral map from the snapshot assuming isothermal gas with Gaussian velocity distribution
    sp_map = specmap(snapshotFile, specTable, testInstrumentFOV, npix, z, center=snapshotCenter, proj='z',
                             tcut=1e6, isothermal=temp * keV2K, nh=nH, gaussvel=(0, sigma_v))

    # Checking that the integrated spectrum matches with the reference one
    assert sp_map['energy'] == pytest.approx(energy, rel=1e-6), errMsg  # [keV]

    sp = sp_map['data'].sum(axis=(0, 1)) * d_ene * sp_map['pixel_size'] ** 2  # [photons s^-1 cm^2]

    # Checking only the sum because with random gaussian velocities bin to bin differences are too high
    assert sp.sum() == pytest.approx(sp_ref.sum(), rel=5e-2), errMsg  # [photons s^-1 cm^2]

    # Creating the SIMPUT file
    if os.path.isfile(simputFile):
        os.remove(simputFile)
    simput(sp_map, simputFile)
    del sp_map

    # Extracting data from Simput file
    hdu_list = fits.open(simputFile)
    flux = hdu_list[1].data['FLUX']  # [erg/s/cm**2]
    fld = hdu_list[2].data['FLUXDENSITY']  # [photons/s/cm**2/keV]
    assert fld.shape[0] == npix**2, errMsg
    assert fld.shape[1] == nene, errMsg
    sp_simput = np.zeros(nene)
    for ind in range(npix**2):
        assert hdu_list[2].data['ENERGY'][ind, :] == pytest.approx(energy, rel=1e-6), errMsg
        flux_ = np.sum(fld[ind, :] * energy) * d_ene * keV2erg  # [erg s^-1 cm^-2]
        sp_simput += flux[ind] / flux_ * fld[ind] * d_ene  # [photons s^-1 cm^-2]

    # Checking only the sum because with random gaussian velocities bin to bin differences are too high
    assert sp_simput.sum() == pytest.approx(sp_ref.sum(), rel=5e-2), errMsg  # [photons s^-1 cm^2]

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    sys_out = sixtesim(simputFile, testInstrumentName, 1e5, evtFile, background=False,
                       seed=42, verbose=0)
    assert sys_out == [0], errMsg
    os.remove(simputFile)

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    makespec(evtFile, phaFile)
    os.remove(evtFile)
    assert os.path.isfile(phaFile), errMsg

    # Fitting the spectrum in the pha file with parameters starting with the right values
    spfit_right_start = SpecFit(phaFile, "tbabs(bapec)")

    fixed_pars = (True, False, False, False, True, False)
    spfit_right_start.run(start=fitPars, fixed=fixed_pars, method='cstat', abund=abund, erange=(e_min, e_max),
                          apecroot=(3, 0, 9))

    assert_fit_results_within_error(spfit_right_start, fitPars, sigma_tol=2, rel=5e-3, msg=errMsg)
    spfit_right_start.clear()
    del spfit_right_start

    # Fitting the spectrum in the pha file starting with wrong parameters
    spfit_wrong_start = SpecFit(phaFile, "tbabs(bapec)")
    os.remove(phaFile)

    fixed_pars = (True, False, False, False, False, False)
    spfit_wrong_start.run(start=startP, fixed=fixed_pars, method='cstat', abund=abund, erange=(e_min, e_max),
                          apecroot=(3, 0, 9))

    assert_fit_results_within_error(spfit_wrong_start, fitPars, sigma_tol=3, rel=5e-3, msg=errMsg)
    spfit_wrong_start.clear()


@pytest.fixture(scope="module", autouse=True)
def on_end_module():
    yield
    clear_file(spFile)
    clear_file(simputFile)
    clear_file(evtFile)
    clear_file(phaFile)
