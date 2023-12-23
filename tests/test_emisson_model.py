import pytest
import numpy as np
import os

from tqdm.auto import tqdm

from xraysim.readgadget.readgadget import readsnap
from xraysim.readgadget.readgadget import readhead
from xraysim.gadgetutils.readspecial import readtemperature
from xraysim.specutils.emisson_models import EmissionModels
from xraysim.specutils.emisson_models import XspecModel
from xraysim.specutils.emisson_models import AtomdbModel
import matplotlib.pyplot as plt

def test_sample_gadget():
    sim_path = '/home/atulit-pc/IdeaProjects/xraysim/tests/inp/snap_Gadget_sample'
    sim_metal = readsnap(sim_path, 'Z   ', 'gas')
    sim_temp = np.array(readtemperature(sim_path, units='KeV'), dtype=float)
    sim_z = 0 if readhead(sim_path, 'redshift') < 0 else readhead(sim_path, 'redshift')

    indices = np.where((sim_temp > 0.08))[0]
    sim_temp = sim_temp[indices]
    sim_metal = sim_metal[indices]

    energies_array = np.linspace(0.1, 10, 3000)
    sim_emission_model_xspec = EmissionModels(model_name='TheThreeHundred-2', energy=energies_array)
    spectrum_xspec = []
    for i in tqdm(range(1000), desc="Processing Regions"):
        spectrum_xspec.append(sim_emission_model_xspec.compute_spectrum(sim_z, sim_temp[i], [sim_metal[i]], 1, False))

    sim_emission_model_atomdb = EmissionModels(model_name='TheThreeHundred-5', energy=energies_array)
    spectrum_atomdb = []
    for i in tqdm(range(1000), desc="Processing Regions"):
        spectrum_atomdb.append(sim_emission_model_atomdb.compute_spectrum(sim_z, sim_temp[i], [sim_metal[i]], 1, False))

    assert np.all([len(i) == len(energies_array) - 1 for i in spectrum_xspec])
    assert isinstance(sim_emission_model_xspec.model, XspecModel)

    assert np.all([len(i) == len(energies_array) - 1 for i in spectrum_atomdb])
    assert isinstance(sim_emission_model_atomdb.model, AtomdbModel)

    assert np.all([spec_xsp == pytest.approx(spec_atmdb) for spec_xsp, spec_atmdb in zip(np.sum(spectrum_atomdb,axis=1),np.sum(spectrum_xspec,axis=1))])

