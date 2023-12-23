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


def process_simulation(sim_path, model_name, model_index, sim_type, num_regions=1000):
    if sim_type == 'Gizmo':
        sim_metal = readsnap(sim_path, 'Metallicity', 'gas')[:, 2:]
    elif sim_type == 'Gadget':
        sim_metal = np.reshape(readsnap(sim_path, 'Z   ', 'gas'), (-1, 1))

    sim_temp = np.array(readtemperature(sim_path, units='KeV'), dtype=float)
    sim_z = 0 if readhead(sim_path, 'redshift') < 0 else readhead(sim_path, 'redshift')

    indices = np.where((sim_temp > 0.08))[0]
    sim_temp = sim_temp[indices]
    sim_metal = sim_metal[indices]

    energies_array = np.linspace(0.1, 10, 2000)

    sim_emission_model = EmissionModels(model_name=model_name, energy=energies_array)

    spectrum = []
    for i in tqdm(range(num_regions), desc="Processing Regions"):
        spectrum.append(sim_emission_model.compute_spectrum(sim_z, sim_temp[i], sim_metal[i], 1, False))

    assert np.all([len(i) == len(energies_array) - 1 for i in spectrum])

    if (model_index == 3) or (model_index == 2):
        assert isinstance(sim_emission_model.model, XspecModel)
    elif (model_index == 4) or (model_index == 5):
        assert isinstance(sim_emission_model.model, AtomdbModel)

    return spectrum


def test_sample_gizmo():
    sim_path = '/home/atulit-pc/IdeaProjects/xraysim/tests/inp/snap_sample.hdf5'

    spectrum_xspec = process_simulation(sim_path, 'TheThreeHundred-3', 3, 'Gizmo')
    spectrum_atomdb = process_simulation(sim_path, 'TheThreeHundred-4', 4, 'Gizmo')

    assert np.all([spec_xsp == pytest.approx(spec_atmdb) for spec_xsp, spec_atmdb in
                   zip(np.sum(spectrum_atomdb, axis=1), np.sum(spectrum_xspec, axis=1))])


def test_sample_gadget():
    sim_path = '/home/atulit-pc/IdeaProjects/xraysim/tests/inp/snap_Gadget_sample'

    spectrum_xspec = process_simulation(sim_path, 'TheThreeHundred-2', 2, 'Gadget')
    spectrum_atomdb = process_simulation(sim_path, 'TheThreeHundred-5', 5, 'Gadget')

    assert np.all([spec_xsp == pytest.approx(spec_atmdb) for spec_xsp, spec_atmdb in
                   zip(np.sum(spectrum_atomdb, axis=1), np.sum(spectrum_xspec, axis=1))])
