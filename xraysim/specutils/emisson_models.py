import json
import os

import numpy as np
import pytest
import xspec as xsp
import pyatomdb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
# For sim Testing

from readgadget.readgadget import readsnap
from readgadget.readgadget import readhead
from gadgetutils.readspecial import readtemperature
from tqdm.auto import tqdm

from gadgetutils.convert import gadget2xspecnorm

# Anders and Grevesse abundance table in terms of number fraction--->angr in mass fraction
# Anders and Grevesse abundance table in terms of mass fraction
# angr_array = (angr_array * atomic_weights) / (np.sum(angr_array * atomic_weights))

Abundance_Table = {
    'Symbols': np.array(['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                         'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']),
    'AbundanceTable': np.array([7.06534941e-01, 2.74100525e-01, 7.05343364e-11, 8.90682759e-11,
                                3.01565655e-09, 3.05603908e-03, 1.09960388e-03, 9.54323263e-03,
                                4.83378824e-07, 1.73980024e-03, 3.44846527e-05, 6.47369649e-04,
                                5.57916578e-05, 6.98837004e-04, 6.12236919e-06, 3.64042128e-04,
                                7.85193027e-06, 1.01642369e-04, 3.61744208e-06, 6.43301606e-05,
                                3.97037310e-08, 3.27796178e-06, 3.57066498e-07, 1.70564600e-05,
                                9.43435125e-06, 1.83190632e-03, 3.43680576e-06, 7.32283794e-05,
                                7.21566472e-07, 1.82390032e-06])
}

Z_solar = np.sum(Abundance_Table['AbundanceTable'][2:])


def str2bool(v):
    """

    :param v: 
    :return:
    """
    if v == 'True':
        return True
    elif v == 'False':
        return False


models_config_file = os.path.join(os.path.dirname(__file__), 'em_reference.json')
with open(models_config_file) as file:
    json_data = json.load(file)


class XspecModel:
    def __init__(self, model_name: str, energy: np.array) -> None:
        """
        This Xspec Model constructor sets the PyXspec model energies, initializes the APEC version, and configures the
        emission model based on the string variable model_name. It can be either APEC or VVAPEC.
        :param model_name: str - type of X-ray Emission Model apec/vvapec
        :param energy: list of float - Represents the range of energy values in KeV for spectrum calculation in pyxspec.
        """
        self.xspec_model_name = model_name
        xsp.AllModels.setEnergies(f"{energy.min()} {energy.max()} {len(energy) - 1} lin")
        xsp.Xset.addModelString("APECROOT", "3.0.9")
        self.xspec_model = xsp.Model(self.xspec_model_name)

    # doesn't change the object itself, that's why we have this warning
    def set_xspec_commands(self, commands: dict) -> None:
        """
        This class method set up all the commands for the XSPEC model
        :param commands: dict : dictionary of commands specific to XSPEC which are set up iteratively inside a loop
        :return: None
        """
        xspec_settings = {
            'abund': lambda cmd: setattr(xsp.Xset, cmd['method'], cmd['arg']),
            'addModelString': lambda cmd: xsp.Xset.addModelString(cmd['arg'][0], cmd['arg'][1]),
        }

        for command in commands:
            xspec_settings.get(command['method'], lambda cmd: None)(command)

    def calculate_spectrum(self, z: float, temperature: float, metallicity: np.array, element_index,
                           norm: float) -> np.array:
        """
        This class method computes the X-ray emission spectra for a gas particle using Pyxspec.
        :param z: float - redshift for the gas particle
        :param temperature: float - Temperature in keV in for the gas particle
        :param metallicity: list of float - metallicity array normalized to Anders and Grevesse solar abundance values
        :param element_index: list of abundance to set
        :param norm: float - xspec normalization value, units - 10^-14 cm^-5
        :return: the emission spectra for the gas particle in the units -
                norm * units from xspec module--->(10^-14 cm^-5) * (photons s^-1 cm^3)---->10^-14 photons s^-1 cm-2
        """
        params = {1: temperature, 32: z, 33: norm} if self.xspec_model_name == 'vvapec' \
            else {1: temperature, 3: z, 4: norm} if self.xspec_model_name == 'apec' \
            else None

        if params is not None:
            params.update({i + 2: metallicity[i] for i in element_index.tolist()})
        # print(params)
        self.xspec_model.setPars(params)

        # self.xspec_model.show()
        result = self.xspec_model.values(0)

        return np.array(result)


class AtomdbModel:

    def __init__(self, model_name: str, energy: np.array) -> None:
        """
        This AtomDB Model constructor sets the AtomDB model energies and set up the AtomDB CIESession.
        :param model_name:str - type of X-ray Emission Model, only vvapec
        :param energy:list of float - Represents the range of energy values in KeV for spectrum calculation in pyatomDB
        """
        self.atomdb_model = pyatomdb.spectrum.CIESession()
        self.energy = energy

    def set_atomdb_commands(self, commands: dict) -> None:
        """
        This class method set up all the commands for the pyatomdb model
        :param commands: dict : dictionary of commands specific to pyatomdb which are set up iteratively inside a loop
        :return: None
        """
        atomdb_settings = {
            'abundset': lambda cmd: setattr(self.atomdb_model, cmd['method'], cmd['arg']),
            'do_eebrems': lambda cmd: setattr(self.atomdb_model, cmd['method'], str2bool(cmd['arg'][0])),
            'set_broadening': lambda cmd: self.atomdb_model.set_broadening(thermal_broadening=str2bool(cmd['arg'][0])),
        }
        for command in commands:
            atomdb_settings.get(command['method'], lambda cmd: None)(command)

    def calculate_spectrum(self, z, temperature, metallicity, element_index, norm) -> np.array:
        """
        This class method computes the X-ray emission spectra for a gas particle using pyatomdb.
        :param z: float - redshift for the gas particle
        :param temperature: float - Temperature in keV in for the gas particle
        :param metallicity: list of float - metallicity array normalized to Anders and Grevesse solar abundance values
        :param element_index:index of elements to set
        :param norm: float - atomdb normalization value, units-cm^-5
        :return: norm * units from atomdb module--->(cm^-5) * (photons s^-1 cm^3)---->photons s^-1 cm-2
        """
        self.atomdb_model.set_response(self.energy * (1 + z), raw=True)
        # print(element_index,metallicity[element_index])
        self.atomdb_model.set_abund(element_index + 1, metallicity[element_index])
        result = self.atomdb_model.return_spectrum(temperature, log_interp=False, teunit='KeV') * norm * (
                1 / (1 + z)) ** 1

        # This extra condition is due the difference between the two libraries when no corresponding spectrum is found
        # in table. The xspec returns an array of 0.0 with size equivalent to number of energy bins whereas atomdb
        # returns simply zero. Therefore, to be consistent next part is required.
        result = np.zeros(len(self.energy) - 1, dtype=np.float32) if isinstance(result, float) else result
        # to match up with xspec
        return np.array(result) * 1E14


class EmissionModels:
    def __init__(self, model_name: str, energy: np.array):
        """
        The EmissionModels constructor first searches for the model 'name' in the JSON record. Once it successfully
        locates the name, it initializes the X-Ray library along with its corresponding commands
        :param model_name: str - Specifies the model name as 'TheThreeHundred-1/2/3/4'.
        :param energy: list of float - Represents the range of energy values in KeV for spectrum calculation.
        """
        self.json_record = next((i for i in json_data if i['name'] == model_name), None)

        if self.json_record is None:
            raise ValueError(f"Model with name '{model_name}' not found in the json file.")

        self.json_record['metals_ref'] = np.array(self.json_record['metals_ref'], dtype=float)
        self.json_record['chemical_elements'] = np.array(self.json_record['chemical_elements'], dtype=str)
        self.energy = energy

        if self.json_record['code'] == 'xspec':
            self.model = XspecModel(self.json_record['model'], self.energy)
            self.model.set_xspec_commands(self.json_record['xset'])

        elif self.json_record['code'] == 'atomdb':
            self.model = AtomdbModel(self.json_record['model'], self.energy)
            self.model.set_atomdb_commands(self.json_record['xset'])

        else:
            raise ValueError(f"Library '{self.json_record['code']}' not supported.")

    def set_metals_ref(self, metal: np.array) -> np.array:
        """
        This class method takes the metallicity array as input and assigns it to metals_ref based on the model type and
        the number of chemical species. This information can later be utilized in the corresponding X-ray library for
        spectrum calculation.
        :param metal: array of float - metallicity corresponding to each sph gas particle
        :return: The chemical species index which is essential for the PyAtomDB library (for the set abundance).
        """
        if self.json_record['n_metals'] == len(metal):
            metal_idx = {('apec', False): [0],
                         ('vvapec', False): np.nonzero(np.in1d(Abundance_Table['Symbols'],
                                                               self.json_record['chemical_elements']))[0] + 1 - 1,
                         ('vvapec', True):
                             np.nonzero(np.in1d(Abundance_Table['Symbols'], self.json_record['chemical_elements']))[
                                 0] + 1 - 1
                         }

            idx = metal_idx.get((self.json_record['model'], self.json_record['n_metals'] > 1))
            np.put(self.json_record['metals_ref'], idx, metal)
            # print(Abundance_Table['Symbols'][idx], idx, '\n')

            self.json_record['metals_ref'][idx] = self.json_record['metals_ref'][idx] / \
                                                  Abundance_Table['AbundanceTable'][idx] \
                if self.json_record['n_metals'] > 1 \
                else self.json_record['metals_ref'][idx] / Z_solar
        else:
            raise ValueError(f"wrong setting n_metals is'{self.json_record['n_metals']} and len of metal'{len(metal)}.")

        return idx

    def compute_spectrum(self, z: float, temperature: float, metallicity: np.array, norm: float,
                         flag_ene: bool = False) -> np.array:
        """
        This class method return the emission spectra for each gas particle at the given energy range using the gas
        physical properties.
        :param z: float - redshift for each gas particle
        :param temperature: float - temperature in KeV for gas particle
        :param metallicity: list of float - Metallicity values for gas particles, normalized to Anders & Grevesse.
        :param norm: float - xspec and pyatomdb normalization factor
        :param flag_ene: bool - conversion -----
        :return:
        """
        chem_idx = self.set_metals_ref(metallicity)
        # print(self.json_record['metals_ref'])

        result = self.model.calculate_spectrum(z, temperature, self.json_record['metals_ref'], chem_idx, norm)
        if flag_ene:
            bins = 0.5 * (np.array(self.energy[1:] + np.array(self.energy)[:-1]))
            result = result * bins

        return np.array(result)


# Testing Line For Checking The Class and setup :
def test_gizmo_sample():
    sim_path = '/home/atulit-pc/IdeaProjects/xraysim/tests/inp/snap_sample.hdf5'
    sim_metal = readsnap(sim_path, 'Metallicity', 'gas')[:, 2:]
    sim_temp = np.array(readtemperature(sim_path, units='KeV'), dtype=float)
    sim_z = 0 if readhead(sim_path, 'redshift') < 0 else readhead(sim_path, 'redshift')
    print(sim_z)

    indices = np.where((sim_temp > 0.08))[0]
    sim_temp = sim_temp[indices]

    sim_metal = sim_metal[indices]

    energies_array = np.linspace(0.1, 10, 2000)

    sim_emission_model_xspec = EmissionModels(model_name='TheThreeHundred-3', energy=energies_array)
    sim_emission_model_atomdb = EmissionModels(model_name='TheThreeHundred-4', energy=energies_array)

    spectrum_xspec = []
    for i in tqdm(range(1000), desc="Processing Regions"):
        spectrum_xspec.append(sim_emission_model_xspec.compute_spectrum(sim_z, sim_temp[i], sim_metal[i], 1, False))

    spectrum_atomdb = []
    for i in tqdm(range(1000), desc="Processing Regions"):
        spectrum_atomdb.append(sim_emission_model_atomdb.compute_spectrum(sim_z, sim_temp[i], sim_metal[i], 1, False))

    for (i, j) in zip(spectrum_atomdb[0:1000:100], spectrum_xspec[0:1000:100]):
        # Plot for spectrum_atomdb
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(0.5 * (energies_array[1:] + energies_array[:-1]), i, ls='-', label='spectrum_atomdb')

        # Plot for spectrum_xspec
        axes[1].plot(0.5 * (energies_array[1:] + energies_array[:-1]), j, ls='-', label='spectrum_xsp')
        for ax in axes:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
        plt.show()


# test_gizmo_sample()
def check_scaling(mod_name):
    energies_array = np.linspace(0.1, 10, 2000)

    redshift = 0
    temperature = [0.7, 1, 9]  # KeV
    metallicity = np.linspace(1E-5, 1E-1, 100).reshape((-1, 1))

    xsp_emission_model = EmissionModels(model_name=mod_name, energy=energies_array)

    spectrum = np.zeros((len(temperature), len(metallicity), len(energies_array) - 1), dtype=np.float32)
    for i in range(len(temperature)):
        for j in tqdm(range(len(metallicity))):
            spectrum[i, j, :] = xsp_emission_model.compute_spectrum(
                redshift, temperature[i], metallicity[j], 1, False)

    for i in range(len(temperature)):
        coefficient = np.array(
            [spearmanr(spectrum[i, :, j], metallicity.flatten())[0] for j in range(len(energies_array) - 1)])
        if not (np.all(coefficient > 0.80)):
            print(i, coefficient[np.where(coefficient < 0.99)[0]])


def test_metallicity_and_bins():
    print('Xspec\n')
    check_scaling(mod_name='TheThreeHundred-2')
    print('AtomDB\n')
    check_scaling(mod_name='TheThreeHundred-5')


# test_metallicity_and_bins()
