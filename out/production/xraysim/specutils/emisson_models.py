import json
import os

import numpy as np
import xspec as xsp
import pyatomdb
import matplotlib.pyplot as plt


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
        xsp.AllModels.setEnergies(f"{energy.min()} {energy.max()} {len(energy) - 1} lin")
        xsp.Xset.addModelString("APECROOT", "3.0.9")
        self.xspec_model = xsp.Model(model_name)

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

    def calculate_spectrum(self, z: float, temperature: float, elements_index: int, metallicity: np.array,
                           norm: float) -> np.array:
        """
        This class method computes the X-ray emission spectra for a gas particle using Pyxspec.
        :param z: float - redshift for the gas particle
        :param temperature: float - Temperature in keV in for the gas particle
        :param elements_index: int - list of index corresponding to metal species
        :param metallicity: list of float - metallicity array normalized to Anders and Grevesse solar abundance values
        :param norm: float - xspec normalization value, units - 10^-14 cm^-5
        :return: the emission spectra for the gas particle in the units -
                norm * units from xspec module--->(10^-14 cm^-5) * (photons s^-1 cm^3)---->10^-14 photons s^-1 cm-2
        """
        params = [temperature, *metallicity.tolist(), z, norm]
        self.xspec_model.setPars(params)
        self.xspec_model.show()
        result = self.xspec_model.values(0)

        return result


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
            'set_broadening': lambda cmd: self.atomdb_model.set_broadening(thermal_broadening=str2bool(cmd['arg'][0]))
        }
        for command in commands:
            atomdb_settings.get(command['method'], lambda cmd: None)(command)

    def calculate_spectrum(self, z, temperature, elements_index, metallicity, norm) -> np.array:
        """
        This class method computes the X-ray emission spectra for a gas particle using pyatomdb.
        :param z: float - redshift for the gas particle
        :param temperature: float - Temperature in keV in for the gas particle
        :param elements_index: int - list of index corresponding to metal species
        :param metallicity: list of float - metallicity array normalized to Anders and Grevesse solar abundance values
        :param norm: float - atomdb normalization value, units-cm^-5
        :return: norm * units from atomdb module--->(cm^-5) * (photons s^-1 cm^3)---->photons s^-1 cm-2
        """
        self.atomdb_model.set_response(self.energy * (1 + z), raw=True)
        self.atomdb_model.set_abund(elements_index + 1, metallicity[elements_index])
        result = self.atomdb_model.return_spectrum(temperature, log_interp=False) * norm * (1 / (1 + z)) ** 1

        return result


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
        metal_idx = {('apec', 1): [0],
                     ('vvapec', 1): np.arange(3, 31, 1) - 1,
                     ('vvapec', 11): np.array([6, 7, 8, 10, 12, 14, 16, 20, 26]) - 1
                     }

        idx = metal_idx.get((self.json_record['model'], self.json_record['n_metals']))

        np.put(self.json_record['metals_ref'], idx, metal)

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
        chem_species_index = self.set_metals_ref(metallicity)

        result = self.model.calculate_spectrum(z, temperature, chem_species_index, self.json_record['metals_ref'], norm)
        if flag_ene:
            bins = 0.5 * (np.array(self.energy[1:] + np.array(self.energy)[:-1]))
            result = result * bins

        return result


# Testing Line For Checking The Class and setup :

# just for initial test, the actual values depend on the emission measure and cosmological angular distance
xspec_norm = 1
pyatomdb_norm = 1

# for xspec - apec with single metallicity
a = EmissionModels('TheThreeHundred-1', np.linspace(0.1, 10, 2000))

print(a.compute_spectrum(0.2, 0.34, np.array([0.04]), xspec_norm, False))
print(a.compute_spectrum(0.2, 0.63, np.array([0.01]), xspec_norm, False))
print(a.compute_spectrum(0.2, 0.23, np.array([0.07]), xspec_norm, False))

# b = EmissionModels('TheThreeHundred-2', np.linspace(0.1, 10, 1000))

# print(b.compute_spectrum(0.1, 3, np.array([0.05]), xspec_norm, False))
# print(b.compute_spectrum(.2, 0.6, np.linspace(0.2, 0.3, 11), xspec_norm, False))
# print(b.compute_spectrum(.2, 0.2, np.linspace(0.4, 0.5, 11), xspec_norm, False))

# b = EmissionModels('TheThreeHundred-4', np.linspace(0.1, 10, 1000))

# print(b.compute_spectrum(0.1, 2, np.array([0.05]), pyatomdb_norm, False))
# print(b.compute_spectrum(0.1, 2, np.linspace(0.2, 0.3, 11), pyatomdb_norm, False))
# print(b.compute_spectrum(0.1, 2, np.linspace(0.2, 0.3, 11), pyatomdb_norm, False))
# print(pydb1)
# print(xspec1)


# For actual sim

# from gadgetutils.convert import gadgget2xspecnorm
# from xraysim.pygadgetreader import readsnap


# sim_path = '/home/atulit-pc/IdeaProjects/xraysim/tests/inp/snap_Gadget_sample'
# mass = readsnap(sim_path,'U   ', 'gas').shape
# print(mass)
# gad

# a = EmissionModels('TheThreeHundred-4', np.linspace(4, 5, 1000))
# print(a.calculate_spectrum(0.1, 0.54, [0.4], xspec_norm, False))
