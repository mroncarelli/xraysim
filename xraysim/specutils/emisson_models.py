import json
import os

import numpy as np
import xspec as xsp
import pyatomdb
import matplotlib.pyplot as plt

# Anders and Grevesse abundance table in terms of number fraction
angr_array = np.array(
    [1.00E+00, 9.77E-02, 1.45E-11, 1.41E-11, 3.98E-10, 3.63E-04, 1.12E-04, 8.51E-04, 3.63E-08, 1.23E-04,
     2.14E-06, 3.80E-05, 2.95E-06, 3.55E-05, 2.82E-07, 1.62E-05, 3.16E-07, 3.63E-06, 1.32E-07, 2.29E-06,
     1.26E-09, 9.77E-08, 1.00E-08, 4.68E-07, 2.45E-07, 4.68E-05, 8.32E-08, 1.78E-06, 1.62E-08, 3.98E-08])

atomic_weights = np.array([1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305,
                           26.982, 28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996,
                           54.938, 55.845, 58.933, 58.693, 63.546, 65.38])

# Anders and Grevesse abundance table in terms of mass fraction
angr_array = (angr_array * atomic_weights) / (np.sum(angr_array * atomic_weights))


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
def check_cases():
    xspec_norm = 1
    pyatomdb_norm = 1
    energy_range = np.linspace(0.1, 10, 2000)
    e_bins = 0.5 * (energy_range[1:] + energy_range[:-1])
    # for xspec - apec with single metallicity
    a = EmissionModels('TheThreeHundred-1', energy_range)

    plt.plot(e_bins, a.compute_spectrum(0.2, 0.34, np.array([0.04]), xspec_norm, False),
             label='T=0.34, Z=0.04')
    plt.plot(e_bins, a.compute_spectrum(0.2, 0.63, np.array([0.3]), xspec_norm, False),
             label='T=0.63, Z=0.3')
    plt.plot(e_bins, a.compute_spectrum(0.2, 0.23, np.array([0.09]), xspec_norm, False),
             label='T=0.23, Z=0.09')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Xspec-Apec+single metallicity')
    plt.show()

    # for xspec - vvapec with single metallicity
    b = EmissionModels('TheThreeHundred-2', energy_range)
    plt.plot(e_bins, b.compute_spectrum(0.2, 0.44, np.array([0.04]), xspec_norm, False),
             label='T=0.44, Z=0.04')
    plt.plot(e_bins, b.compute_spectrum(0.2, 0.73, np.array([0.33]), xspec_norm, False),
             label='T=0.73, Z=0.33')
    plt.plot(e_bins, b.compute_spectrum(0.2, 0.93, np.array([0.07]), xspec_norm, False),
             label='T=0.93, Z=0.07')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Xspec-vvapec+single metallicity')
    plt.show()

    # for xspec - vvapec with 11 species metallicity
    c = EmissionModels('TheThreeHundred-3', energy_range)
    plt.plot(e_bins, c.compute_spectrum(0.2, 0.44, np.linspace(0.04, 0.1, 11), xspec_norm, False),
             label='T=0.44, Z=[0.04, 0.1]')
    plt.plot(e_bins, c.compute_spectrum(0.2, 0.73, np.linspace(0.1, 0.7, 11), xspec_norm, False),
             label='T=0.73, Z=[0.1,0.7]')
    plt.plot(e_bins, c.compute_spectrum(0.2, 0.93, np.linspace(0.004, 0.006, 11), xspec_norm, False),
             label='T=0.93, Z=[0.004, 0.006]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Xspec-vvapec+11 species metallicity')
    plt.show()

    # for pyatomdb - vvapec with 11 species metallicity
    d = EmissionModels('TheThreeHundred-4', energy_range)
    plt.plot(e_bins, d.compute_spectrum(0.2, 0.54, np.linspace(0.04, 0.1, 11), pyatomdb_norm, False),
             label='T=0.54, Z=[0.04, 0.1]')
    plt.plot(e_bins, d.compute_spectrum(0.2, 0.83, np.linspace(0.1, 0.7, 11), pyatomdb_norm, False),
             label='T=0.83, Z=[0.1,0.7]')
    plt.plot(e_bins, d.compute_spectrum(0.2, 3, np.linspace(0.004, 0.006, 11), pyatomdb_norm, False),
             label='T=3, Z=[0.004, 0.006]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('pyatomdb-vvapec+11 species metallicity')
    plt.show()


check_cases()
# For actual sim

# from gadgetutils.convert import gadgget2xspecnorm
# from xraysim.pygadgetreader import readsnap


# sim_path = '/home/atulit-pc/IdeaProjects/xraysim/tests/inp/snap_Gadget_sample'
# mass = readsnap(sim_path,'U   ', 'gas').shape
# print(mass)
# gad

# a = EmissionModels('TheThreeHundred-4', np.linspace(4, 5, 1000))
# print(a.calculate_spectrum(0.1, 0.54, [0.4], xspec_norm, False))
