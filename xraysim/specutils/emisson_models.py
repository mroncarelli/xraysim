import json
import os

import numpy as np
import xspec as xsp
import pyatomdb
import pyspex as spex

from gadgetutils.phys_const import kpc2cm, Xp, m_p, Msun2g, Mpc2cm
from astropy.cosmology import FlatLambdaCDM

# For sim Testing
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from readgadget.readgadget import readsnap
from readgadget.readgadget import readhead
from gadgetutils.readspecial import readtemperature
from tqdm.auto import tqdm
from gadgetutils.convert import gadget2xspecnorm

# Cosmological parameters
h0 = 67.77
omega_m = 0.27
omega_l = 0.73
omega_r = 0.0

cosmo = FlatLambdaCDM(H0=h0, Om0=omega_m)

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


def spex_norm(mi, rhoi, nei, h):
    conversion_factor = (1E10 * Msun2g * Xp / m_p) ** 2 * (h) * (kpc2cm ** -3)
    return mi * rhoi * nei * conversion_factor


def str2bool(v):
    """

    :param v: 
    :return:
    """
    if v == 'True':
        return True
    elif v == 'False':
        return False


current_directory = os.getcwd()

models_config_file = os.path.join(current_directory, 'em_reference.json')

print(models_config_file)
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
        self.atomdb_model_name = model_name
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


class SpexModel:

    def __init__(self, model_name: str, energy: np.array) -> None:
        # model name
        self.spex_model_name = model_name

        # initiate the spex session
        self.spex_model = spex.Session()

        # cosmo settings
        self.spex_model.dist_cosmo(h0, omega_m, omega_l, omega_r)

        # energy grid - not in logspace
        self.spex_model.egrid(energy.min(), energy.max(), len(energy) - 1, 'kev', False)

        # CIE model with cosmological redshift on to it
        self.spex_model.mod.comp_new('red', isect=1)
        self.spex_model.mod.comp_new('cie', isect=1)
        self.spex_model.com_rel(1, 2, np.array([1]))

    def set_spex_commands(self, commands: dict) -> None:
        """
        This class method set up all the commands for the spex model
        :param commands: dict : dictionary of commands specific to pyatomdb which are set up iteratively inside a loop
        :return: None
        """
        spex_settings = {
            'abundset': lambda cmd: self.spex_model.abun(cmd['arg'])
        }
        for command in commands:
            spex_settings.get(command['method'], lambda cmd: None)(command)

    def calculate_spectrum(self, z, temperature, metallicity, element_index, norm) -> np.array:
        element_index = [i.zfill(2) for i in element_index.astype(str)]

        # this is required for cosmological distance calculation which is required in final spectrum calculation
        self.spex_model.dist(1, z, 'z')

        # this is required for applying redshift to CIE model - energy shift and S(E) - redshift correction
        self.spex_model.par(1, 1, 'z', z)
        # the redshift is cosmo for flag - 0 and peculiar velocity for flag - 1, for now let us consider 0
        self.spex_model.par(1, 1, 'flag', 0)

        # this nenhV, I have to set it according to spex unit, spex just take nenhV and in multiple of E64 m-3
        self.spex_model.par(1, 2, 'norm', norm)  # 1 * E64 m-3
        self.spex_model.par(1, 2, 't', temperature)
        # we want interpolation on temperature for spectrum to be linear, therefore zero
        self.spex_model.par(1, 2, 'logt', 0)
        self.spex_model.par_show()
        for element_index_i, abund_i in zip(element_index, metallicity):
            self.spex_model.par(1, 2, element_index_i, abund_i)

        self.spex_model.calc()
        self.spex_model.mod_spectrum.get(1)

        # The spectrum given by spex is in ph bin-1 s-1 m-2. In order to compare with xspec cgs unit is required
        # i.e. ph bin-1 cm-2 s-1 (hence, multiplied by 1E-4)
        return np.array(self.spex_model.mod_spectrum.spectrum) * 1E-4


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

        elif self.json_record['code'] == 'spex':
            self.model = SpexModel(self.json_record['model'], self.energy)
            self.model.set_spex_commands(self.json_record['xset'])

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("removing dummy files")
        dir_name = "/home/atulit-pc/IdeaProjects/xraysim/xraysim/specutils/"
        test = os.listdir(dir_name)

        for item in test:
            if item.endswith(".dum"):
                os.remove(os.path.join(dir_name, item))

# testing line for gadget




sim_path = '/home/atulit-pc/MockXray/MockSpectra_GadgetX/Simulation/snap_119'
sim_metal = readsnap(sim_path, 'Z   ', 'gas', dtype=float)
sim_mass = readsnap(sim_path, 'MASS', 'gas', dtype=float)
sim_density = readsnap(sim_path, 'RHO ', 'gas', dtype=float)
sim_ne = readsnap(sim_path, 'NE  ', 'gas', dtype=float)

sim_temp = np.array(readtemperature(sim_path, units='KeV'), dtype=float)
sim_z = 0 if readhead(sim_path, 'redshift') < 0 else readhead(sim_path, 'redshift')

indices = np.where((sim_temp > 0.08) & (sim_metal > 0.0))[0]
sim_temp = sim_temp[indices]
sim_metal = sim_metal[indices]
sim_mass = sim_mass[indices]
sim_density = sim_density[indices]
sim_ne = sim_ne[indices]

print(cosmo.angular_diameter_distance(sim_z))
xsp_norm = np.array(
    gadget2xspecnorm(sim_mass, sim_density, 1E3 * cosmo.angular_diameter_distance(sim_z).value, h0 / 100, sim_ne),
    dtype=float)
sim_metal = np.reshape(sim_metal, (-1, 1))
print(xsp_norm)
spx_norm = spex_norm(sim_mass, sim_density, sim_ne, h0 / 100) * 1E6
spx_norm = spx_norm / 1E64
erange = np.linspace(.1, 10, 9901)
ebins_mid = 0.5 * (erange[1:] + erange[:-1])
print(spx_norm)

xsp_spectrum = []
spx_spectrum = []
with EmissionModels('TheThreeHundred-2', erange) as em:
    xsp_spectrum.append(em.compute_spectrum(sim_z, sim_temp[100], sim_metal[100], xsp_norm[100], False))


with EmissionModels('TheThreeHundred-6', erange) as em:
    spx_spectrum.append(em.compute_spectrum(sim_z, sim_temp[100], sim_metal[100], spx_norm[100], False))

plt.plot(ebins_mid, xsp_spectrum[0], label='xspec')
plt.plot(ebins_mid, spx_spectrum[0], label='spex')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

plt.plot(ebins_mid, (xsp_spectrum[0] - spx_spectrum[0])/xsp_spectrum[0], label='diff')
plt.show()


