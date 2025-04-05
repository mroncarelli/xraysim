import json
import os

import gadgetutils.convert
import numpy as np

from xraysim.specutils.xraylibraries import XspecModel, AtomdbModel, SpexModel

from gadgetutils.phys_const import kpc2cm, Xp, m_p, Msun2g
from astropy.cosmology import FlatLambdaCDM

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


# current_directory = os.getcwd()

# models_config_file = os.path.join(current_directory, 'em_reference.json')

# print(models_config_file)
# with open(models_config_file) as file:
#     json_data = json.load(file)
models_config_file = os.path.join(os.path.dirname(__file__), 'em_reference.json')
with open(models_config_file) as file:
    json_data = json.load(file)


# this is the collection of models and their commands setting-abundance setting, broadening and metal reference
# [print(i) for i in json_data ]

class EmissionModels:
    def __init__(self, model_name: str, energy: np.array):
        """
        The EmissionModels constructor first searches for the model 'name' in the JSON record. Once it successfully
        locates the name, it initializes the X-Ray library along with its corresponding commands
        :param model_name: str - Specifies the model name as 'TheThreeHundred-1/2/3/4'.
        :param energy: list of float - Represents the range of energy values in KeV for spectrum calculation.
        """
        self.json_record = next((i for i in json_data if i['name'] == model_name), None)

        # setting all the variables for the instances of the class - self
        if self.json_record is None:
            raise ValueError(f"Model with name '{model_name}' not found in the json file.")

        self.json_record['metals_ref'] = np.array(self.json_record['metals_ref'], dtype=float)
        self.json_record['chemical_elements'] = np.array(self.json_record['chemical_elements'], dtype=str)
        self.energy = energy
        # check all the set instances variable
        # print(self.json_record)

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

    def set_metals_ref(self, metal) -> np.array:
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
            #print(self.json_record['metals_ref'],idx,Abundance_Table['Symbols'][idx])
            #print(Abundance_Table['Symbols'][idx], idx, '\n')

            # scaling with respect to Anders and Grevesse values
            if self.json_record['n_metals'] > 1:
                self.json_record['metals_ref'][idx] = self.json_record['metals_ref'][idx] / Abundance_Table['AbundanceTable'][idx]
            else:
                self.json_record['metals_ref'][idx] = self.json_record['metals_ref'][idx] / Z_solar

        else:
            raise ValueError(f"wrong setting n_metals is'{self.json_record['n_metals']} and len of metal'{len(metal)}.")

        return idx

    def compute_spectrum(self, z, temperature, metallicity, norm,
                         flag_ene=False):
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
            bins = 0.5 * (np.array(self.energy[1:] + np.array(self.energy)[:-1]))  # [10^-14 keV s^-1 cm-2]
            result = result * bins

        return np.array(result, dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("removing dummy files")
        dir_name = os.getcwd()
        test = os.listdir(dir_name)

        for item in test:
            if item.endswith(".dum"):
                os.remove(os.path.join(dir_name, item))
                
                
# testing line for gadget
#T = 9.0404 # KeV
#Z = [0.000192149] # Metallicity
#energy_bins = np.linspace(0.1, 2.4, 101)
#model_atom_db = EmissionModels(model_name='TheThreeHundred-2', energy=energy_bins)
# print(model_atom_db.compute_spectrum(0.2, T, Z, 1.0))
#import matplotlib.pyplot as plt

# norm_1 = gadgetutils.convert.gadget2xspecnorm(0.0239, 124.807E-9, ((1/100)*(3E5)*1000),.677)

# norm_3 = gadgetutils.convert.gadget2xspecnorm(0.0239, 124.807E-9, ((3/100)*(3E5)*1000),.677)

# energy_bins = 0.5*(energy_bins[1:]+energy_bins[:-1])
# plt.plot(energy_bins, model_atom_db.compute_spectrum(1, T, Z, norm_1*(1/(1+1))**2), color='r')
# plt.plot(energy_bins*2, 3*3*2*2*model_atom_db.compute_spectrum(3, T, Z, norm_3*(1/(1+3))**2), color = 'k')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# integrated_lum = np.sum( energy_bins*model_atom_db.compute_spectrum(1, T, Z, norm_1*(1/(1+1))**2) )
# print(integrated_lum)
