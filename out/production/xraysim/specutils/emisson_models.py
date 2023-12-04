import json
import os

import numpy as np
import xspec as xsp
import pyatomdb
import matplotlib.pyplot as plt


def str2bool(v):
    if v == 'True':
        return True
    elif v == 'False':
        return False


models_config_file = os.path.join(os.path.dirname(__file__), 'em_reference.json')
with open(models_config_file) as file:
    json_data = json.load(file)


class XspecModel:
    def __init__(self, model_name, energy):
        xsp.AllModels.setEnergies(f"{energy.min()} {energy.max()} {len(energy) - 1} lin")
        xsp.Xset.addModelString("APECROOT", "3.0.9")
        self.xspec_model = xsp.Model(model_name)

    # doesn't change the object itself, that's why we have this warning
    def set_xspec_commands(self, commands):
        xspec_settings = {
            'abund': lambda cmd: setattr(xsp.Xset, cmd['method'], cmd['arg']),
            'addModelString': lambda cmd: xsp.Xset.addModelString(cmd['arg'][0], cmd['arg'][1]),
        }

        for command in commands:
            xspec_settings.get(command['method'], lambda cmd: None)(command)

    def calculate_spectrum(self, z, temperature, elements_index, metallicity, norm):
        params = [temperature, *metallicity.tolist(), z, norm]
        self.xspec_model.setPars(params)
        self.xspec_model.show()
        result = self.xspec_model.values(0)

        return result


class AtomdbModel:

    def __init__(self, model_name, energy):
        self.atomdb_model = pyatomdb.spectrum.CIESession()
        self.energy = energy

    def set_atomdb_commands(self, commands):
        atomdb_settings = {
            'abundset': lambda cmd: setattr(self.atomdb_model, cmd['method'], cmd['arg']),
            'do_eebrems': lambda cmd: setattr(self.atomdb_model, cmd['method'], str2bool(cmd['arg'][0])),
            'set_broadening': lambda cmd: self.atomdb_model.set_broadening(thermal_broadening=str2bool(cmd['arg'][0]))
        }
        for command in commands:
            print(type(str2bool(command['arg'][0])))
            print(type(command['arg'][0]))
            atomdb_settings.get(command['method'], lambda cmd: None)(command)

    def calculate_spectrum(self, z, temperature, elements_index, metallicity, norm):
        self.atomdb_model.set_response(self.energy * (1 + z), raw=True)
        self.atomdb_model.set_abund(elements_index + 1, metallicity[elements_index])
        result = self.atomdb_model.return_spectrum(temperature, log_interp=False) * norm * (1 / (1 + z)) ** 1

        return result


class EmissionModels:
    def __init__(self, model_name: str, energy: np.ndarray):

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

    def set_metals_ref(self, metal):

        metal_idx = {('apec', 1): [0],
                     ('vvapec', 1): np.arange(3, 31, 1) - 1,
                     ('vvapec', 11): np.array([6, 7, 8, 10, 12, 14, 16, 20, 26]) - 1
                     }

        idx = metal_idx.get((self.json_record['model'], self.json_record['n_metals']))

        np.put(self.json_record['metals_ref'], idx, metal)

        return idx

    def compute_spectrum(self, z, temperature, metallicity, norm, flag_ene=False):

        chem_species_index = self.set_metals_ref(metallicity)

        result = self.model.calculate_spectrum(z, temperature, chem_species_index, self.json_record['metals_ref'], norm)
        if flag_ene:
            bins = 0.5 * (np.array(self.energy[1:] + np.array(self.energy)[:-1]))
            result = result * bins

        return result


# Testing Line For Checking The Class and setup :

xspec_norm = 1
pyatomdb_norm = 1

# a = EmissionModels('TheThreeHundred-1', np.linspace(0.1, 10, 1000))

# print(a.compute_spectrum(0.1, 0.34, [0.04], xspec_norm, False))
# print(a.compute_spectrum(.2, 0.6, [0.01], xspec_norm, False))
# print(a.compute_spectrum(.2, 0.2, [0.07], xspec_norm, False))

# b = EmissionModels('TheThreeHundred-2', np.linspace(0.1, 10, 1000))

# print(b.compute_spectrum(0.1, 3, [0.05], xspec_norm, False))
# print(b.compute_spectrum(.2, 0.6, np.linspace(0.2, 0.3, 11), xspec_norm, False))
# print(b.compute_spectrum(.2, 0.2, np.linspace(0.4, 0.5, 11), xspec_norm, False))

b = EmissionModels('TheThreeHundred-4', np.linspace(0.1, 10, 1000))

print(b.compute_spectrum(0.1, 3, [0.05], pyatomdb_norm, False))

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
