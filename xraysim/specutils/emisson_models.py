import json
import os

import numpy as np
import xspec as xsp
import pyatomdb
import matplotlib.pyplot as plt


models_config_file = os.path.join(os.path.dirname(__file__), 'em_reference.json')
with open(models_config_file) as file:
    json_data = json.load(file)


class EmissionModels:
    def __init__(self, model_name: str, energy: np.ndarray):

        self.json_record = next((i for i in json_data if i['name'] == model_name), None)

        if self.json_record is None:
            raise ValueError(f"Model with name '{model_name}' not found in the json file.")

        self.json_record['metals_ref'] = np.array(self.json_record['metals_ref'], dtype=float)
        self.energy = energy

        if self.json_record['code'] == 'xspec':
            xsp.AllModels.setEnergies(f"{self.energy.min()} {self.energy.max()} {len(self.energy) - 1} lin")
            xsp.Xset.addModelString("APECROOT", "3.0.9")
            self.model = xsp.Model(self.json_record['model'])

        elif self.json_record['code'] == 'atomdb':
            self.model = pyatomdb.spectrum.CIESession()

        self.set_commands()

    def set_commands(self):

        init_settings = {
            'abund': lambda cmd: setattr(xsp.Xset, cmd['method'], cmd['arg']),
            'addModelString': lambda cmd: xsp.Xset.addModelString(cmd['arg'][0], cmd['arg'][1]),
            'abundset': lambda cmd: setattr(self.model, cmd['method'], cmd['arg']),
            'do_eebrems': lambda cmd: setattr(self.model, cmd['method'], cmd['arg']),
            'set_broadening': lambda cmd: self.model.set_broadening(thermal_broadening=True)
        }

        for command in self.json_record['xset']:
            init_settings.get(command['method'], lambda cmd: None)(command)

    def set_metals_ref(self, metal):

        metal_idx = {('apec', 1): [0],
                     ('vvapec', 1): np.arange(3, 31, 1) - 1,
                     ('vvapec', 11): np.array([6, 7, 8, 10, 12, 14, 16, 20, 26]) - 1
                     }

        idx = metal_idx.get((self.json_record['model'], self.json_record['n_metals']))

        np.put(self.json_record['metals_ref'], idx, metal)

        return idx

    def calculate_spectrum(self, z, temperature, metallicity, norm, flag_ene=False):

        elements_index = self.set_metals_ref(metallicity)

        if self.json_record['code'] == 'xspec':
            params = [temperature, *(self.json_record['metals_ref']).tolist(), z, norm]
            self.model.setPars(params)
            self.model.show()
            result = self.model.values(0)

        elif self.json_record['code'] == 'atomdb':
            self.model.set_response(self.energy*(1+z), raw=True)
            self.model.set_abund(elements_index+1,self.json_record['metals_ref'][elements_index])
            result = self.model.return_spectrum(temperature, log_interp=False)*norm * (1/(1+z))**1

        if flag_ene:
            bins = 0.5 * (np.array(self.model.energies(0))[1:] + np.array(self.model.energies(0))[:-1])
            result = result * bins

        return result


# Testing Line For Checking The Class and setup :

xspec_norm = 1
pyatomdb_norm= 1E14

# a = EmissionModels('TheThreeHundred-1', np.linspace(0.1, 10, 1000))

# print(a.calculate_spectrum(0.1, 0.34, [0.04], xspec_norm, False))
# print(a.calculate_spectrum(.2, 0.6, [0.01], xspec_norm, False))
# print(a.calculate_spectrum(.2, 0.2, [0.07], xspec_norm, False))

b = EmissionModels('TheThreeHundred-2', np.linspace(0.1, 10, 1000))

xspec1 = (b.calculate_spectrum(0.1, 3, [0.05], xspec_norm, False))
# print(b.calculate_spectrum(.2, 0.6, np.linspace(0.2, 0.3, 11), xspec_norm, False))
# print(b.calculate_spectrum(.2, 0.2, np.linspace(0.4, 0.5, 11), xspec_norm, False))


b = EmissionModels('TheThreeHundred-4', np.linspace(0.1, 10, 1000))

pydb1 = b.calculate_spectrum(0.1, 3, [0.05], pyatomdb_norm, False)


print(pydb1)
print(xspec1)

plt.plot(0.5*(np.linspace(0.1, 10, 1000)[1:]+np.linspace(0.1, 10, 1000)[:-1]),xspec1)
plt.plot(0.5*(np.linspace(0.1, 10, 1000)[1:]+np.linspace(0.1, 10, 1000)[:-1]),pydb1)

plt.xscale('log')
plt.yscale('log')
plt.show()
# For actual sim

# from gadgetutils.convert import gadgget2xspecnorm
# from xraysim.pygadgetreader import readsnap


# sim_path = '/home/atulit-pc/IdeaProjects/xraysim/tests/inp/snap_Gadget_sample'
# mass = readsnap(sim_path,'U   ', 'gas').shape
# print(mass)
# gad

#a = EmissionModels('TheThreeHundred-4', np.linspace(4, 5, 1000))
#print(a.calculate_spectrum(0.1, 0.54, [0.4], xspec_norm, False))
