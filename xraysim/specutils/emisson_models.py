import json
import os

import numpy as np
import xspec as xsp


class EmissionModels:
    def __init__(self, model_name: str, energy: np.ndarray):
        models_config_file = os.path.join(os.path.dirname(__file__), 'em_reference.json')
        with open(models_config_file) as file:
            json_data = json.load(file)
        self.json_record = next((i for i in json_data if i['name'] == model_name), None)

        if self.json_record is None:
            raise ValueError(f"Model with name '{model_name}' not found in the json file.")

        self.json_record['metals_ref'] = np.array(self.json_record['metals_ref'])
        self.energy = energy
        self.set_commands()

        self.model = xsp.Model(self.json_record['model'])

        print(self.json_record['metals_ref'], self.json_record['n_metals'])

    def set_commands(self):
        xsp.AllModels.setEnergies(f"{self.energy.min()} {self.energy.max()} {len(self.energy) - 1} lin")
        xsp.Xset.addModelString("APECROOT", "3.0.9")

        init_settings = {
            'abund': lambda cmd: setattr(xsp.Xset, cmd['method'], cmd['arg']),
            'addModelString': lambda cmd: xsp.Xset.addModelString(cmd['arg'][0], cmd['arg'][1])
        }

        for cmd in self.json_record['xset']:
            init_settings[cmd['method']](cmd)

    def set_metals_ref(self, metal):

        if self.json_record['name'] == 'TheThreeHundred-1':
            self.json_record['metals_ref'][0] = metal[0]

        elif self.json_record['name'] == 'TheThreeHundred-2':
            self.json_record['metals_ref'][2:] = metal[0]

        elif self.json_record['name'] == 'TheThreeHundred-3':
            met_species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26]) - 1
            self.json_record['metals_ref'][met_species] = metal

    def calculate_spectrum(self, z, temperature, metallicity, norm, flag_ene=False):

        self.set_metals_ref(metallicity)

        params = [temperature, *(self.json_record['metals_ref']).tolist(), z, norm]
        self.model.setPars(params)
        self.model.show()
        result = self.model.values(0)
        if flag_ene:
            bins = 0.5 * (np.array(self.model.energies(0))[1:] + np.array(self.model.energies(0))[:-1])
            result = result * bins

        return result


# Testing Line For Checking The Class and setup :

from gadgetutils.convert import gadgget2xspecnorm

# gad
#

xspec_norm = 1E-14

a = EmissionModels('TheThreeHundred-1', np.linspace(0.1, 10, 1000))

print(a.calculate_spectrum(0.1, 0.34, [0.04], xspec_norm, False))
print(a.calculate_spectrum(.2, 0.6, [0.01], xspec_norm, False))
print(a.calculate_spectrum(.2, 0.2, [0.07], xspec_norm, False))

b = EmissionModels('TheThreeHundred-3', np.linspace(0.1, 10, 1000))

print(b.calculate_spectrum(0.1, 0.34, np.linspace(0.1, 0.3, 11), xspec_norm, False))
print(b.calculate_spectrum(.2, 0.6, np.linspace(0.2, 0.3, 11), xspec_norm, False))
print(b.calculate_spectrum(.2, 0.2, np.linspace(0.4, 0.5, 11), xspec_norm, False))
