import json
import os

import numpy as np
import xspec as xsp
# from gadgetutils.convert import gadgget2xspecnorm


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
        met_species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26]) - 1
        self.json_record['metals_ref'][met_species] = metal

    def calculate_spectrum(self, z, temperature, metallicity, flag_ene=False):

        self.set_metals_ref(metallicity)

        params = [temperature, *(self.json_record['metals_ref']).tolist(), z, 1E-14]
        self.model.setPars(params)
        self.model.show()
        result = self.model.values(0)
        if flag_ene:
            bins = 0.5 * (np.array(self.model.energies(0))[1:] + np.array(self.model.energies(0))[:-1])
            result = result * bins

        return result


# Testing Line For Checking The Class :
a = EmissionModels('TheThreeHundred-3', np.linspace(0.1, 10, 1000))

print(a.calculate_spectrum(0.1, 0.34, np.linspace(0.5, 0.6, 11), False))
print(a.calculate_spectrum(.2, 0.6, np.linspace(0.1, .2, 11), False))
