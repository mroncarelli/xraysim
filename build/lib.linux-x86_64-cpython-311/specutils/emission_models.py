import json
import os

import numpy as np
import xspec as xsp

# Initialization of global variables
models_config_file = os.path.join(os.path.dirname(__file__), 'em_reference.json')
with open(models_config_file) as file:
    json_data = json.load(file)


def spectrum(model: str, energy: np.ndarray, z, temperature, flag_ene=False) -> np.ndarray:
    result = np.ndarray([len(energy)])
    # TODO Implement here model calculation
    return result
