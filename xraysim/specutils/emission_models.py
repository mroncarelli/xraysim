import json
import os

import numpy as np
import xspec as xsp


# Initialization of global variables
models_config_file = os.path.join(os.path.dirname(__file__), 'em_reference.json')
with open(models_config_file) as file:
    json_data = json.load(file)

# Emission models initialize in the json file
Models = {
    em_model['name'].lower(): {
        'code':       em_model.get('code'),
        'model':      em_model.get('model'),
        'commands':   em_model.get('xset'),
        'n_metals':   em_model.get('n_metals'),
        'metals_ref': np.array(em_model.get('metals_ref'))
    }
    for em_model in json_data
}


def set_metals_ref(model, metal):

    # for apec model with single metallicity
    def set_apec1(met):

        model['metals_ref'] = met[:]

    # vvapec model with single metallicity but elements higher than Helium are assigned this metallicity
    def set_vvapec1(met):

        model['metals_ref'][2:] = met[0]

    # vvapec model with metallicity defined for 11 metals from the Gadget sims
    def set_vvapec2(met):
        # this has been set for now according to gizmo simba run paper, which says it tracks 11 elements
        # [(H, He, C, N, O, Ne, Mg, Si, S, Ca, and Fe)]
        # but I think met_species array can be set up in JSON config file, won't have to change much here then
        met_species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26]) - 1
        model['metals_ref'][met_species] = met

    set_metal_arr = {

        ('apec', 1): set_apec1,
        ('vvapec', 1): set_vvapec1,
        ('vvapec', 11): set_vvapec2
    }

    set_metal_arr[(model['model'], model['n_metals'])](metal)


def identify_model(model :str, metal) -> dict:
    model_id = None
    num_metal = len(metal)
    for key, value in Models.items():

        if value['model'] == model and value['n_metals'] == num_metal:
            model_id = Models[key]

    return model_id


def model_init(em_model, arr_energy):
    model = None

    xsp.AllModels.setEnergies(f"{arr_energy.min()} {arr_energy.max()} {len(arr_energy) - 1} lin")
    xsp.Xset.addModelString("APECROOT", "3.0.9")

    init_settings = {
        'abund': lambda cmd: setattr(xsp.Xset, cmd['method'], cmd['arg']),
        'addModelString': lambda cmd: xsp.Xset.addModelString(cmd['arg'][0], cmd['arg'][1])
    }

    for cmd in em_model['commands']:
        init_settings[cmd['method']](cmd)

    model = xsp.Model(em_model['model'])

    return model


def set_params(em_model, temp, redshift, norm):
    params = [temp, *(em_model['metals_ref']).tolist(), redshift, norm]
    return params


def spectrum(model: str, energy: np.ndarray, z, temperature, metallicity, flag_ene=False) -> np.ndarray:

    # Final emission spectrum for the model
    result = np.ndarray([len(energy)])
    # TODO Implement here model calculation

    # Identify the model from the JSON configuration file
    emission_model = identify_model(model, metallicity)

    # set_metals_reff array
    set_metals_ref(emission_model, metallicity)

    # Initialize the model
    model = model_init(emission_model, energy)

    # set parameters for the model
    # for actual norm I require information - emission measure, angular cosmological distance, i.e. the
    # factor on the webpage
    model.setPars(set_params(emission_model, temperature, z, 1E-14))
    model.show()

    result = model.values(0)

    if flag_ene:
        bins = 0.5*(np.array(model.energies(0))[1:] + np.array(model.energies(0))[:-1])
        result = result * bins

    return result


# Testing line

print(spectrum('apec', np.linspace(0,10,14800), 1, 3, 0.07*np.ones(1), False))
# HAVE TO FIGURE OUT HOW METALLICITY GIVEN IN TERMS OF MASS FRACTION FROM SIM CAN BE CHANGED TO SOLAR ABUNDANCE ACCORDIN
# TO ANDERS AND GREVEASSE
