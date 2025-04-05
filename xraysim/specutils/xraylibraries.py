import numpy as np
import xspec as xsp
import pyatomdb
import pyspex as spex

h0 = 67.77
omega_m = 0.27
omega_l = 0.73
omega_r = 0.0


def str2bool(v):
    """

    :param v:
    :return:
    """
    if v == 'True':
        return True
    elif v == 'False':
        return False


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

        # This is to turn off the logs
        xsp.Xset.chatter = 0
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

    def calculate_spectrum(self, z, temperature, metallicity, element_index,
                           norm) -> np.array:
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
        # print(element_index,metallicity)
        if (params is not None) and (len(element_index)>1):
            params.update({i + 2: metallicity[i] for i in element_index.tolist()})
        else:
            params.update({2: metallicity[0]})

        params = {key: np.float64(value) for key, value in params.items()}

        #for key, value in params.items():
        #    print(f"Data type of {key}: {(value)}")

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

        # to visualize the parameters for the spex model
        # self.spex_model.par_show()

        for element_index_i, abund_i in zip(element_index, metallicity):
            self.spex_model.par(1, 2, element_index_i, abund_i)

        self.spex_model.calc()
        self.spex_model.mod_spectrum.get(1)

        # The spectrum given by spex is in ph bin-1 s-1 m-2. In order to compare with xspec cgs unit is required
        # i.e. ph bin-1 cm-2 s-1 (hence, multiplied by 1E-4)
        return np.array(self.spex_model.mod_spectrum.spectrum) * 1E-4
