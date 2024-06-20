import os
import sys

sys.path.append(os.environ.get("HEADAS") + "/lib/python")
# TODO: the three lines above are necessary only to make the code work in IntelliJ (useful for debugging)

import xspec as xsp

xsp.Xset.chatter = 0
xsp.Xset.allowNewAttributes = True


def xspec_clear() -> None:
    """
    Deletes all models, data and chains from the Xspec session.
    :return: None
    """
    xsp.AllModels.clear()
    xsp.AllData.clear()
    xsp.AllChains.clear()
    return None


def ignore_string(x) -> str:
    """
    Turns a variable containing a number into a string that contains a dot ('.') for floating points.
    This is mandatory for the ignore command of Xspec as otherwise it would consider channel instead of
    energy.
    :param x: Energy value (either float, int or str).
    :return: (str) The sting suited for the ignore command.
    """

    result = str(x)
    if "." not in result:
        result += "."
    return result


def ignore(spectrum: xsp.Spectrum, erange=(None, None)) -> None:
    """
    Sets the energy range to ignore on a spectrum.
    :param spectrum: (xspec.Spectrum) Spectrum.
    :param erange: (float 2, or int 2, or str 2) Energy range.
    :return: None
    """
    if len(erange) >= 2:
        if erange[0] is not None and erange[1] is not None:
            if erange[1] <= erange[0]:
                print("ERROR in ignore: invalid input:", erange, "Second argument must be larger than the first")
                raise ValueError

    # Setting lower energy limit
    if erange[0] is not None:
        spectrum.ignore("**-" + ignore_string(erange[0]))
    # Setting lower energy limit
    if erange[1] is not None:
        spectrum.ignore(ignore_string(erange[1]) + "-**")

    return None


class SpecFit(xsp.Model):
    def __init__(self, spectrum, model, bkg='USE_DEFAULT', rmf='USE_DEFAULT', arf='USE_DEFAULT', setPars=None):
        xspec_clear()
        xsp.Model.__init__(self, model, sourceNum=1, setPars=setPars)
        self.spectrum = xsp.Spectrum(spectrum, backFile=bkg, respFile=rmf, arfFile=arf)

    def parnames(self) -> list:
        """
        Return the list of model parameter names.
        :param self: (ModelFit)
        :return: (list) List with string containing the parameter names.
        """
        return [self(self.startParIndex + index).name for index in range(self.nParameters)]

    def fitresult(self) -> dict:
        """
        Return the fit results in a user-friendly format.
        :return: (dict) The fit result.
        """
        return {
            "parnames": self.parnames(),
            "values": [self(self.startParIndex + index).values[0] for index in range(self.nParameters)],
            "sigma": [self(self.startParIndex + index).sigma for index in range(self.nParameters)],
            "statistic": xsp.Fit.statistic,
            "dof": xsp.Fit.dof,
            "rstat": xsp.Fit.statistic / (xsp.Fit.dof - 1),
            "covariance": xsp.Fit.covariance,
            "method": xsp.Fit.statMethod,
            "nIterations": xsp.Fit.nIterations,
            "criticalDelta": xsp.Fit.criticalDelta
        }

    def fit(self, erange=(None, None), start=None, fixed=None, method="chi", niterations=100, criticaldelta=1.e-3):
        """
        Standard procedure to fit spectra.
        :param erange: (float, float) Energy range [keV]. If the first (second) elements is None the lower (higher)
            energy limit is not set. Default (None, None), i.e. all energy channels are considered.
        :param start: (float n) Starting parameters for the fit. The size depends on the model.
        :param fixed: (bool n) Indicates whether a parameter is fixed (True) or free (False). Default all False.
        :param method: (str) Fitting method, can be 'chi' or 'cstat'. Default 'chi'.
        :param niterations: (int) Number of iterations. Default 100.
        :param criticaldelta: (float) The absolute change in the fit statistic between iterations, less than which the
            fit is deemed to have converged.
        """

        xsp.AllData.ignore("bad")

        # Energy range
        ignore(self.spectrum, erange)

        # Initial conditions
        if start is not None:
            for index, par in enumerate(start):
                self(self.startParIndex + index).values = par

        # Fixed/free parameter
        if fixed is not None:
            for index, frozen in enumerate(fixed):
                self(self.startParIndex + index).frozen = frozen
        else:
            for index in range(self.nParameters):
                self(self.startParIndex + index).frozen = False

        # Statistic method
        if method is not None:
            xsp.Fit.statMethod = method

        # Number of iterations
        if niterations is not None:
            xsp.Fit.nIterations = niterations

        # Critical delta
        if criticaldelta is not None:
            xsp.Fit.criticalDelta = criticaldelta

        # Fitting
        xsp.Fit.perform()
