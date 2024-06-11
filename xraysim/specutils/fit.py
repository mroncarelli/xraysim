import sys
import os
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


def get_param_names(model: xsp.Model) -> list:
    """
    Return the list of model parameter names.
    :param model: (xspec.Model) The xspec model object.
    :return: (list) List with string containing the parameter names.
    """
    return [model(model.startParIndex + index).name for index in range(model.nParameters)]


def generic(spectrum, model: str, erange=(None, None), start=None, fixed=(False, False, False, False, False),
            method="chi", niterations=100, criticaldelta=1.e-3, bkg='USE_DEFAULT', rmf='USE_DEFAULT',
            arf='USE_DEFAULT') -> xsp.Model:
    """
    Generic procedure to fit spectra.
    :param spectrum: (str or xsp.Spectrum) Spectrum to be fitted.
    :param model: (str) Model used for the fit.
    :param erange: (float, float) Energy range [keV]. If the first (second) elements is None the lower (higher) energy
        limit is not set. Default (None, None), i.e. all energy channels are considered.
    :param start: (float n) Starting parameters for the fit. The size depends on the model.
    :param fixed: (bool n) Indicates whether a parameter is fixed (True) or free (False). Default all False.
    :param method: (str) Fitting method, can be 'chi' or 'cstat'. Default 'chi'.
    :param niterations: (int) Number of iterations. Default 100.
    :param criticaldelta: (float) The absolute change in the fit statistic between iterations, less than which the fit
        is deemed to have converged.
    :param bkg: (str) Background file. Default 'USE_DEFAULT', i.e. it is derived from the spectrum.
    :param rmf: (str) Response matrix file. Default 'USE_DEFAULT', i.e. it is derived from the spectrum.
    :param arf: (str) Ancillary response file. Default 'USE_DEFAULT', i.e. it is derived from the spectrum.
    :return: (xsp.Model) An Xspec model containing the fit result, including a fitResult property with the summary of
        the fit results stored in a dictionary.
    """

    xspec_clear()

    if type(spectrum) == str:
        spectrum_ = xsp.Spectrum(spectrum, backFile=bkg, respFile=rmf, arfFile=arf)  # Pha file
    elif type(spectrum) == xsp.Spectrum:
        spectrum_ = spectrum
    else:
        print("ERROR in bapec: invalid input spectrum.")
        raise ValueError

    xsp.AllData.ignore("bad")

    # Model
    model_ = xsp.Model(model)
    param_list = get_param_names(model_)
    index_start = model_.startParIndex

    # Energy range
    ignore(spectrum_, erange)

    # Initial conditions
    if start is not None:
        for index, par in enumerate(start):
            model_(index_start + index).values = par

    # Fixed/free parameter
    if fixed is not None:
        for index, frozen in enumerate(fixed):
            model_(index_start + index).frozen = frozen

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

    # Output
    model_.fitResult = {
        "parnames": param_list,
        "values": [model_(index_start + index).values[0] for index in range(model_.nParameters)],
        "sigma": [model_(index_start + index).sigma for index in range(model_.nParameters)],
        "statistic": xsp.Fit.statistic,
        "dof": xsp.Fit.dof,
        "covariance": xsp.Fit.covariance,
        "method": method,
        "nIterations": niterations,
        "criticalDelta": criticaldelta
    }

    return model_
