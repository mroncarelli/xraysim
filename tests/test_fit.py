import os
import sys

sys.path.append(os.environ.get("HEADAS") + "/lib/python")
# TODO: the three lines above are necessary only to make the code work in IntelliJ (useful for debugging)

from xraysim.specutils.specfit import *

input_dir = os.environ.get('XRAYSIM') + '/tests/inp/'
spectrum_bapec = input_dir + 'bapec_fakeit_for_test.pha'
spectrum_apec = input_dir + 'apec_fakeit_for_test.pha'
rmf = input_dir + 'resolve_h5ev_2019a.rmf'
arf = input_dir + 'resolve_pnt_heasim_noGV_20190701.arf'


def assert_fit_results_within_tolerance(specfit: SpecFit, reference, tol=1.) -> None:
    """
    Checks that a Xspec model containing a fit result matches the reference values within
    tolerance
    :param specfit: (SpecFit) Xspec model containing the fit results
    :param reference: (float, tuple) Reference values
    :param tol: Tolerance in units of statistical error
    :return: None
    """

    assert specfit.nParameters == len(reference)
    for ind, val in enumerate(reference):
        ind_fit = specfit.startParIndex + ind
        if not specfit(ind_fit).frozen:
            assert abs(specfit(ind_fit).values[0] - val) < tol * specfit(ind_fit).sigma


def test_apec_fit_start_with_correct_parameters():
    # Fitting the apec spectrum produced with fakeit, and starting from the correct parameters should
    # lead to the correct result, within tolerance
    true_pars = (7., 0.2, 0.15, 0.1)
    specfit = SpecFit(spectrum_apec, "apec", rmf=rmf, arf=arf)
    specfit.fit(start=true_pars, method="cstat")
    assert_fit_results_within_tolerance(specfit, true_pars, tol=2.)


def test_bapec_fit_start_with_correct_parameters():
    # Fitting the bapec spectrum produced with fakeit, and starting from the correct parameters should
    # lead to the correct result, within tolerance
    true_pars = (5., 0.3, 0.2, 300., 0.1)
    specfit = SpecFit(spectrum_bapec, "bapec", rmf=rmf, arf=arf)
    specfit.fit(start=true_pars, method="cstat")
    assert_fit_results_within_tolerance(specfit, true_pars, tol=2.)


def test_apec_and_bapec_fit_start_with_correct_parameters():
    # Fitting the apec and bapec spectrum produced with fakeit, one after the other, starting from the correct
    # parameters should lead to the correct result, within tolerance
    true_pars = (7., 0.2, 0.15, 0.1)
    specfit = SpecFit(spectrum_apec, "apec", rmf=rmf, arf=arf)
    specfit.fit(start=true_pars, method="cstat")
    assert_fit_results_within_tolerance(specfit, true_pars, tol=2.)

    true_pars = (5., 0.3, 0.2, 300., 0.1)
    specfit = SpecFit(spectrum_bapec, "bapec", rmf=rmf, arf=arf)
    specfit.fit(start=true_pars, method="cstat")
    assert_fit_results_within_tolerance(specfit, true_pars, tol=2.)
