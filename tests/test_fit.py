from xraysim.specutils.fit import *

indir = os.environ.get('XRAYSIM') + '/tests/inp/'

spectrum_bapec = indir + 'bapec_fakeit_for_test.pha'
rmf = indir + 'resolve_h5ev_2019a.rmf'
arf = indir + 'resolve_pnt_heasim_noGV_20190701.arf'


def assert_fit_results_within_tolerance(fit_result: xsp.Model, reference, tol=1.) -> None:
    """
    Checks that an Xspec model containing a fit result matches the reference values within
    tolerance
    :param fit_result: (xspec.Model) Xspec model containing the fit results
    :param reference: (float, tuple) Reference values
    :param tol: Tolerance in units of statistical error
    :return: None
    """

    assert fit_result.nParameters == len(reference)
    for ind, val in enumerate(reference):
        ind_fit = fit_result.startParIndex + ind
        assert abs(fit_result(ind_fit).values[0] - val) < tol * fit_result(ind_fit).sigma


def test_bapec_fit_start_with_correct_parameters():
    # Fitting the bapec spectrum produced with fakeit, and starting from the correct parameters should
    # lead to the correct result, within tolerance
    true_pars = (5., 0.3, 0.2, 300., 0.1)
    fit_result = generic(spectrum_bapec, "bapec", start=true_pars, method="cstat", rmf=rmf, arf=arf)
    assert_fit_results_within_tolerance(fit_result, true_pars, tol=2.)
