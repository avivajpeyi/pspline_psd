import numpy as np
import matplotlib.pyplot as plt
import pytest
from pspline_psd.utils import get_fz
from pspline_psd.splines import dbspline, knot_locator, _generate_initial_weights
from pspline_psd.bayesian_utilities.whittle_utilities import get_unormalised_psd
from pspline_psd.bayesian_utilities.bayesian_functions import llike
from scipy.fft import fft
import matplotlib.pyplot as plt

plt.style.use('default')
# import gridspec from matplotlib
from matplotlib import gridspec

try:
    import rpy2
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import default_converter
    from rpy2.robjects.packages import importr
except ImportError:
    rpy2 = None


@pytest.mark.skipif(rpy2 is None, reason="rpy2 required for this test")
def test_direct_comparison(helpers):
    data = helpers.load_raw_data()
    n = len(data)
    omega = np.linspace(0, 1, n // 2 + 1)
    degree = 3
    k = 32

    r_data = __compute_r_psd(data, k, degree, omega)
    py_data = __compute_py_psd(data, k, degree, omega)

    fig = __make_comparison_plot(r_data, py_data)
    fig.savefig(f"{helpers.OUTDIR}/r_package_comparison.png")
    # plt.show()


def __compute_r_psd(data, k, degree, omega):
    r_pspline = importr("psplinePsd")
    np_cv_rules = default_converter + numpy2ri.converter

    with np_cv_rules.context():
        fz = r_pspline.fast_ft(data)
        pdgrm = np.power(fz, 2)
        v = r_pspline.get_initial_weights(pdgrm, k)
        knots = r_pspline.knotLoc(data=data, k=k, degree=degree, eqSpaced=True)
        dblist = r_pspline.dbspline(omega, knots, degree)
        psd = r_pspline.qpsd(omega, k, v, degree, dblist)

        tau_vals = np.geomspace(0.01, 1, 100)
        lnl = np.zeros(len(tau_vals))
        for i, tau in enumerate(tau_vals):
            lnl[i] = r_pspline.llike(omega, fz, k, v, tau=tau, pdgrm=pdgrm, degree=degree, db_list=dblist)

    return {'fz': fz, 'v': v, 'dblist': dblist, 'psd': psd, 'tau_vals': tau_vals, 'lnl': lnl}


def __compute_py_psd(data, k, degree, omega):
    fz = get_fz(data)
    pdgrm = np.power(np.abs(fz), 2)
    v = _generate_initial_weights(pdgrm, k)
    knots = knot_locator(data=data, k=k, degree=degree, eqSpaced=True)
    dblist = dbspline(x=omega, knots=knots, degree=degree)
    psd = get_unormalised_psd(v, dblist)

    tau_vals = np.geomspace(0.01, 1, 100)
    lnl = np.zeros(len(tau_vals))
    for i, tau in enumerate(tau_vals):
        lnl[i] = llike(v, τ=tau, pdgrm=pdgrm, db_list=dblist)

    return {'fz': fz, 'v': v, 'dblist': dblist.T, 'psd': psd, 'tau_vals': tau_vals, 'lnl': lnl}


def __make_comparison_plot(r_data, py_data):
    fig, axes = plt.subplots(5, 1, figsize=(10, 10))
    rkwgs = dict(color='tab:red', alpha=0.5, lw=4, zorder=-10, label='r')
    pykwgs = dict(color='tab:blue', alpha=1, ls='--', lw=2, label='py')
    difkwgs = dict(color='tab:green', alpha=1, lw=2, label='diff')

    # plot fz
    axes[0].plot(r_data['fz'], **rkwgs)
    axes[0].plot(py_data['fz'], **pykwgs)
    axes[0].set_ylabel('fz')
    axes[0].legend()

    # plot v
    axes[1].plot(r_data['v'], **rkwgs)
    axes[1].plot(py_data['v'], **pykwgs)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('v')

    # plot dblist
    for i in range(len(r_data['dblist'])):
        axes[2].plot(r_data['dblist'][i], **rkwgs)
        axes[2].plot(py_data['dblist'][i], **pykwgs)
    axes[2].set_ylabel('dblist')

    # plot psd
    # relative difference bw psd
    rel_diff = np.abs(r_data['psd'] - py_data['psd']) / np.abs(r_data['psd'])
    axes[3].plot(rel_diff[1:-3], **difkwgs)
    axes[3].set_ylabel(r'$\frac{|S_{\rm r} - S_{\rm py} |}{|S_{r}|}$')

    # plot lnl
    # relative difference bw lnl
    rel_diff = np.abs(r_data['lnl'] - py_data['lnl']) / np.abs(r_data['lnl'])
    axes[4].plot(r_data['tau_vals'], rel_diff, **difkwgs)
    # axes[4].plot(py_data['tau_vals'], py_data['lnl'], **pykwgs)
    axes[4].set_ylabel(r'$\frac{|lnl_{\rm r} - lnl_{\rm py} |}{|lnl_{r}|}$')
    axes[4].set_xlabel('tau')
    axes[4].set_xscale('log')

    plt.tight_layout()
    return fig


@pytest.mark.skipif(rpy2 is None, reason="rpy2 required for this test")
def test_mcmc_comparison(helpers):

    nsteps = 2000
    data = helpers.load_raw_data()
    r_psd, r_psd_p05, r_psd_p95 = __r_mcmc(data, nsteps)
    py_psd, py_psd_p05, py_psd_p95 = __py_mcmc(data, nsteps)
    n, newn = len(data), len(py_psd)
    periodogram = np.abs(np.power(fft(data), 2) / (2 * np.pi * n))[0:newn]
    psd_x = np.linspace(0, 3.14, newn)
    plt.style.use(
        'https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.plot(figsize=(8, 4))
    plt.scatter(psd_x[1:], periodogram[1:], color='k', label='Data', s=0.75)
    plt.plot(psd_x[1:], py_psd[1:], color='tab:orange', alpha=0.5, label='Python (Uniform CI)')
    plt.fill_between(psd_x[1:], py_psd_p05[1:], py_psd_p95[1:], color='tab:orange', alpha=0.2, linewidth=0.0)
    plt.plot(psd_x[1:], r_psd[1:], color='tab:green', alpha=0.5, label='R (Uniform CI)')
    plt.fill_between(psd_x[1:], r_psd_p05[1:], r_psd_p95[1:], color='tab:green', alpha=0.2, linewidth=0.0)
    # turn off grid
    plt.grid(False)
    # set font to sans-serif
    # legend but increase size of markers
    plt.legend(markerscale=5, frameon=False)
    plt.ylabel('PSD')
    plt.xlabel('Freq')
    plt.tight_layout()
    # turn off minor ticks
    plt.minorticks_off()
    plt.savefig(f'{helpers.OUTDIR}/psd_comparison.png', dpi=300)


def __r_mcmc(data, nsteps=200):
    r_pspline = importr("psplinePsd")
    np_cv_rules = default_converter + numpy2ri.converter

    burnin = int(0.15 * nsteps)
    with np_cv_rules.context():
        mcmc = r_pspline.gibbs_pspline(data, burnin=burnin, Ntotal=nsteps, degree=3, eqSpacedKnots=True)
    return mcmc['psd.median'], mcmc['psd.u05'], mcmc['psd.u95']


from pspline_psd.sample.gibbs_pspline_simple import gibbs_pspline_simple


def __py_mcmc(data, nsteps=200):
    burnin = int(0.15 * nsteps)
    mcmc = gibbs_pspline_simple(data, burnin=burnin, Ntotal=nsteps, degree=3, eqSpacedKnots=True,
                                metadata_plotfn="py_mcmc.png")
    psd_quants = mcmc.psd_quantiles
    return psd_quants[0, :], psd_quants[1, :], psd_quants[2, :]
