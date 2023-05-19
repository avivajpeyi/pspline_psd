import matplotlib.pyplot as plt
import numpy as np

from pspline_psd.bayesian_utilities import llike, lpost, lprior
from pspline_psd.bayesian_utilities.bayesian_functions import _vPv
from pspline_psd.bayesian_utilities.whittle_utilities import psd_model, unroll_psd
from pspline_psd.sample.gibbs_pspline_simple import _format_data, _get_initial_values
from pspline_psd.splines import BSpline, PSpline, dbspline, get_penalty_matrix, knot_locator
from pspline_psd.utils import get_fz, get_periodogram

MAKE_PLOTS = True


# TESTING USING
# gibbs_pspline(data, burnin=100, Ntotal=1000, degree = 3, k=5)


def test_psd_unroll():
    ar = unroll_psd(np.array([1, 2, 3, 4]), n=8)
    assert np.allclose(ar, np.array([1, 2, 2, 3, 3, 4, 4, 4]))
    ar = unroll_psd(np.array([1, 2, 3]), n=6)
    assert np.allclose(ar, np.array([1, 2, 2, 3, 3, 3]))
    ar = unroll_psd(np.array([1, 2, 3]), n=5)

    assert np.allclose(ar, np.array([1, 2, 2, 3, 3]))


def test_lprior():
    v = np.array([-68.6346650, 4.4997348, 1.6011013, -0.1020887])
    P = np.array(
        [
            [1e-6, 0.00, 0.0000000000, 0.0000000000],
            [0.00, 1e-6, 0.0000000000, 0.0000000000],
            [0.00, 0.00, 0.6093175700, 0.3906834292],
            [0.00, 0.00, 0.3906834292, 0.3340004330],
        ]
    )

    assert np.isclose(_vPv(v, P), 1.442495205)

    val = lprior(k=5, v=v, τ=0.1591549431, τα=0.001, τβ=0.001, φ=1, φα=1, φβ=1, δ=1, δα=1e-04, δβ=1e-04, P=P)
    assert np.isclose(val, 0.1120841558)


def test_dblist(helpers):
    db_list = helpers.load_db_list()

    # normalise each basis function
    # db_list = [db / np.max(db) for db in db_list]

    # plot each basis function
    for i, db in enumerate(db_list):
        plt.plot(db, color='gray')

    data = helpers.load_raw_data()

    degree = 3
    k = 32
    τ, δ, φ, fz, periodogram, V, omega = _get_initial_values(data, k)
    fz = get_fz(data)
    periodogram = get_periodogram(fz)
    knots = knot_locator(data, k=k, degree=degree, eqSpaced=True)
    db_list = dbspline(omega, knots, degree=degree)

    # twin axes
    ax = plt.gca()
    ax2 = ax.twinx()
    db_list = db_list.T
    # db_list = [db / np.max(db) for db in db_list]
    for i, db in enumerate(db_list):
        ax2.plot(db, color='red', ls='--')

    plt.show()


def test_llike(helpers):
    data = helpers.load_raw_data()
    degree = 3
    k = 32
    τ, δ, φ, fz, periodogram, V, omega = _get_initial_values(data, k)
    fz = get_fz(data)
    periodogram = get_periodogram(fz)
    knots = knot_locator(data, k=k, degree=degree, eqSpaced=True)
    db_list = dbspline(omega, knots, degree=degree)
    n = len(periodogram)

    psd = psd_model(V, db_list, n)
    assert not np.any(psd == 0)

    llike_val = llike(v=V, τ=τ, pdgrm=periodogram, db_list=db_list)
    assert not np.isnan(llike_val)

    ll_vals = helpers.load_ll()
    highest_ll_idx = np.argmax(ll_vals)
    best_V = helpers.load_v()[:, highest_ll_idx]
    best_τ = helpers.load_tau()[highest_ll_idx]
    best_llike_val = llike(v=best_V, τ=best_τ, pdgrm=periodogram, db_list=db_list)

    assert best_llike_val == ll_vals[highest_ll_idx]
    assert np.abs(llike_val - best_llike_val) < 100
    best_psd = psd_model(best_V, db_list, n)

    if MAKE_PLOTS:
        fig = plot_psd(periodogram, [psd, best_psd], [f'PSD {llike_val:.2f}', f'PSD {best_llike_val:.2f}'], db_list)
        fig.savefig(f'{helpers.OUTDIR}/test_llike.png')
        fig.show()


def plot_psd(periodogram, psds, labels, db_list):
    plt.plot(periodogram / np.sum(periodogram), label='periodogram', color='k')
    for psd, l in zip(psds, labels):
        plt.plot(psd / np.sum(psd), label=l)
    ylims = plt.gca().get_ylim()
    basis = db_list
    net_val = max(periodogram)

    basis = basis / net_val
    for idx, bi in enumerate(basis.T):
        kwgs = dict(color=f'C{idx+2}', lw=0.1, zorder=-1)
        if idx == 0:
            kwgs['label'] = 'basis'
        bi = unroll_psd(bi, n=len(periodogram))
        plt.plot(bi / net_val, **kwgs)
    plt.ylim(*ylims)
    plt.ylabel('PSD')
    plt.legend(loc='upper right')
    return plt.gcf()
