from pspline_psd.bayesian_utilities import llike, lpost, lprior
from pspline_psd.bayesian_utilities.whittle_utilities import psd_model
from pspline_psd.utils import get_fz, get_periodogram
from pspline_psd.splines import knot_locator, dbspline, PSpline, BSpline, get_penalty_matrix
from pspline_psd.sample.gibbs_pspline_simple import _get_initial_values, _format_data
import numpy as np
import matplotlib.pyplot as plt

MAKE_PLOTS = True


def test_llike(helpers):
    dataset = helpers.load_data_0()
    data = _format_data(dataset['data'])

    degree = 3
    k = 32
    tau, delta, phi, fz, periodogram, V, omega = _get_initial_values(data, k)
    fz = get_fz(dataset['data'])
    periodogram = get_periodogram(fz)
    knots = knot_locator(data, k=k, degree=degree, eqSpaced=False)
    db_list = dbspline(data, knots, degree=degree)
    n = len(periodogram)
    llike_val = llike(v=V, tau=tau, pdgrm=periodogram, db_list=db_list)
    assert not np.isnan(llike_val)
    psd = psd_model(V,  db_list, n)

    highest_ll_idx = np.argmax(dataset['ll.trace'])
    best_V = dataset['V'][:, highest_ll_idx]
    best_tau = dataset['tau'][highest_ll_idx]
    best_llike_val = llike(v=best_V, tau=best_tau, pdgrm=periodogram, db_list=db_list)
    assert llike_val < best_llike_val
    best_psd = psd_model(best_V, db_list, n)

    if MAKE_PLOTS:
        fig = plot_psd(periodogram, [psd, best_psd], [f'PSD {llike_val:.2f}', f'PSD {best_llike_val:.2f}'], db_list)
        fig.savefig(f'{helpers.OUTDIR}/test_llike.png')
        fig.show()


def plot_psd(periodogram, psds, labels, db_list):
    plt.plot(periodogram / np.sum(periodogram), label='periodogram')
    for psd, l in zip(psds, labels):
        plt.plot(psd / np.sum(psd), label=l, alpha=0.5)
    ylims = plt.gca().get_ylim()
    basis = db_list.toarray()
    net_val = np.sum(basis)

    basis = basis / net_val
    for idx, bi in enumerate(basis.T):
        kwgs = dict(color='gray', alpha=0.01)
        if idx == 0:
            kwgs['label'] = 'basis'
        plt.plot(bi / np.sum(bi), **kwgs)
    plt.ylim(*ylims)
    plt.ylabel('PSD')
    plt.legend(loc='upper right')
    return plt.gcf()
