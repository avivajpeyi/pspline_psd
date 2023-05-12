import matplotlib.pyplot as plt
import numpy as np
from pspline_psd.splines import BSpline, PSpline, knot_locator, dbspline
from pspline_psd.utils import get_fz

from scipy import interpolate

MAKE_PLOTS = True


def test_spline():
    degree = 2
    knots = np.array([0, 1, 2, 3, 4, 5, 6])
    coeff = np.array([-1, 2, 0, -1])

    bspline = BSpline(t=knots, c=coeff, k=degree)
    pspline = PSpline(t=knots, c=coeff, k=degree)

    assert np.allclose(bspline(0.5), pspline(0.5))


def test_b_spline_matrix(helpers):
    x = np.linspace(-5, 5, 100)
    knots = np.array([-5, 0, 5])
    degree = 2
    B_norm = dbspline(x, knots=knots, degree=degree)
    basis = B_norm.toarray()

    assert np.allclose(np.sum(basis, axis=1), 1)

    if MAKE_PLOTS:
        for i in range(len(basis.T)):
            plt.plot(x, basis[:, i].ravel(), label=f'basis {i}', color=f"C{i}")

        # sum all basis functions
        plt.plot(x, np.sum(basis, axis=1), label='sum of basis functions')
        # plot knots
        plt.plot(np.array([-5, 0, 5]), np.zeros(3), 'x', color='k', label='knots')

        plt.legend(loc='upper left')
        plt.savefig(f'{helpers.OUTDIR}/test_b_spline_matrix.png')

        plt.show()
