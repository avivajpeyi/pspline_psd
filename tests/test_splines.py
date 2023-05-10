import matplotlib.pyplot as plt
import numpy as np
from pspline_psd.splines import BSpline, PSpline, knot_locator, dbspline
from pspline_psd.utils import get_fz

MAKE_PLOTS = True



def test_spline():
    degree = 2
    knots = np.array([0, 1, 2, 3, 4, 5, 6])
    coeff = np.array([-1, 2, 0, -1])

    bspline = BSpline(t=knots, c=coeff, k=degree)
    pspline = PSpline(t=knots, c=coeff, k=degree)

    assert np.allclose(bspline(0.5), pspline(0.5))

    if MAKE_PLOTS:
        import matplotlib.pyplot as plt
        x = np.linspace(0, 6, 100)
        plt.plot(x, bspline(x), label='BSpline')
        plt.plot(x, pspline(x), label='PSpline')
        # add knots
        plt.plot(knots, np.zeros_like(knots), 'o', label='knots')
        plt.legend()
        plt.show()




def test_knot_locator(helpers):
    data_0 = helpers.load_data_0()
    data = data_0['data']
    # format data:
    data = data - np.mean(data)
    data = data / np.std(data)
    knots = knot_locator(data, k=10, degree=3, eqSpaced=False)
    spline_list = dbspline(data, knots, degree=3)
    pdgm = get_fz(data) ** 2
    plt.Figure()
    plt.plot(pdgm)


