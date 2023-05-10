from scipy.interpolate import BSpline
import numpy as np
from scipy.interpolate import interp1d
from .utils import get_fz


class PSpline(BSpline):
    """Penalized B-spline."""
    pass


def knot_locator(data: np.ndarray, k: int, degree: int, eqSpaced: bool = False):
    if (eqSpaced == True):
        knots = np.linspace(0, 1, num=k - degree + 1)
        return knots

    data = data - np.mean(data)
    FZ = get_fz(data)
    pdgrm = abs(FZ) ** 2

    aux = np.sqrt(pdgrm)
    dens = abs(aux - np.mean(aux)) / np.std(aux)
    n = len(pdgrm)

    dens = dens / np.sum(dens)
    cumf = np.cumsum(dens)

    df = interp1d(np.linspace(0, 1, num=n), cumf, kind='linear', fill_value=(0, 1))

    invDf = interp1d(df(np.linspace(0, 1, num=n)), np.linspace(0, 1, num=n), kind='linear', fill_value=(0, 1), bounds_error=False)

    # knots based on periodogram peaks
    knots = invDf(np.linspace(0, 1, num=k - degree + 1))
    return knots


def dbspline(x: np.ndarray, knots: np.ndarray, degree=3):
    knots_mult = np.concatenate((np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)))
    nknots = len(knots_mult)
    B = BSpline(knots_mult, np.eye(nknots), degree + 1)(x)

    bs_int = (knots_mult[(degree + 1):] - knots_mult[:(nknots - degree - 1)]) / (degree + 1)
    bs_int[bs_int == 0] = np.inf
    B_norm = B.T / bs_int

    return B_norm
