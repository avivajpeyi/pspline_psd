import numpy as np
from scipy.interpolate import BSpline, interp1d

from .utils import get_fz


class PSpline(BSpline):
    """Penalized B-spline."""

    pass


def knot_locator(data: np.ndarray, k: int, degree: int, eqSpaced: bool = False):
    """Determines the knot locations for a B-spline basis of degree `degree` and `k` knots.

    Returns
    -------
    knots : np.ndarray of shape (k - degree + 1,)

    #TODO: ask if there is a simple way to test if this is correct.

    """
    if eqSpaced:
        knots = np.linspace(0, 1, num=k - degree + 1)
        return knots

    data = data - np.mean(data)
    FZ = get_fz(data)
    pdgrm = np.power(np.abs(FZ), 2)

    aux = np.sqrt(pdgrm)
    dens = np.abs(aux - np.mean(aux)) / np.std(aux)
    n = len(pdgrm)

    dens = dens / np.sum(dens)
    cumf = np.cumsum(dens)

    df = interp1d(np.linspace(0, 1, num=n), cumf, kind='linear', fill_value=(0, 1))

    invDf = interp1d(
        df(np.linspace(0, 1, num=n)), np.linspace(0, 1, num=n), kind='linear', fill_value=(0, 1), bounds_error=False
    )

    # knots based on periodogram peaks
    knots = invDf(np.linspace(0, 1, num=k - degree + 1))

    return knots


def dbspline(x: np.ndarray, knots: np.ndarray, degree=3, normalize=True):
    """Generate a B-spline density basis of any degree

    Returns:
    --------
    B : np.ndarray of shape (len(x), len(knots) + degree -1 [I THINK])


    #TODO: ask if there is a simple way to test if this is correct.

    """
    knots_with_boundary = np.r_[[knots[0]] * degree, knots, [knots[-1]] * degree]
    n_knots = len(knots_with_boundary)  # number of knots (including the external knots)
    assert n_knots == degree * 2 + len(knots)

    # TODO: the R version has degree + 1 here... why?
    B = BSpline.design_matrix(x, knots_with_boundary, degree)

    if normalize:
        # normalize the basis functions
        mid_to_end_knots = knots_with_boundary[degree + 1 :]
        start_to_mid_knots = knots_with_boundary[: (n_knots - degree - 1)]
        bs_int = (mid_to_end_knots - start_to_mid_knots) / (degree + 1)
        bs_int[bs_int == 0] = np.inf
        B = B / bs_int

    assert B.shape == (len(x), len(knots) + degree - 1)
    # assert np.allclose(np.sum(B, axis=1), 1), 'Basis functions do not sum to 1'
    return B


def get_penalty_matrix(basis: np.ndarray, Lfdobj: int) -> np.ndarray:
    """Computes the penalty matrix for a B-spline basis of degree `degree` and `k` knots.

    Returns
    -------
    penalty_matrix : np.ndarray of shape (k - degree - 1, k - degree - 1)
    """
    raise NotImplementedError('Not implemented yet')
