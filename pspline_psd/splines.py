import numpy as np
from scipy.interpolate import BSpline, interp1d
from .logger import logger


def initialise_splines(
    periodogram: np.array, k: int, degree: int,
    omega: np.array, diffMatrixOrder: int,
    eqSpacedKnots: bool = True
):
    """ Initialise the splines given a periodogram and a number of knots"""
    V = generate_initial_spline_weights(periodogram, k)
    knots = knot_locator(periodogram, k, degree, eqSpacedKnots)
    db_list = dbspline(omega, knots, degree)
    P = get_penalty_matrix(k, db_list, diffMatrixOrder, eqSpacedKnots)
    return V, knots, db_list, P


def knot_locator(periodogram: np.ndarray, k: int, degree: int, eqSpaced: bool = False):
    """Determines the knot locations for a B-spline basis of degree `degree` and `k` knots.

    Returns
    -------
    knots : np.ndarray of shape (k - degree + 1,)


    """
    if eqSpaced:
        knots = np.linspace(0, 1, num=k - degree + 1)
        return knots
    else:
        logger.warning("knot_locator has not been tested for eqSpaced=False")

    aux = np.sqrt(periodogram)
    dens = np.abs(aux - np.mean(aux)) / np.std(aux)
    n = len(periodogram)

    dens = dens / np.sum(dens)
    cumf = np.cumsum(dens)

    df = interp1d(np.linspace(0, 1, num=n), cumf, kind='linear', fill_value=(0, 1))

    invDf = interp1d(
        df(np.linspace(0, 1, num=n)), np.linspace(0, 1, num=n), kind='linear', fill_value=(0, 1), bounds_error=False
    )

    # knots based on periodogram peaks
    knots = invDf(np.linspace(0, 1, num=k - degree + 1))

    return knots


def dbspline(x: np.ndarray, knots: np.ndarray, degree: int = 3, normalize: bool = True) -> np.ndarray:
    """Generate a B-spline density basis of any degree

    Returns:
    --------
    B : np.ndarray of shape (len(x), len(knots) + degree -1).

    """
    knots_with_boundary = np.r_[[knots[0]] * degree, knots, [knots[-1]] * degree]
    n_knots = len(knots_with_boundary)  # number of knots (including the external knots)
    assert n_knots == degree * 2 + len(knots)

    B = BSpline.design_matrix(x, knots_with_boundary, degree)

    if normalize:
        # "normalize" the basis functions
        mid_to_end_knots = knots_with_boundary[degree + 1:]
        start_to_mid_knots = knots_with_boundary[: (n_knots - degree - 1)]
        bs_int = (mid_to_end_knots - start_to_mid_knots) / (degree + 1)
        bs_int[bs_int == 0] = np.inf
        B = B / bs_int
        # TODO: this doesnt sum to 1, ask @pmat747 about this

    assert B.shape == (len(x), len(knots) + degree - 1)
    return B


def get_penalty_matrix(k, db_list, diffMatrixOrder, eqSpacedKnots) -> np.ndarray:
    """Computes the penalty matrix for a B-spline basis of degree `degree` and `k` knots.

    Returns
    -------
    penalty_matrix : np.ndarray of shape (k - degree - 1, k - degree - 1)
    """

    if eqSpacedKnots:
        P = __diff_matrix(k - 1, d=diffMatrixOrder)
        P = np.dot(P.T, P)
    else:
        raise NotImplementedError('Not implemented yet')

    epsilon = 1e-6
    P = P + epsilon * np.eye(P.shape[1])  # P^(-1)=Sigma (Covariance matrix)
    return P


def generate_initial_spline_weights(periodogram: np.ndarray, k: int) -> np.ndarray:
    """Generate initial weights for the spline basis functions"""
    scaled_periodogram = periodogram / np.sum(periodogram)
    # k equally spaced points
    idx = np.linspace(0, len(scaled_periodogram) - 1, k)
    idx = np.round(idx).astype(int)
    w = scaled_periodogram[idx]

    assert len(w) == k
    w[w == 0] = 1e-50  # prevents log(0) errors
    w = w / np.sum(w)
    w0 = w[:-1]
    v = np.log(w0 / (1 - np.sum(w0)))
    # convert nans to very small
    v[np.isnan(v)] = -1e50
    v = v.reshape(-1, 1)
    assert v.shape == (k - 1, 1)
    return v


def __diff_matrix(k, d=2):
    assert d < k, "d must be lower than k"
    assert np.all(np.array([d, k])) > 0, "d, k must be +ive ints"
    out = np.eye(k)
    for i in range(d):
        out = np.diff(out)
    return out.T
