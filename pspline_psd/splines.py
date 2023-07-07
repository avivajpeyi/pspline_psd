import numpy as np
from scipy.interpolate import BSpline, interp1d


def knot_locator(pdgrm: np.ndarray, k: int, degree: int, eqSpaced: bool = False):
    """Determines the knot locations for a B-spline basis of degree `degree` and `k` knots.

    Returns
    -------
    knots : np.ndarray of shape (k - degree + 1,)
    (The x-positions of the knots)

    #TODO: ask if there is a simple way to test if this is correct.

    """
    if eqSpaced:
        knots = np.linspace(0, 1, num=k - degree + 1)
        return knots

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


    """
    knots_with_boundary = np.r_[[knots[0]] * degree, knots, [knots[-1]] * degree]
    n_knots = len(knots_with_boundary)  # number of knots (including the external knots)
    assert n_knots == degree * 2 + len(knots)

    # TODO: the R version has degree + 1 here... why?
    B = BSpline.design_matrix(x, knots_with_boundary, degree)

    if normalize:
        # normalize the basis functions
        mid_to_end_knots = knots_with_boundary[degree + 1:]
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


def _get_initial_spline_data(periodogram, k, degree, omega, diffMatrixOrder, eqSpacedKnots):
    V = _generate_initial_weights(periodogram, k)
    knots = knot_locator(periodogram, k, degree, eqSpacedKnots)
    db_list = dbspline(omega, knots, degree)
    if eqSpacedKnots:
        P = diff_matrix(k - 1, d=diffMatrixOrder)
        P = np.dot(P.T, P)
    else:
        P = get_penalty_matrix(db_list, diffMatrixOrder)
        P = P / np.linalg.norm(P)
    epsilon = 1e-6
    P = P + epsilon * np.eye(P.shape[1])  # P^(-1)=Sigma (Covariance matrix)
    return V, db_list, P, knots


def _generate_initial_weights(periodogram, k):
    scaled_periodogram = periodogram / np.sum(periodogram)
    # TODO keep k equidistant points
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


def diff_matrix(k, d=2):
    assert d < k, "d must be lower than k"
    assert np.all(np.array([d, k])) > 0, "d, k must be +ive ints"
    out = np.eye(k)
    for i in range(d):
        out = np.diff(out)
    return out.T
