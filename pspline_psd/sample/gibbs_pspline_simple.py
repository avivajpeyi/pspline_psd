import numpy as np
from scipy import sparse
from scipy import stats
from ..utils import get_fz
from ..logger import logger
from ..bayesian_utilities import llike, lpost, lprior, qpsd
from ..splines import knot_locator, dbspline, PSpline, BSpline, get_penalty_matrix


def gibbs_pspline_simple(
    data: np.ndarray,
    Ntotal: int,
    burnin: int, thin: int = 1,
    tau_alpha: float = 0.001,
    tau_beta: float = 0.001,
    phi_alpha: float = 1,
    phi_beta: float = 1,
    delta_alpha: float = 1e-04,
    delta_beta: float = 1e-04,
    k: int = None,
    eqSpacedKnots: bool = False,
    degree: int = 3,
    diffMatrixOrder: int = 2,
    printIter: int = 100
):
    kwargs = locals()
    data, k = _argument_preconditions(**kwargs)
    n = len(data)

    fz = get_fz(data)
    periodogram = abs(fz) ** 2

    # freq lists
    idxs = np.arange(0, n // 2)
    omega = 2 * idxs / n
    lambda_ = np.pi * omega

    # Empty lists for the MCMC samples
    n_samples = round(Ntotal / thin)
    samples = np.zeros((n_samples, 3))
    tau = np.zeros(n_samples)
    phi = np.zeros(n_samples)
    delta = np.zeros(n_samples)

    # starting values #TODO: why these values?
    tau[0] = stats.variation(data) / (2 * np.pi)
    delta[0] = delta_alpha / delta_beta
    phi[0] = phi_alpha / (phi_beta * delta[0])

    # starting value for the weights
    w = periodogram / np.sum(periodogram)
    w = w[np.round(np.linspace(0, len(w), k)).astype(int)]  # TODO: why are we truncating to len k?
    assert len(w) == k
    w[w == 0] = 1e-50  # TODO: why are we doing this?
    w = w / np.sum(w)
    w = w[:-1]  # TODO: why are we ditching last element?
    v = np.log(w / (1 - np.sum(w)))
    V = np.matrix(v).T


    knots = knot_locator(data, k, eqSpacedKnots)
    db_list = dbspline(data, knots, degree)

    if eqSpacedKnots:
        P = diff_matrix(k - 1, d=diffMatrixOrder)
        P = P.T @ P
    else:
        P = get_penalty_matrix(db_list, diffMatrixOrder)
        P = P / np.linalg.norm(P)
    epsilon = 1e-6
    P = P + epsilon * np.eye(P.shape[1]) # P^(-1)=Sigma (Covariance matrix)




    return samples



def diff_matrix(k, d=2):
    assert d < k, "d must be lower than k"
    assert np.all(np.array([d, k])) > 0, "d, k must be +ive ints"
    out = np.eye(k)
    for i in range(d):
        out = np.diff(out)
    return out

def _argument_preconditions(
    data: np.ndarray,
    Ntotal: int,
    burnin: int, thin: int = 1,
    tau_alpha: float = 0.001,
    tau_beta: float = 0.001,
    phi_alpha: float = 1,
    phi_beta: float = 1,
    delta_alpha: float = 1e-04,
    delta_beta: float = 1e-04,
    k: int = None,
    eqSpacedKnots: bool = False,
    degree: int = 3,
    diffMatrixOrder: int = 2,
    printIter: int = 100
):
    assert data.shape[0] > 2, "data must be a non-empty np.array"
    assert burnin < Ntotal, "burnin must be less than Ntotal"
    pos_ints = np.array([thin, Ntotal, burnin, printIter])
    assert np.all(pos_ints >= 0) and np.all(pos_ints % 1 == 0), "thin, Ntotal, burnin must be +ive ints"
    assert Ntotal > 0, "Ntotal must be a positive integer"
    pos_flts = np.array([tau_alpha, tau_beta, phi_alpha, phi_beta, delta_alpha, delta_beta])
    assert np.all(pos_flts > 0), "tau.alpha, tau.beta, phi.alpha, phi.beta, delta.alpha, delta.beta must be +ive"
    assert isinstance(eqSpacedKnots, bool), "eqSpacedKnots must be a boolean"
    assert degree in [0, 1, 2, 3, 4, 5], "degree must be between 0 and 5"
    assert diffMatrixOrder in [0, 1, 2], "diffMatrixOrder must be either 0, 1, or 2"
    assert degree > diffMatrixOrder, "penalty order must be lower than the bspline density degree"

    n = len(data)
    is_even = n % 2 == 0
    # TODO: why is this necessary?
    if is_even:  # remove 0 and last element
        data = data[1:-1]
        n = n - 2
    else:  # remove 0 element
        data = data[1:]
        n = n - 1
    assert n % 2 == 0, "n must be even"

    # ensure mean-centered data
    data_original = data.copy()
    if abs(np.mean(data)) > 1e-4:
        data = data - np.mean(data)
    data = data / np.std(data)
    if not np.allclose(data, data_original):
        logger.warning(
            "data was not mean-centered and/or scaled to unit variance. "
            "This has been done automatically."
        )

    if k is None:
        if eqSpacedKnots:
            k = int(np.ceil(n / 2))
        else:
            k = 10
    assert k >= degree + 2, "k must be at least degree + 2"

    assert (Ntotal - burnin) / thin < k, "Must have (Ntotal-burnin)/thin > k"
    assert k - 2 >= diffMatrixOrder, "diffMatrixOrder must be lower than or equal to k-2"

    return data, k
