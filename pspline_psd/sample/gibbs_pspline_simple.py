import numpy as np
from scipy import sparse
import random
import time
from scipy import stats
from ..utils import get_fz, get_periodogram
from ..logger import logger
from ..bayesian_utilities import llike, lpost, lprior
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
    kwargs.update({'data': data, 'k': k})
    tau0, delta0, phi0, fz, periodogram, V0, omega = _get_initial_values(**kwargs)

    # Empty lists for the MCMC samples
    n_samples = round(Ntotal / thin)
    samples = np.zeros((n_samples, 3))
    tau = np.zeros(n_samples)
    phi = np.zeros(n_samples)
    delta = np.zeros(n_samples)

    # starting values #TODO: why these values? -- shouldnt these be optimized based on lnL?
    tau[0] = tau0
    delta[0] = delta0
    phi[0] = phi0
    knots = knot_locator(data, k, eqSpacedKnots)
    db_list = dbspline(data, knots, degree)

    if eqSpacedKnots:
        P = diff_matrix(k - 1, d=diffMatrixOrder)
        P = P.T @ P
    else:
        P = get_penalty_matrix(db_list, diffMatrixOrder)
        P = P / np.linalg.norm(P)
    epsilon = 1e-6
    P = P + epsilon * np.eye(P.shape[1])  # P^(-1)=Sigma (Covariance matrix)

    ll_trace = None  # log likelihood trace
    count = None  # count of accepted proposals
    sigma = 1  # proposal distribution variance for weights
    count = 0.4  # starting value for count of accepted proposals #TODO: why 0.4?
    k1 = k - 1

    # Random values
    Zs = np.random.normal(size=(Ntotal - 1) * k1)
    Zs = np.reshape(Zs, (Ntotal - 1, k1))
    Us = np.log(np.random.uniform(size=(Ntotal - 1) * k1))
    Us = np.reshape(Us, (Ntotal - 1, k1))

    # initial values for proposal
    phi_store = phi[0]
    tau_store = tau[0]
    delta_store = delta[0]
    V_store = V0[:, 0]
    ptime = time.process_time()

    for i in range(n_samples):

        adj = (j - 1) * thin
        V_star = V_store.copy()
        aux = random.sample(k1, 1)[0]

        for i in range(1, thin + 1):
            iter = i + adj
            if iter % printIter == 0:
                print(
                    "Iteration {}, Time elapsed {} minutes".format(iter, round((time.process_time() - ptime) / 60, 2)))
            f_store = lpost(omega,
                            fz,
                            k,
                            V_store,
                            tau_store,
                            tau_alpha,
                            tau_beta,
                            phi_store,
                            phi_alpha,
                            phi_beta,
                            delta_store,
                            delta_alpha,
                            delta_beta,
                            P,
                            periodogram,
                            degree,
                            db_list)

    return samples


def diff_matrix(k, d=2):
    assert d < k, "d must be lower than k"
    assert np.all(np.array([d, k])) > 0, "d, k must be +ive ints"
    out = np.eye(k)
    for i in range(d):
        out = np.diff(out)
    return out


def _get_initial_values(data, k,
                        phi_alpha: float = 1,
                        phi_beta: float = 1,
                        delta_alpha: float = 1e-04,
                        delta_beta: float = 1e-04, ):
    tau = stats.variation(data) / (2 * np.pi)
    delta = delta_alpha / delta_beta
    phi = phi_alpha / (phi_beta * delta)
    fz = get_fz(data)
    periodogram = get_periodogram(fz)
    V = _generate_initial_weights(data, k)
    n = len(data)
    omega = 2 * np.arange(0, n // 2) / n
    return tau, delta, phi, fz, periodogram, V, omega


def _generate_initial_weights(periodogram, k):
    w = periodogram / np.sum(periodogram)
    w = w[np.round(np.linspace(0, len(w) - 1, k)).astype(int)]  # TODO: why are we truncating to len k?
    assert len(w) == k
    w[w == 0] = 1e-50  # TODO: why are we doing this?
    w = w / np.sum(w)
    w = w[:-1]  # TODO: why are we ditching last element?
    v = np.log(w / (1 - np.sum(w)))
    # convert nans to very small
    v[np.isnan(v)] = -1e50
    V = np.matrix(v).T
    return V


def _format_data(data):
    n = len(data)
    is_even = n % 2 == 0
    # TODO: why is this necessary?
    if is_even:  # remove 0 and last element
        data = data[1:-2]
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
    return data


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

    data = _format_data(data)
    n = len(data)

    if k is None:
        if eqSpacedKnots:
            k = int(np.ceil(n / 2))
        else:
            k = 10
    assert k >= degree + 2, "k must be at least degree + 2"

    assert (Ntotal - burnin) / thin < k, "Must have (Ntotal-burnin)/thin > k"
    assert k - 2 >= diffMatrixOrder, "diffMatrixOrder must be lower than or equal to k-2"

    return data, k
