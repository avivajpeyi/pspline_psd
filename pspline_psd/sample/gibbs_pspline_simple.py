import numpy as np
from scipy import sparse
from scipy import stats
from ..utils import get_fz
from ..logger import logger


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


    #
    # # create the design matrix
    # if degree == 0:
    #     X = np.ones((n, 1))
    # else:
    #     if eqSpacedKnots:
    #         knots = np.arange(1, n + 1, n / (k + 1)).astype(int)
    #     else:
    #         knots = np.sort(np.random.choice(np.arange(1, n + 1), size=k, replace=False))
    #     X = np.zeros((n, k + degree - 1))
    #     for j in range(k + degree - 1):
    #         X[:, j] = np.power(data, j)
    #     for j in range(1, degree):
    #         for i in range(k):
    #             X[knots[i]:, j * k + i] = np.power(data[knots[i]:] - data[knots[i]], j)
    #
    # # create the difference matrix
    # if diffMatrixOrder == 0:
    #     D = np.eye(k + degree - 2)
    # else:
    #     D = np.zeros((k + degree - diffMatrixOrder - 1, k + degree - 2))
    #     for i in range(k + degree - diffMatrixOrder - 1):
    #         D[i, i:i + diffMatrixOrder + 1] = np.array(
    #             [(-1) ** j * stats.comb(diffMatrixOrder, j) for j in range(diffMatrixOrder + 1)])
    #
    # # initialize the parameters
    # beta = np.zeros(k + degree - 1)
    # alpha = np.zeros(k + degree - 2)
    # phi = 1
    # delta = 1
    #
    # # create the sparse matrix for the Gibbs sampler
    # A = sparse.csc_matrix(np.dot(X.T, X) / phi + D.T.dot(D) / delta)
    #
    # # run the Gibbs sampler
    # samples = np.zeros((int((Ntotal - burnin) / thin), k + degree - 1))
    # for i in range(Ntotal):
    #     # sample beta
    #     beta_mean = np.dot(X.T, data) / phi + np.dot(D.T, alpha) / delta
    #     beta_cov = np.linalg.inv(A)
    #     beta = np.random.multivariate_normal(beta_mean, beta_cov)
    #
    #     # sample alpha
    #     alpha_mean = np.dot(D, beta)
    #     alpha_cov = np.linalg.inv(D.T.dot(D) / delta + tau_alpha * np.eye(k + degree - 2))
    #     alpha = np.random.multivariate_normal(alpha_mean, alpha_cov)
    #
    #     # sample phi
    #     phi_shape = n / 2 + phi_alpha
    #     phi_rate = 0.5 * np.sum(np.power(data - np.dot(X, beta), 2)) + phi_beta
    #     phi = np.random.gamma(phi_shape, 1 / phi_rate)
    #
    #     # sample delta
    #     delta_shape = (k + degree - 2) / 2 + delta_alpha
    #     delta_rate = 0.5 * np.dot(alpha, alpha) + delta_beta
    #     delta = np.random.gamma(delta_shape, 1 / delta_rate)
    #
    #     # save the sample
    #     if i >= burnin and (i - burnin) % thin == 0:
    #         samples[int((i - burnin) / thin), :] = beta

    return samples


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
