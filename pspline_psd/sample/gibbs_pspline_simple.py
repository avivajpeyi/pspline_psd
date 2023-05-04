import numpy as np
from scipy import sparse
from scipy import stats
from ..utils import fast_ft


def gibbs_pspline_simple(data, Ntotal, burnin, thin=1, tau_alpha=0.001, tau_beta=0.001, phi_alpha=1, phi_beta=1,
                         delta_alpha=1e-04, delta_beta=1e-04, k=None, eqSpacedKnots=False, degree=3, diffMatrixOrder=2,
                         printIter=100):
    if burnin >= Ntotal:
        raise ValueError("burnin must be less than Ntotal")
    if any(np.array([Ntotal, thin]) % 1 != 0) or any(np.array([Ntotal, thin]) <= 0):
        raise ValueError("Ntotal and thin must be strictly positive integers")
    if (burnin % 1 != 0) or (burnin < 0):
        raise ValueError("burnin must be a non-negative integer")
    if any(np.array([tau_alpha, tau_beta]) <= 0):
        raise ValueError("tau.alpha and tau.beta must be strictly positive")
    if any(np.array([phi_alpha, phi_beta]) <= 0):
        raise ValueError("phi.alpha and phi.beta must be strictly positive")
    if any(np.array([delta_alpha, delta_beta]) <= 0):
        raise ValueError("delta.alpha and delta.beta must be strictly positive")
    if not isinstance(eqSpacedKnots, bool):
        raise TypeError("eqSpacedKnots must be a logical value")
    n = len(data)

    if n % 2:
        # if n is odd, remove the last element
        data = data[:-1]
        n = n - 1

    if k is None:
        if eqSpacedKnots:
            k = int(np.ceil(n / 2))
        else:
            k = 10

    if degree < 0 or degree > 5:
        raise ValueError("degree must be between 0 and 5")

    if diffMatrixOrder < 0 or diffMatrixOrder > 2:
        raise ValueError("diffMatrixOrder must be either 0, 1, or 2")

    # create the design matrix
    if degree == 0:
        X = np.ones((n, 1))
    else:
        if eqSpacedKnots:
            knots = np.arange(1, n + 1, n / (k + 1)).astype(int)
        else:
            knots = np.sort(np.random.choice(np.arange(1, n + 1), size=k, replace=False))
        X = np.zeros((n, k + degree - 1))
        for j in range(k + degree - 1):
            X[:, j] = np.power(data, j)
        for j in range(1, degree):
            for i in range(k):
                X[knots[i]:, j * k + i] = np.power(data[knots[i]:] - data[knots[i]], j)

    # create the difference matrix
    if diffMatrixOrder == 0:
        D = np.eye(k + degree - 2)
    else:
        D = np.zeros((k + degree - diffMatrixOrder - 1, k + degree - 2))
        for i in range(k + degree - diffMatrixOrder - 1):
            D[i, i:i + diffMatrixOrder + 1] = np.array(
                [(-1) ** j * stats.comb(diffMatrixOrder, j) for j in range(diffMatrixOrder + 1)])

    # initialize the parameters
    beta = np.zeros(k + degree - 1)
    alpha = np.zeros(k + degree - 2)
    phi = 1
    delta = 1

    # create the sparse matrix for the Gibbs sampler
    A = sparse.csc_matrix(np.dot(X.T, X) / phi + D.T.dot(D) / delta)

    # run the Gibbs sampler
    samples = np.zeros((int((Ntotal - burnin) / thin), k + degree - 1))
    for i in range(Ntotal):
        # sample beta
        beta_mean = np.dot(X.T, data) / phi + np.dot(D.T, alpha) / delta
        beta_cov = np.linalg.inv(A)
        beta = np.random.multivariate_normal(beta_mean, beta_cov)

        # sample alpha
        alpha_mean = np.dot(D, beta)
        alpha_cov = np.linalg.inv(D.T.dot(D) / delta + tau_alpha * np.eye(k + degree - 2))
        alpha = np.random.multivariate_normal(alpha_mean, alpha_cov)

        # sample phi
        phi_shape = n / 2 + phi_alpha
        phi_rate = 0.5 * np.sum(np.power(data - np.dot(X, beta), 2)) + phi_beta
        phi = np.random.gamma(phi_shape, 1 / phi_rate)

        # sample delta
        delta_shape = (k + degree - 2) / 2 + delta_alpha
        delta_rate = 0.5 * np.dot(alpha, alpha) + delta_beta
        delta = np.random.gamma(delta_shape, 1 / delta_rate)

        # save the sample
        if i >= burnin and (i - burnin) % thin == 0:
            samples[int((i - burnin) / thin), :] = beta

    return samples
