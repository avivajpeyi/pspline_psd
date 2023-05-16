# TODO: convert to cpp/cython
import numpy as np


def psd_model(v: np.ndarray, db_list: np.ndarray, n: int):
    unorm_psd = get_unormalised_psd(v, db_list)
    return unroll_psd(unorm_psd, n)


def get_unormalised_psd(v: np.ndarray, db_list: np.ndarray):
    """Compute unnormalised PSD using random mixture of B-splines

    Parameters
    ----------
    v : np.ndarray
        Vector of spline coefficients (length k)

    db_list : np.ndarray
        Matrix of B-spline basis functions (k x n)

    Returns
    -------
    psd : np.ndarray

    """
    v = np.array(v)
    expV = np.exp(v)

    if np.any(np.isinf(expV)):
        ls = np.logaddexp(0, v)
        weight = np.exp(v - ls)
    else:
        ls = 1 + np.sum(expV)
        weight = expV / ls

    s = 1 - np.sum(weight)
    weight = np.append(weight, 0 if s < 0 else s).ravel()

    psd = density_mixture(densities=db_list.toarray().T, weights=weight)
    epsilon = 1e-20
    psd = np.maximum(psd, epsilon)

    return psd


def density_mixture(weights: np.ndarray, densities: np.ndarray) -> np.ndarray:
    """build a density mixture, given mixture weights and densities"""
    assert len(weights) == densities.shape[0], "weights and densities must have the same length"
    n = densities.shape[1]
    res = np.zeros(n)
    for i in range(len(weights)):
        for j in range(n):
            res[j] += weights[i] * densities[i, j]
    return res


def unroll_psd(qPsd, n):
    """unroll PSD from qPsd to psd of length n"""
    q = np.zeros(n)
    odd_len = n % 2 == 1
    q[0] = qPsd[0]
    N = (n - 1) // 2
    for i in range(1, N + 1):
        j = 2 * i - 1
        q[j] = qPsd[i]
        q[j + 1] = qPsd[i]

    if odd_len:
        q[-1] = qPsd[-1]
    return q
