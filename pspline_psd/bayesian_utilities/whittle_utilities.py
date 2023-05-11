# TODO: convert to cpp/cython
import numpy as np

def densityMixture(weights, densities):
    if len(weights) != densities.shape[0]:
        return np.array([])
    n = densities.shape[1]
    res = np.zeros(n)
    for omega in range(n):
        res[omega] = 0.0
    for j in range(len(weights)):
        for omega in range(n):
            res[omega] += weights[j] * densities[j, omega]
    return res

def unrollPsd(qPsd, n):
    q = np.zeros(n)
    q[0] = qPsd[0]
    N = (n-1)//2
    for i in range(1, N+1):
        j = 2 * i - 1
        q[j] = qPsd[i]
        q[j+1] = qPsd[i]
    if not (n % 2):
        q[n-1] = qPsd[-1]
    return q
