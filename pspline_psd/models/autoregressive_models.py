import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample

def ar1(rho: float, sigma: float, y0: float, n: int) -> np.array:
    """
    Simulate an AR(1) process with parameters rho and sigma, starting at y0.

    Args:
        rho (float): AR(1) parameter
        sigma (float): standard deviation of the noise
        y0 (float): initial condition
        n (int): number of time steps to simulate

    Returns:
        y (np.array): simulated AR(1) process
    """
    # Allocate space and draw epsilons
    y = np.empty(n)
    eps = np.random.normal(0, sigma, n)

    # Initial condition and step forward
    y[0] = y0
    for t in range(1, n):
        y[t] = rho * y[t - 1] + eps[t]

    return y
