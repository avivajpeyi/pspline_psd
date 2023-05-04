import numpy as np
from pspline_psd.models import ar1
import matplotlib.pyplot as plt
from pspline_psd.sample.gibbs_pspline_simple import gibbs_pspline_simple

def test_simple_example():
    np.random.seed(0)  # for reproducibility
    data = ar1(rho=0.9, sigma=1, y0=0, n=100)
    data = data - np.mean(data)

    samples = gibbs_pspline_simple(
        data=data,
        Ntotal=1000,
        burnin=100,
        degree=3
    )
    print(samples)
