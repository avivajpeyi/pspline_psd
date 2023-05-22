import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def test_gp(helpers):
    dataset = helpers.load_data_0()
    data = dataset['data']
    periodogram = dataset['pdgrm']
    idx = np.array([i for i in range(len(periodogram))])
    plt.plot(periodogram)

    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, α=0.01)
    gaussian_process.fit(idx.reshape(-1, 1), periodogram)

    x = np.linspace(min(idx), max(idx), 1000).reshape(-1, 1)
    mean_prediction, std_prediction = gaussian_process.predict(x, return_std=True)
    plt.plot(x, mean_prediction, label="Mean prediction", color='C2')
    plt.fill_between(
        x.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        α=0.5,
        label=r"95% confidence interval",
        color='C2',
    )
    plt.show()
