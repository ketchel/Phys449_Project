import matplotlib.pyplot as plt
from jax import numpy as np
from src.model import eigenpairs


def plot_results(op, results, hyper):
    opt_params, averages, beta, net_u, logs = results
    ndim = hyper["ndim"]
    # Test data
    n_star = 128
    # x_1d = np.linspace(0.0, np.pi, n_star)[:, None]
    x_star = np.linspace(0.0, np.pi, n_star)
    grid = np.meshgrid(x_star, x_star)
    test_input = np.array(grid).T.reshape(-1, ndim)
    evals, efuns = eigenpairs(opt_params, test_input,
                              averages, beta, net_u, op)
    print('Predicted eigenvalues: {}'.format(evals))
    efuns_plot = efuns[:, 0].reshape(128, 128)
    plt.imshow(efuns_plot)
    plt.show()
