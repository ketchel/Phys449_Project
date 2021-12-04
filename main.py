import matplotlib.pyplot as plt
from jax import numpy as np
from src.nn_gen import MLP
from src.params import get_params
from src.util import laplacian_2d
from src.data_gen import DataGenerator
from src.model import SpIN


def main():
    hyper = get_params()
    # Problem setup
    ndim = hyper["ndim"]

    # Create data sampler
    dataset = DataGenerator(hyper)

    model = SpIN(laplacian_2d, MLP, hyper)
    opt_params, averages, beta = model.train(dataset, nIter=6000)

    # Test data
    n_star = 128
    # x_1d = np.linspace(0.0, np.pi, n_star)[:, None]
    x_star = np.linspace(0.0, np.pi, n_star)
    grid = np.meshgrid(x_star, x_star)
    test_input = np.array(grid).T.reshape(-1, ndim)

    evals, efuns = model.eigenpairs(opt_params, test_input, averages, beta)
    print('Predicted eigenvalues: {}'.format(evals))

    efuns_plot = efuns[:, 0].reshape(128, 128)
    plt.imshow(efuns_plot)
    plt.show()


if __name__ == '__main__':
    main()
