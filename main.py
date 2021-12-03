import matplotlib.pyplot as plt
from jax import numpy as np
from src.util import laplacian_2d, MLP
from src.data_gen import Sampler, DataGenerator
from src.model import SpIN
import argparse
import json


def get_params():
    # Default values
    lr = 1e-4
    ndim = 2
    neig = 4
    num_iters = 15000
    num_layers = 4
    num_hidden = 64
    batch_size = 64
    results = "results"
    param = None

    parser = argparse.ArgumentParser(
        description="Jax-Based Spectral Inference Network")
    parser.add_argument('--param', default=param, help='parameter file name')

    args = parser.parse_args()
    param = args.param

    if param:
        f = open(args.param, "r")
        hyper = json.loads(f.read())
    else:
        hyper = {
            "lr": lr,
            "num_iters": num_iters,
            "batch_size": batch_size,
            "ndim": ndim,
            "neig": neig,
            "num_hidden": num_hidden,
            "num_layers": num_layers,
            "results": results,
        }
    return hyper


if __name__ == '__main__':
    hyper = get_params()
    # Problem setup
    ndim = hyper["ndim"]
    neig = hyper["neig"]

    # Domain boundaries
    dom_coords = np.array([[0, np.pi]])

    # Create data sampler
    dom_sampler = Sampler(ndim, dom_coords)

    dataset = DataGenerator(dom_sampler, batch_size=128)

    # Test data
    n_star = 128
    x_1d = np.linspace(0.0, np.pi, n_star)[:, None]
    x_star = np.linspace(0.0, np.pi, n_star)
    grid = np.meshgrid(x_star, x_star)
    test_input = np.array(grid).T.reshape(-1, ndim)

    layers = [ndim, 64, 64, 64, 32, neig]
    model = SpIN(laplacian_2d, layers, MLP)
    opt_params, averages, beta = model.train(dataset, nIter=6000)

    evals, efuns = model.eigenpairs(opt_params, test_input, averages, beta)
    print('Predicted eigenvalues: {}'.format(evals))

    efuns_plot = efuns[:, 3].reshape(128, 128)
    plt.imshow(efuns_plot)
    plt.show()
