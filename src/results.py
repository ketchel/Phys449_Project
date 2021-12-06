import matplotlib.pyplot as plt
from jax import numpy as np
import numpy
from src.spin import eigpairs
from pandas import DataFrame
import os


def save_results(op, results, hyper):
    params, avrgs, beta, fnet, logs = results
    loss_log, evals_log = logs
    ndim = hyper["ndim"]
    neig = hyper["neig"]
    results = hyper["results"]
    grid_size = hyper["grid_size"]
    llim = hyper["box_min"]
    hlim = hyper["box_max"]
    parent_dir = os.getcwd()
    results_dir = os.path.join(parent_dir, results)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Test data
    if ndim == 1:
        test_input = np.linspace(llim, hlim, grid_size)[:, None]
    elif ndim == 2:
        x_star = np.linspace(llim, hlim, grid_size)
        grid = np.meshgrid(x_star, x_star)
        test_input = np.array(grid).T.reshape(-1, ndim)
    else:
        raise Exception("dimensions other than 1 or 2 are not supported yet.")

    evals, efuns = eigpairs(fnet, op, params, test_input, avrgs, beta)

    print('Predicted eigenvalues: {}'.format(evals))
    numpy.save(os.path.join(results_dir, 'loss'), loss_log)
    numpy.save(os.path.join(results_dir, 'evals'), evals_log)
    DataFrame(loss_log).to_csv(os.path.join(results_dir, 'loss.csv'),
                               header=False, index=False)
    DataFrame(evals_log).to_csv(os.path.join(results_dir, 'evals.csv'),
                                header=False, index=False)

    for i in range(neig):
        img = efuns[:, i].reshape(grid_size, grid_size)
        plt.imsave(os.path.join(results_dir, 'efun' + str(i+1) + '.png'), img)
    plt.subplot(1,2,1)
    plt.plot(loss_log)
    plt.subplot(1, 2, 2)
    plt.plot(evals_log)
    plt.savefig(os.path.join(results_dir, 'graphs' + '.png'))


    for i, (W, b) in enumerate(params):
        shape = W.shape
        arr = numpy.zeros((shape[0] + 1, shape[1]))
        arr[0, :] = b[:]
        arr[1:, :] = W
        numpy.save(os.path.join(results_dir, 'param' + str(i+1)), arr)
        DataFrame(arr).to_csv(
            os.path.join(results_dir, 'param' + str(i+1) + '.csv'),
            header=False, index=False)
