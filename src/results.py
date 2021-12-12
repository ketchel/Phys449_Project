import matplotlib.pyplot as plt
from jax import numpy as np
import numpy
from src.spin import eigen
from pandas import DataFrame
import os
import shutil


def save_results(op, results, hyper):
    params, Sigma_avg, _, beta, fnet, loss_log, evals_log = results
    ndim = hyper["ndim"]
    neig = hyper["neig"]
    results = hyper["results"]
    grid_size = hyper["grid_size"]
    box_min = hyper["box_min"]
    box_max = hyper["box_max"]

    # Setup path
    parent_dir = os.getcwd()
    results_dir = os.path.join(parent_dir, results)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    # Test data
    if ndim == 1:
        test_input = np.linspace(box_min, box_max, grid_size)[:, None]
    elif ndim == 2:
        x_star = np.linspace(box_min, box_max, grid_size)
        grid = np.meshgrid(x_star, x_star)
        test_input = np.array(grid).T.reshape(-1, ndim)
    else:
        raise Exception("dimensions other than 1 or 2 are not supported yet.")

    evals, efuns = eigen(fnet, op, params, test_input, Sigma_avg, beta)

    print('Predicted eigenvalues: {}'.format(evals))
    numpy.save(os.path.join(results_dir, 'loss'), loss_log)
    numpy.save(os.path.join(results_dir, 'evals'), evals_log)
    DataFrame(loss_log).to_csv(os.path.join(results_dir, 'loss.csv'),
                               header=False, index=False)
    DataFrame(evals_log).to_csv(os.path.join(results_dir, 'evals.csv'),
                                header=False, index=False)

    if ndim == 1:
        xpts = np.linspace(box_min, box_max, grid_size)
        plt.plot(xpts, efuns)
        plt.ylabel('Eigenfunctions')
        plt.xlabel('Domain')
        path = os.path.join(results_dir, 'efuns.png')
        plt.savefig(path)
    elif ndim == 2:
        for i in range(neig):
            img = efuns[:, i].reshape(grid_size, grid_size)
            path = os.path.join(results_dir, 'efun' + str(i+1) + '.png')
            plt.imsave(path, img)
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(loss_log)
    plt.ylabel('Loss')
    plt.xlabel('iteration')
    plt.savefig(os.path.join(results_dir, 'Loss' + '.png'))
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(evals_log)
    plt.ylabel('Eigenvalues')
    plt.xlabel('iteration')
    plt.savefig(os.path.join(results_dir, 'Eigenvalues' + '.png'))
    plt.clf()
    plt.cla()
    plt.close()

    for i, (W, b) in enumerate(params):
        shape = W.shape
        arr = numpy.zeros((shape[0] + 1, shape[1]))
        arr[0, :] = b[:]
        arr[1:, :] = W
        path = os.path.join(results_dir, 'layer' + str(i+1))
        numpy.save(path, arr)
        DataFrame(arr).to_csv(path + '.csv', header=False, index=False)
