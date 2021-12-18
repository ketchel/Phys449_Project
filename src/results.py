import matplotlib.pyplot as plt
from jax import numpy as np
from numpy import zeros, save
from pandas import DataFrame
from os import getcwd, mkdir, path
from shutil import rmtree


def save_results(op, results, hyper):
    params, eigen, losses, evals = results
    ndim = hyper["ndim"]
    neig = hyper["neig"]
    results = hyper["results"]
    grid_size = hyper["grid_size"]
    box_min = hyper["box_min"]
    box_max = hyper["box_max"]

    # Setup path
    parent_dir = getcwd()
    results_dir = path.join(parent_dir, results)
    if path.exists(results_dir):
        rmtree(results_dir)
    mkdir(results_dir)

    # Test data
    if ndim == 1:
        test_input = np.linspace(box_min, box_max, grid_size)[:, None]
    elif ndim == 2:
        x_star = np.linspace(box_min, box_max, grid_size)
        grid = np.meshgrid(x_star, x_star)
        test_input = np.array(grid).T.reshape(-1, ndim)
    else:
        raise Exception("dimensions other than 1 or 2 are not supported yet.")

    evals, efuns = eigen(params, test_input)

    print('Predicted eigenvalues: {}'.format(evals))
    save(path.join(results_dir, 'loss'), losses)
    save(path.join(results_dir, 'evals'), evals)
    DataFrame(losses).to_csv(path.join(results_dir, 'loss.csv'),
                             header=False, index=False)
    DataFrame(evals).to_csv(path.join(results_dir, 'evals.csv'),
                            header=False, index=False)

    if ndim == 1:
        xpts = np.linspace(box_min, box_max, grid_size)
        plt.plot(xpts, efuns)
        plt.ylabel('Eigenfunctions')
        plt.xlabel('Domain')
        the_path = path.join(results_dir, 'efuns.png')
        plt.savefig(the_path)
    elif ndim == 2:
        for i in range(neig):
            img = efuns[:, i].reshape(grid_size, grid_size)
            the_path = path.join(results_dir, 'efun' + str(i+1) + '.png')
            plt.imsave(the_path, img)
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('iteration')
    plt.savefig(path.join(results_dir, 'Loss' + '.png'))
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(evals)
    plt.ylabel('Eigenvalues')
    plt.xlabel('iteration')
    plt.savefig(path.join(results_dir, 'Eigenvalues' + '.png'))
    plt.clf()
    plt.cla()
    plt.close()

    for i, (W, b) in enumerate(params):
        shape = W.shape
        arr = zeros((shape[0] + 1, shape[1]))
        arr[0, :] = b[:]
        arr[1:, :] = W
        the_path = path.join(results_dir, 'layer' + str(i+1))
        save(the_path, arr)
        DataFrame(arr).to_csv(the_path + '.csv', header=False, index=False)
