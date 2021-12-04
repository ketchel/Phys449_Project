from jax import numpy as np
import argparse
import json


def get_params():
    # Default values
    lr = 1e-4
    box_min = 0.0
    box_max = np.pi
    ndim = 2
    neig = 4
    num_iters = 15000
    num_layers = 4
    num_hidden = 64
    batch_size = 64
    grid_size = 128
    results = "results"
    param = None

    parser = argparse.ArgumentParser(
        description="Jax-Based Spectral Inference Network")
    parser.add_argument('--param', default=param, help='parameter file name')
    args = parser.parse_args()
    param = args.param

    if param is not None:
        f = open(args.param, "r")
        hyper = json.loads(f.read())
    else:
        hyper = {
            "lr": lr,
            "box_min": box_min,
            "box_max": box_max,
            "ndim": ndim,
            "neig": neig,
            "num_iters": num_iters,
            "num_layers": num_layers,
            "num_hidden": num_hidden,
            "batch_size": batch_size,
            "results": results,
            "grid_size": grid_size
        }
    return hyper
