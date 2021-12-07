from jax import numpy as np
from jax import vmap, jacfwd
from jax.config import config
from functools import partial
from jax import jit
import os

use_gpu = True
if use_gpu:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
    config.update("jax_platform_name", "gpu")
    print("using gpu as device\n")
else:
    config.update("jax_platform_name", "cpu")
    print("using cpu as device\n")


@partial(jit, static_argnums=(0))
def laplacian_1d(u_fn, params, inputs):
    def action(params, x):
        u_xx = jacfwd(jacfwd(u_fn, 1), 1)(params, x)
        return u_xx
    vec_fun = vmap(action, in_axes=(None, 0))
    laplacian = vec_fun(params, inputs)
    return np.squeeze(laplacian)


@partial(jit, static_argnums=(0))
def laplacian_2d(u_fn, params, inputs):
    def fun(params, x, y): return u_fn(params, np.array([x, y]))

    def action(params, x, y):
        u_xx = jacfwd(jacfwd(fun, 1), 1)(params, x, y)
        u_yy = jacfwd(jacfwd(fun, 2), 2)(params, x, y)
        return u_xx + u_yy
    vec_fun = vmap(action, in_axes=(None, 0, 0))
    laplacian = vec_fun(params, inputs[:, 0], inputs[:, 1])
    return laplacian


@partial(jit, static_argnums=(0))
def schrodinger_1d(u_fn, params, inputs):
    def action(params, x):
        u_xx = jacfwd(jacfwd(u_fn, 1), 1)(params, x)
        return u_xx - (u_fn(params, x) / np.linalg.norm(x))
    vec_fun = vmap(action, in_axes=(None, 0))
    laplacian = vec_fun(params, inputs)
    return np.squeeze(laplacian)


@partial(jit, static_argnums=(0))
def schrodinger_2d(u_fn, params, inputs):
    def fun(params, x, y): return u_fn(params, np.array([x, y]))

    def action(params, x, y):
        u_xx = jacfwd(jacfwd(fun, 1), 1)(params, x, y)
        u_yy = jacfwd(jacfwd(fun, 2), 2)(params, x, y)
        return u_xx + u_yy - (fun(params, x, y) / np.linalg.norm([x, y]))
    vec_fun = vmap(action, in_axes=(None, 0, 0))
    schrodinger = vec_fun(params, inputs[:, 0], inputs[:, 1])
    return schrodinger


def get_operator(hyper):
    ndim = hyper["ndim"]
    if hyper["operator"] == "laplacian":
        ndim = hyper["ndim"]
        if ndim == 1:
            op = laplacian_1d
        elif ndim == 2:
            op = laplacian_2d
        else:
            raise Exception(
                "dimensions other than 1 or 2 are not supported yet.")
    elif hyper["operator"] == "schrodinger":
        if ndim == 1:
            op = schrodinger_1d
        elif ndim == 2:
            op = schrodinger_2d
        else:
            raise Exception(
                "dimensions other than 1 or 2 are not supported yet.")
    else:
        raise Exception("Operator not defined. avail operators:"
                        + "laplacian_1d, laplacian_2d, schrodinger")
    return op
