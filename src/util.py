from jax import numpy as np
from jax import vmap, jacfwd
from jax.config import config
import os

use_gpu = False
if use_gpu:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
    config.update("jax_platform_name", "gpu")
else:
    config.update("jax_platform_name", "cpu")


def laplacian_1d(u_fn, params, inputs):
    def action(params, inputs):
        u_xx = jacfwd(jacfwd(u_fn, 1), 1)(params, inputs)
        return u_xx
    vec_fun = vmap(action, in_axes=(None, 0))
    laplacian = vec_fun(params, inputs)
    return np.squeeze(laplacian)


def laplacian_2d(u_fn, params, inputs):
    def fun(params, x, y): return u_fn(params, np.array([x, y]))

    def action(params, x, y):
        u_xx = jacfwd(jacfwd(fun, 1), 1)(params, x, y)
        u_yy = jacfwd(jacfwd(fun, 2), 2)(params, x, y)
        return u_xx + u_yy
    vec_fun = vmap(action, in_axes=(None, 0, 0))
    laplacian = vec_fun(params, inputs[:, 0], inputs[:, 1])
    return laplacian
