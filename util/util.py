import jax.numpy as np
from jax import random, jit, vmap, jacfwd
from jax.experimental import optimizers
from jax.nn import sigmoid, softplus
from jax import tree_multimap
from jax import ops


def laplacian_1d(u_fn, params, inputs):
    def action(params, inputs):
        u_xx = jacfwd(jacfwd(u_fn, 1), 1)(params, inputs)
        return u_xx
    vec_fun = vmap(action, in_axes = (None, 0))
    laplacian = vec_fun(params, inputs)
    return np.squeeze(laplacian)


def laplacian_2d(u_fn, params, inputs):
    fun = lambda params,x,y: u_fn(params, np.array([x,y]))
    def action(params,x,y):
        u_xx = jacfwd(jacfwd(fun, 1), 1)(params,x,y)
        u_yy = jacfwd(jacfwd(fun, 2), 2)(params,x,y)
        return u_xx + u_yy
    vec_fun = vmap(action, in_axes = (None, 0, 0))
    laplacian = vec_fun(params, inputs[:,0], inputs[:,1])
    return laplacian


def MLP(layers):
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W = random.normal(k1, (d_in, d_out))
            b = random.normal(k2, (d_out,))
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = sigmoid(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs
    return init, apply


# def plotting_function():
#   evals, efuns = model.eigenpairs(opt_params, test_input, averages, beta)
#   print('Predicted eigenvalues: {}'.format(evals))
#   evals_true, efuns_true = exact_eigenpairs_1d(x_star[:,None], neig)
#   print('True eigenvalues: {}'.format(evals_true))

#  efuns_plot = efuns[:,0].reshape(128,128)
#   plt.imshow(efuns_plot)