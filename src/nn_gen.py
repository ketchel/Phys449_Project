from jax import numpy as np
from jax import random
from jax.nn import sigmoid


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

    def apply(params, x):
        for i, (W, b) in enumerate(params):
            x = np.dot(x, W) + b
            if i != len(params) - 1:
                x = sigmoid(x)
        return x

    return init, apply
