from jax import random


class DataGenerator():
    def __init__(self, hyper, key=random.PRNGKey(1234)):
        self.dim = hyper["ndim"]
        self.min = hyper["box_min"]
        self.max = hyper["box_max"]
        self.batch_size = hyper["batch_size"]
        self.key = key

    def __getitem__(self, index):
        self.key, subkey = random.split(self.key)
        return random.uniform(subkey, (self.batch_size, self.dim),
                              minval=self.min, maxval=self.max)
