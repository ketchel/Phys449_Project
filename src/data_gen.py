from jax import random, jit
from functools import partial
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    def __init__(self, hyper, key=random.PRNGKey(1234)):
        self.dim = hyper["ndim"]
        self.min = hyper["box_min"]
        self.max = hyper["box_max"]
        self.batch_size = hyper["batch_size"]
        self.delta = self.max - self.min
        self.key = key

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        inputs = self.min + self.delta * \
            random.uniform(key, (self.batch_size, self.dim))
        return inputs

    def __getitem__(self, index):
        self.key, subkey = random.split(self.key)
        X = self.__data_generation(subkey)
        return X
