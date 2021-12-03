from jax import random, jit
from functools import partial
from torch.utils import data


class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, name=None):
        self.dim = dim
        self.coords = coords
        self.name = name

    def sample(self, N, key=random.PRNGKey(1234)):
        min = self.coords.min(1)
        diff = self.coords.max(1) - self.coords.min(1)
        x = min + diff*random.uniform(key, (N, self.dim))
        return x


class DataGenerator(data.Dataset):
    def __init__(self, dom_sampler, mu_X=0.0, sigma_X=1.0, batch_size=64):
        'Initialization'
        self.mu_X = mu_X
        self.sigma_X = sigma_X
        self.dom_sampler = dom_sampler
        self.batch_size = batch_size
        self.key = random.PRNGKey(1234)

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        inputs = self.dom_sampler.sample(self.batch_size, key)
        return inputs

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        X = self.__data_generation(subkey)
        return X
