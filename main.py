import jax.numpy as np
from jax import random, jit, vmap, jacfwd
from jax.experimental import optimizers
from jax.nn import sigmoid, softplus
from jax import tree_multimap
from jax import ops

from src.model import SpIN
from util.util import MLP
from util.util import laplacian_2d
from src.data_gen import Sampler, DataGenerator

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange
import matplotlib.pyplot as plt

if __name__ == '__main__':
  # Problem setup
  ndim = 2
  neig = 4

  # Domain boundaries
  dom_coords = np.array([[0, np.pi]])

  # Create data sampler
  dom_sampler = Sampler(ndim, dom_coords)

  dataset = DataGenerator(dom_sampler, batch_size=128)

  # Test data
  n_star = 128
  x_1d = np.linspace(0.0, np.pi, n_star)[:,None]
  x_star = np.linspace(0.0, np.pi, n_star)
  grid = np.meshgrid(x_star, x_star)
  test_input = np.array(grid).T.reshape(-1, ndim)

  layers = [ndim, 64, 64, 64, 32, neig]
  model = SpIN(laplacian_2d, layers, MLP)
  opt_params, averages, beta = model.train(dataset, nIter = 6000)

  evals, efuns = model.eigenpairs(opt_params, test_input, averages, beta)
  print('Predicted eigenvalues: {}'.format(evals))

  efuns_plot = efuns[:,3].reshape(128,128)
  plt.imshow(efuns_plot)
