import jax.numpy as np
from jax import random, jit, vmap, jacfwd
from jax.experimental import optimizers
from jax.nn import sigmoid, softplus
from jax import tree_multimap
from jax import ops


import itertools
from functools import partial
from torch.utils import data
from tqdm import trange
import matplotlib.pyplot as plt


class SpIN:
    # Initialize the class
    def __init__(self, operator, layers, MLP):

        # Callable operator function
        self.operator = operator

        # Network initialization and evaluation functions
        self.net_init, self.net_apply = MLP(layers)

        # Initialize network parameters
        params = self.net_init(random.PRNGKey(0))

        # Optimizer initialization and update functions
        lr = optimizers.exponential_decay(1e-4, decay_steps=1000, decay_rate=0.9)
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.rmsprop(lr)
        self.opt_state = self.opt_init(params)

        # Decay parameter
        self.beta = 1.0

        # Number of eigenvalues
        self.neig = layers[-1]

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.evals_log = []

    def apply_mask(self, inputs, outputs):
        # mask is used to zero the boundary points.
        mask = 0.1
        if len(inputs.shape) == 2:
            for i in range(inputs.shape[1]):
                mask *= np.maximum((-inputs[:,i]**2 + np.pi * inputs[:,i]), 0)
            mask = np.expand_dims(mask, -1)

        elif len(inputs.shape) == 1:
            for x in inputs:
                mask *= np.maximum((-x ** 2 + np.pi * x ), 0)

        return mask*outputs

    def net_u(self, params, inputs):
        outputs = self.net_apply(params, inputs)
        outputs = self.apply_mask(inputs, outputs)
        return outputs

    def evaluate_spin(self, params, inputs, averages, beta):
        # Fetch batch
        n = inputs.shape[0]
        sigma_avg, _ = averages

        # Evaluate model
        u = self.net_u(params, inputs)
        sigma = np.dot(u.T, u)/n
        sigma_avg = (1.0 - beta) * sigma_avg + beta * sigma # $\bar{\Sigma}$

        # Cholesky
        chol = np.linalg.cholesky(sigma_avg)
        choli = np.linalg.inv(chol) # $L^{-1}$

        # Operator
        operator = self.operator(self.net_u, params, inputs)
        pi = np.dot(operator.T, u)/n # $\Pi$
        rq = np.dot(choli, np.dot(pi, choli.T)) # $\Lambda$

        return (u, choli, pi, rq, operator), sigma_avg

    def masked_gradients(self, params, inputs, outputs, averages, beta):
        # Fetch batch
        n = inputs.shape[0]
        u, choli, _, rq, operator = outputs
        _, sigma_jac_avg = averages

        dl = np.diag(np.diag(choli))
        triu = np.triu(np.dot(rq, dl))

        grad_sigma = -1.0 * np.matmul(choli.T, triu) # \frac{\partial tr(\Lambda)}{\partial \Sigma}
        grad_pi = np.dot(choli.T, dl) # \frac{\partail tr(\Lambda){\partial \Pi}}

        grad_param_pre = jacfwd(self.net_u)
        grad_param = vmap(grad_param_pre, in_axes = (None, 0))

        grad_theta = grad_param(params, inputs) # \frac{\partial u}{\partial \theta}

        sigma_jac = tree_multimap(lambda x:
                                  np.tensordot(u.T, x, 1),
                                  grad_theta) # frac{\partail \Sigma}{\partial \theta}

        sigma_jac_avg = tree_multimap(lambda x,y:
                                      (1.0-beta) * x + beta * y,
                                      sigma_jac_avg,
                                      sigma_jac)

        gradient_pi_1 = np.dot(grad_pi.T, operator.T)

        # gradient  = \frac{\partial tr(\Lambda)}{\partial \theta}
        gradients = tree_multimap(lambda x,y:
                                  (np.tensordot(gradient_pi_1, x, ([0,1],[1,0])) +
                                   1.0 * np.tensordot(grad_sigma.T, y,([0,1],[1,0])))/n,
                                  grad_theta,
                                  sigma_jac_avg)
        # Negate for gradient ascent
        gradients = tree_multimap(lambda x: -1.0*x, gradients)

        return gradients, sigma_jac_avg

    def loss_and_grad(self, params, batch):
        # Fetch batch
        inputs, averages, beta = batch

        # Evaluate SPIN model
        outputs, sigma_avg = self.evaluate_spin(params,
                                                inputs,
                                                averages,
                                                beta)

        # Compute loss
        _, _, _, rq, _ = outputs
        eigenvalues = np.diag(rq)  # eigenvalues are the diagonal entries of $\Lamda$
        loss = np.sum(eigenvalues)

        # Compute masked gradients
        gradients, sigma_jac_avg = self.masked_gradients(params,
                                                         inputs,
                                                         outputs,
                                                         averages,
                                                         beta)

        # Store updated averages
        averages = (sigma_avg, sigma_jac_avg)

        return loss, gradients, averages

    def init_sigma_jac(self, params, inputs):
        u = self.net_u(params, inputs)
        grad_param = jacfwd(self.net_u)
        grad_theta = grad_param(params, inputs)
        sigma_jac = tree_multimap(lambda x: np.tensordot(u.T, x, 1),
                                          grad_theta)
        return sigma_jac


    # Define a jit-compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        loss, gradients, averages = self.loss_and_grad(params, batch)
        opt_state = self.opt_update(i, gradients, opt_state)
        return loss, opt_state, averages

    # Optimize parameters in a loop
    def train(self, dataset, nIter = 10000):
        inputs = iter(dataset)
        pbar = trange(nIter)

        # Initialize moving averages
        sigma_avg = np.ones(self.neig)
        sigma_jac_avg = self.init_sigma_jac(self.get_params(self.opt_state), next(inputs))
        averages = (sigma_avg, sigma_jac_avg)

        # Main training loop
        for it in pbar:
            # Set beta
            cnt = next(self.itercount)
            beta = self.beta if cnt > 0 else 1.0

            # Create batch
            batch = next(inputs), averages, beta

            # Run one gradient descent update
            loss, self.opt_state, averages = self.step(cnt, self.opt_state, batch)

            # Logger
            params = self.get_params(self.opt_state)
            evals, _ = self.eigenpairs(params, next(inputs), averages, beta)
            self.loss_log.append(loss)
            self.evals_log.append(evals)
            pbar.set_postfix({'Loss': loss})

        return params, averages, beta


    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def eigenpairs(self, params, inputs, averages, beta):
        outputs, _ = self.evaluate_spin(params, inputs, averages, beta)
        u, choli, _, rq, _ = outputs
        evals = np.diag(rq)
        efuns = np.matmul(u, choli.T)
        return evals, efuns