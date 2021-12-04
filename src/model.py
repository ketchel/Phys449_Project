from jax import numpy as np
from jax import jit
from jax import random, vmap, jacfwd
from jax import tree_multimap
import itertools
from functools import partial
from tqdm import trange
import optax


def net_u_full(params, inputs, net_apply):
    """
    """
    outputs = net_apply(params, inputs)
    outputs = apply_mask(inputs, outputs)
    return outputs


@jit
def apply_mask(inputs, outputs):
    # mask is used to zero the boundary points.
    mask = 0.1
    if len(inputs.shape) == 2:
        for i in range(inputs.shape[1]):
            mask *= np.maximum((-inputs[:, i] **
                               2 + np.pi * inputs[:, i]), 0)
        mask = np.expand_dims(mask, -1)
    elif len(inputs.shape) == 1:
        for x in inputs:
            mask *= np.maximum((-x ** 2 + np.pi * x), 0)
    return mask*outputs


def evaluate_spin(net_u, operator, params, inputs, averages, beta):
    # Fetch batch
    n = inputs.shape[0]
    sigma_avg, _ = averages

    # Evaluate model
    u = net_u(params, inputs)
    sigma = np.dot(u.T, u)/n
    sigma_avg = (1.0 - beta) * sigma_avg + beta * sigma  # $\bar{\Sigma}$

    # Cholesky
    chol = np.linalg.cholesky(sigma_avg)
    choli = np.linalg.inv(chol)  # $L^{-1}$

    # Operator
    operator = operator(net_u, params, inputs)
    pi = np.dot(operator.T, u)/n  # $\Pi$
    rq = np.dot(choli, np.dot(pi, choli.T))  # $\Lambda$

    return (u, choli, pi, rq, operator), sigma_avg


def masked_gradients(net_u, params, inputs, outputs, averages, beta):
    # Fetch batch
    n = inputs.shape[0]
    u, choli, _, rq, operator = outputs
    _, sigma_jac_avg = averages

    dl = np.diag(np.diag(choli))
    triu = np.triu(np.dot(rq, dl))

    # \frac{\partial tr(\Lambda)}{\partial \Sigma}
    grad_sigma = -1.0 * np.matmul(choli.T, triu)
    # \frac{\partail tr(\Lambda){\partial \Pi}}
    grad_pi = np.dot(choli.T, dl)

    grad_param_pre = jacfwd(net_u)
    grad_param = vmap(grad_param_pre, in_axes=(None, 0))

    # \frac{\partial u}{\partial \theta}
    grad_theta = grad_param(params, inputs)

    # frac{\partail \Sigma}{\partial \theta}
    sigma_jac = tree_multimap(lambda x:
                              np.tensordot(u.T, x, 1),
                              grad_theta)

    sigma_jac_avg = tree_multimap(lambda x, y:
                                  (1.0-beta) * x + beta * y,
                                  sigma_jac_avg,
                                  sigma_jac)

    gradient_pi_1 = np.dot(grad_pi.T, operator.T)

    # gradient  = \frac{\partial tr(\Lambda)}{\partial \theta}
    gradients = tree_multimap(
        lambda x, y:
        (np.tensordot(gradient_pi_1, x, ([0, 1], [1, 0]))
            + 1.0*np.tensordot(grad_sigma.T, y, ([0, 1], [1, 0])))/n,
        grad_theta, sigma_jac_avg)
    # Negate for gradient ascent
    gradients = tree_multimap(lambda x: -1.0*x, gradients)

    return gradients, sigma_jac_avg


@partial(jit, static_argnums=(0))
def init_sigma_jac(net_u, params, inputs):
    u = net_u(params, inputs)
    grad_param = jacfwd(net_u)
    grad_theta = grad_param(params, inputs)
    sigma_jac = tree_multimap(lambda x: np.tensordot(u.T, x, 1),
                              grad_theta)
    return sigma_jac


def eigenpairs(params, inputs, averages, beta, net_u, operator):
    outputs, _ = evaluate_spin(net_u, operator, params, inputs, averages, beta)
    u, choli, _, rq, _ = outputs
    evals = np.diag(rq)
    efuns = np.matmul(u, choli.T)
    return evals, efuns


@partial(jit, static_argnums=(0, 1))
def loss_and_grad(net_u, operator, params, batch):
    # Fetch batch
    inputs, averages, beta = batch

    # Evaluate SPIN model
    outputs, sigma_avg = evaluate_spin(net_u, operator, params, inputs,
                                       averages, beta)

    # Compute loss
    _, _, _, rq, _ = outputs
    # eigenvalues are the diagonal entries of $\Lamda$
    eigenvalues = np.diag(rq)
    loss = np.sum(eigenvalues)

    # Compute masked gradients
    gradients, sigma_jac_avg = masked_gradients(net_u, params, inputs, outputs,
                                                averages, beta)

    # Store updated averages
    averages = (sigma_avg, sigma_jac_avg)

    return loss, gradients, averages


@partial(jit, static_argnums=(0, 1, 2))
def step(net_u, op, tx, params, opt_state, batch):
    loss, gradients, averages = loss_and_grad(net_u, op, params, batch)
    updates, opt_state = tx.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)
    return params, loss, opt_state, averages


def train(operator, dataset, MLP, hyper):
    neig = hyper["neig"]
    ndim = hyper["ndim"]
    num_hidden = hyper["num_hidden"]
    num_layers = hyper["num_layers"]
    num_iters = hyper["num_iters"]
    lr = hyper["lr"]
    layers = [ndim]
    for i in range(num_layers-1):
        layers.append(num_hidden)
    layers.append(neig)
    net_init, net_apply = MLP(layers)
    exp_lr = optax.exponential_decay(lr, transition_steps=1000, decay_rate=0.9)
    params = net_init(random.PRNGKey(0))
    tx = optax.adam(exp_lr)
    opt_state = tx.init(params)
    net_u = jit(partial(net_u_full, net_apply=net_apply))

    inputs = iter(dataset)
    pbar = trange(num_iters)

    beta = 1.0
    itercount = itertools.count()
    loss_log = []
    evals_log = []

    # Initialize moving averages
    sigma_avg = np.ones(neig)
    sigma_jac_avg = init_sigma_jac(net_u, params, next(inputs))
    averages = (sigma_avg, sigma_jac_avg)

    # Main training loop
    for it in pbar:
        # Set beta
        cnt = next(itercount)
        beta = beta if cnt > 0 else 1.0

        # Create batch
        batch = next(inputs), averages, beta

        # Run one gradient descent update
        params, loss, opt_state, averages = step(net_u, operator, tx, params,
                                                 opt_state, batch)

        # Logger
        evals, _ = eigenpairs(params, next(inputs), averages, beta, net_u,
                              operator)
        loss_log.append(loss)
        evals_log.append(evals)
        pbar.set_postfix({'Loss': loss})

    logs = (loss_log, evals_log)
    return (params, averages, beta, net_u, logs)
