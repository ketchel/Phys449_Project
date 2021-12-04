from jax import numpy as np
import numpy
from jax import tree_map, tree_multimap, random, jit, vmap, jacfwd
import itertools
from functools import partial
from tqdm import trange
import optax


@jit
def apply_mask(inputs, outputs):
    r"""
    Applies the boundary condition. Values outside the domain are zeroed out,
    while values inside the domain are scaled down. This specific mask assumes
    the domain is an ndim-dimensional box from 0 to pi.
    """
    mask = 0.1
    if len(inputs.shape) == 2:
        for i in range(inputs.shape[1]):
            mask *= np.maximum((-inputs[:, i] ** 2 + np.pi * inputs[:, i]), 0)
        mask = np.expand_dims(mask, -1)
    elif len(inputs.shape) == 1:
        for x in inputs:
            mask *= np.maximum((-x ** 2 + np.pi * x), 0)
    return mask*outputs


def net_u_full(params, inputs, net_apply):
    r"""
    Full mapping of the inputs over the neural network for a certain set of
    parameters.
    """
    outputs = net_apply(params, inputs)
    outputs = apply_mask(inputs, outputs)
    return outputs


def evaluate_spin(net_u, operator, params, inputs, averages, beta):
    r"""
    Evaluates the network for a given input, params, and moving averages.
    Returns the evaluated function, the inverse cholesky decomposition, the
    covariance and lambda matrices (see the paper for details), the operator
    (eg. laplacian) of the function evaluated for the input, and the average of
    sigma (see paper).
    """
    # Fetch batch
    n = inputs.shape[0]
    sigma_avg, _ = averages

    # Evaluate model
    u = net_u(params, inputs)
    sigma = np.dot(u.T, u)/n
    sigma_avg = (1.0 - beta) * sigma_avg + beta * sigma  # $\bar{\Sigma}$

    # Cholesky
    choli = np.linalg.inv(np.linalg.cholesky(sigma_avg))  # $L^{-1}$

    # Operator
    operator = operator(net_u, params, inputs)
    pi = np.dot(operator.T, u)/n  # $\Pi$
    rq = np.dot(choli, np.dot(pi, choli.T))  # $\Lambda$

    return (u, choli, pi, rq, operator), sigma_avg


def masked_gradients(net_u, params, inputs, outputs, averages, beta):
    r"""
    Implements eq. 14 from the paper.
    """
    # Fetch batch
    n = inputs.shape[0]
    u, choli, _, rq, operator = outputs
    _, sigma_jac_avg = averages

    diag = np.diag(np.diag(choli))
    triu = np.triu(np.dot(rq, diag))

    # \frac{\partial tr(\Lambda)}{\partial \Sigma}
    grad_sigma = -np.matmul(choli.T, triu)

    # \frac{\partail tr(\Lambda){\partial \Pi}}
    grad_pi = np.dot(choli.T, diag)

    # \frac{\partial u}{\partial \theta}
    grad_theta = vmap(jacfwd(net_u), in_axes=(None, 0))(params, inputs)

    # frac{\partail \Sigma}{\partial \theta}
    sigma_jac = tree_map(lambda x: np.tensordot(u.T, x, 1), grad_theta)

    sigma_jac_avg = tree_multimap(lambda x, y: (1. - beta) * x + beta * y,
                                  sigma_jac_avg, sigma_jac)

    # gradient  = \frac{\partial tr(\Lambda)}{\partial \theta}
    gradients = tree_multimap(
        lambda x, y: (
            np.tensordot(np.dot(grad_pi.T, operator.T), x, ([0, 1], [1, 0]))
            + np.tensordot(grad_sigma.T, y, ([0, 1], [1, 0]))
        )/n, grad_theta, sigma_jac_avg)

    # Negate for gradient ascent
    gradients = tree_map(lambda x: -1.0*x, gradients)

    return gradients, sigma_jac_avg


@partial(jit, static_argnums=(0))
def init_sigma_jac(net_u, params, inputs):
    r"""
    Initializes the moving average of the jacobian of Sigma.
    """
    u = net_u(params, inputs)
    grad_theta = jacfwd(net_u)(params, inputs)
    sigma_jac = tree_multimap(lambda x: np.tensordot(u.T, x, 1), grad_theta)
    return sigma_jac


def eigenpairs(params, inputs, averages, beta, net_u, operator):
    r"""
    Computes both the eigenvalues and eigenfunctions. The eigenvalues are the
    diagonal elements of $\Lambda$ and the eigenfunctions are recovered by
    multiplying the output of the neural network by the inverse transpose
    cholesky. For more details see section A of the supplementary materials of
    the paper.
    """
    outputs, _ = evaluate_spin(net_u, operator, params, inputs, averages, beta)
    u, choli, _, rq, _ = outputs
    evals = np.diag(rq)
    efuns = np.matmul(u, choli.T)
    return evals, efuns


@partial(jit, static_argnums=(0, 1))
def loss_and_grad(net_u, operator, params, batch):
    r"""
    Compute the loss function and the gradient for learning. Acts similarly to
    jax.value_and_grad() but instead uses masked gradient and returns the
    moving averages.
    """
    # Fetch batch
    inputs, averages, beta = batch

    # Evaluate SPIN model
    outputs, sigma_avg = evaluate_spin(net_u, operator, params, inputs,
                                       averages, beta)

    # Compute loss
    _, _, _, rq, _ = outputs
    loss = np.sum(np.diag(rq))  # Sum of the eigenvalues

    # Compute masked gradients
    gradients, sigma_jac_avg = masked_gradients(net_u, params, inputs, outputs,
                                                averages, beta)

    # Store updated averages
    averages = (sigma_avg, sigma_jac_avg)

    return loss, gradients, averages


@partial(jit, static_argnums=(0, 1, 2))
def step(net_u, op, tx, params, opt_state, batch):
    r"""
    Applies one step of the training algorithm.
    """
    loss, gradients, averages = loss_and_grad(net_u, op, params, batch)
    updates, opt_state = tx.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)
    return params, loss, opt_state, averages


def train(operator, dataset, MLP, hyper):
    r"""
    Main function of this module. Once the operator, dataset and network of
    parameters are set, we can train the parameters.
    """
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
    loss_log = numpy.zeros(num_iters)
    evals_log = numpy.zeros((num_iters, neig))

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
        loss_log[cnt] = loss
        evals_log[cnt] = evals
        pbar.set_postfix({'Loss': loss})

    logs = (loss_log, evals_log)
    return (params, averages, beta, net_u, logs)
