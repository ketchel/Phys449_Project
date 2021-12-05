from jax import numpy as np
import numpy
from jax import tree_map, tree_multimap, random, jit, vmap, jacfwd
import itertools
from functools import partial
from tqdm import trange
import optax


def evaluate_spin(fnet, op, params, x, avrgs, beta):
    r"""
    Evaluates the network for a given input, params, and moving averages.
    Returns the evaluated function, the inverse cholesky decomposition, the
    covariance and lambda matrices (see the paper for details), the operator
    (eg. laplacian) of the function evaluated for the input, and the average of
    sigma (see paper).
    """
    n = x.shape[0]
    u = fnet(params, x)
    Sigma_avg, _ = avrgs  # Get previous avg
    Sigma = np.dot(u.T, u)/n  # Calc new Sigma
    Sigma_avg = (1.0 - beta) * Sigma_avg + beta * Sigma  # Update avg
    L_inv = np.linalg.inv(np.linalg.cholesky(Sigma_avg))
    op_eval = op(fnet, params, x)
    Pi = np.dot(op_eval.T, u)/n
    Lambda = np.dot(L_inv, np.dot(Pi, L_inv.T))
    return (u, L_inv, Pi, Lambda, op_eval), Sigma_avg


def eigpairs(fnet, op, params, x, avrgs, beta):
    r"""
    Computes both the eigenvalues and eigenfunctions. The eigenvalues are the
    diagonal elements of $\Lambda$ and the eigenfunctions are recovered by
    multiplying the output of the neural network by the inverse transpose
    cholesky. For more details see section A of the supplementary materials of
    the paper.
    """
    outputs, _ = evaluate_spin(fnet, op, params, x, avrgs, beta)
    u, choli, _, rq, _ = outputs
    evals = np.diag(rq)
    efuns = np.matmul(u, choli.T)
    return evals, efuns


@partial(jit, static_argnums=(0, 1))
def loss_and_grad(fnet, op, params, batch):
    r"""
    Compute the loss function and the gradient for learning. Acts similarly to
    jax.value_and_grad() but instead uses masked gradient. It also returns the
    moving averages. Masked gradient is implemented here (Refer to eq. 14).
    """
    inputs, avrgs, beta = batch
    n = inputs.shape[0]
    outputs, Sigma_avg = evaluate_spin(fnet, op, params, inputs, avrgs, beta)
    _, _, _, Lambda, _ = outputs
    loss = np.sum(np.diag(Lambda))  # Sum of the eigenvalues

    # Fetch batch
    u, L_inv, _, Lambda, op_eval = outputs
    _, Sigma_jac_avg = avrgs

    L_diag = np.diag(np.diag(L_inv))

    # \frac{\partial tr(\Lambda)}{\partial \Sigma}
    Sigma_grad = -np.matmul(L_inv.T, np.triu(np.dot(Lambda, L_diag)))

    # \frac{\partail tr(\Lambda){\partial \Pi}}
    Pi_grad = np.dot(L_inv.T, L_diag)

    # \frac{\partial u}{\partial \theta}
    theta_grad = vmap(jacfwd(fnet), in_axes=(None, 0))(params, inputs)

    # frac{\partail \Sigma}{\partial \theta}
    sigma_jac = tree_map(lambda x: np.tensordot(u.T, x, 1), theta_grad)

    Sigma_jac_avg = tree_multimap(lambda x, y: (1. - beta) * x + beta * y,
                                  Sigma_jac_avg, sigma_jac)

    # gradient  = \frac{\partial tr(\Lambda)}{\partial \theta}
    gradients = tree_multimap(
        lambda x, y: (
            np.tensordot(np.dot(Pi_grad.T, op_eval.T), x, ([0, 1], [1, 0]))
            + np.tensordot(Sigma_grad.T, y, ([0, 1], [1, 0]))
        )/n, theta_grad, Sigma_jac_avg)

    # Negate for gradient ascent
    gradients = tree_map(lambda x: -1.0*x, gradients)

    # Store updated averages
    avrgs = (Sigma_avg, Sigma_jac_avg)

    return loss, gradients, avrgs


@partial(jit, static_argnums=(0, 1, 4))
def step(fnet, op, params, batch, tx, tx_state):
    r"""
    Applies one step of the training algorithm.
    """
    loss, gradients, averages = loss_and_grad(fnet, op, params, batch)
    updates, opt_state = tx.update(gradients, tx_state)
    params = optax.apply_updates(params, updates)
    return params, loss, opt_state, averages


def train(op, dataset, MLP, hyper):
    r"""
    Main function of this module. Once the operator, dataset and network of
    parameters are set, we can train the parameters.
    """
    # Unpack params
    neig = hyper["neig"]
    ndim = hyper["ndim"]
    num_hidden = hyper["num_hidden"]
    num_layers = hyper["num_layers"]
    num_iters = hyper["num_iters"]
    lr = hyper["lr"]
    verbosity = hyper["verbosity"]

    # Build layers list
    layers = [ndim]
    for i in range(num_layers-1):
        layers.append(num_hidden)
    layers.append(neig)

    # Initialize things
    net_init, net_apply = MLP(layers)
    exp_lr = optax.exponential_decay(lr, transition_steps=1000, decay_rate=0.9)
    params = net_init(random.PRNGKey(0))
    tx = optax.adam(exp_lr)
    tx_state = tx.init(params)

    @jit
    def fnet(params, x):
        logits = net_apply(params, x)
        mask = 0.1
        if len(x.shape) == 2:
            for i in range(x.shape[1]):
                mask *= np.maximum((-x[:, i] ** 2 + np.pi * x[:, i]), 0)
            mask = np.expand_dims(mask, -1)
        elif len(x.shape) == 1:
            for x in x:
                mask *= np.maximum((-x ** 2 + np.pi * x), 0)
        return mask*logits

    inputs = iter(dataset)
    beta = 1.0
    itercount = itertools.count()
    loss_log = numpy.zeros(num_iters)
    evals_log = numpy.zeros((num_iters, neig))

    if verbosity >= 0:
        print("Network Shape:")
        for i, (W, b) in enumerate(params):
            print('{}: W.shape = {} and b.shape = {}'.format(i,
                                                             W.shape, b.shape))
        print("")

    # Initialize moving averages
    sigma_avg = np.ones(neig)
    init_sample = next(inputs)
    u = fnet(params, init_sample)
    theta_grad = jacfwd(fnet)(params, init_sample)
    sigma_jac_avg = tree_map(lambda x: np.tensordot(u.T, x, 1), theta_grad)

    avrgs = (sigma_avg, sigma_jac_avg)

    # Main training loop
    pbar = trange(num_iters)
    for it in pbar:
        # Set beta
        counter = next(itercount)
        beta = beta if counter > 0 else 1.0

        # Create batch
        batch = next(inputs), avrgs, beta

        # Run one gradient descent update
        params, loss, tx_state, avrgs = step(fnet, op, params, batch,
                                             tx, tx_state)

        # Logger
        evals, _ = eigpairs(fnet, op, params, next(inputs), avrgs, beta)
        loss_log[counter] = loss
        evals_log[counter] = evals
        pbar.set_postfix({'Loss': loss})

    logs = (loss_log, evals_log)
    return (params, avrgs, beta, fnet, logs)
