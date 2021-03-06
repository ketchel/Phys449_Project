from jax import numpy as np
from jax.numpy import diag, tensordot, triu
from numpy import zeros
from jax import tree_map, tree_multimap, random, jit, vmap, jacfwd
from itertools import count
from functools import partial
from tqdm import trange
from optax import apply_updates, exponential_decay, adam
from src.util import fnet_box


def forward(fnet, op, params, x, Sigma_avg, beta):
    r"""
    Performs the feed forward over the neural network. Returns the evaluated
    function, the inverse cholesky decomposition, the covariance and lambda
    matrices (see the paper for details), the operator (eg. laplacian) of the
    function evaluated for the input, and the average of sigma (see paper).
    """
    batch_size = x.shape[0]
    fnet_eval = fnet(params, x)
    op_eval = op(fnet, params, x)
    Sigma = (fnet_eval.T @ fnet_eval)/batch_size  # Calc new Sigma
    Pi = (op_eval.T @ fnet_eval)/batch_size
    Sigma_avg = (1.0 - beta) * Sigma_avg + beta * Sigma  # Update avg
    L_inv = np.linalg.inv(np.linalg.cholesky(Sigma_avg))
    Lambda = L_inv @ Pi @ L_inv.T
    return fnet_eval, op_eval, L_inv, Pi, Lambda, Sigma_avg


def eigen(fnet, op, Sigma_avg, beta, params, x):
    r"""
    Computes both the eigenvalues and eigenfunctions. The eigenvalues are the
    diagonal elements of $\Lambda$ and the eigenfunctions are recovered by
    multiplying the output of the neural network by the inverse transpose
    cholesky. For more details see section A of the supplementary materials of
    the paper.
    """
    outputs = forward(fnet, op, params, x, Sigma_avg, beta)
    fnet_eval, _, L_inv, _, Lambda, _ = outputs
    return diag(Lambda), fnet_eval @ L_inv.T


@partial(jit, static_argnums=(0, 1))
def backward(fnet, op, params, batch):
    r"""
    Compute the loss function and the gradient for learning. Acts similarly to
    jax.value_and_grad() but instead uses masked gradient. It also returns the
    moving averages. Masked gradient is implemented here (Refer to eq. 14).
    """
    x, Sigma_avg, Sigma_jac_avg, beta = batch
    batch_size = x.shape[0]
    outputs = forward(fnet, op, params, x, Sigma_avg, beta)
    fnet_eval, op_eval, L_inv, _, Lambda, Sigma_avg = outputs
    loss = sum(diag(Lambda))  # Sum of the eigenvalues
    L_diag = diag(diag(L_inv))
    Sigma_grad = -L_inv.T @ triu(Lambda @ L_diag)
    Pi_grad = L_diag @ L_inv @ op_eval.T

    # The rest of these calculations act on the parameters, which is a list ADT
    # instead of a numpy array. As such we need the Jax pytrees functionality.
    backprop = vmap(jacfwd(fnet), in_axes=(None, 0))(params, x)
    Sigma_jac = tree_map(lambda x: tensordot(fnet_eval.T, x, 1), backprop)
    Sigma_jac_avg = tree_multimap(lambda x, y: (1. - beta) * x + beta * y,
                                  Sigma_jac_avg, Sigma_jac)
    grads = tree_multimap(
        lambda x, y: -1.0/batch_size*(
            tensordot(Pi_grad, x, ([0, 1], [1, 0]))
            + tensordot(Sigma_grad.T, y, ([0, 1], [1, 0]))),
        backprop, Sigma_jac_avg)

    return loss, grads, Sigma_avg, Sigma_jac_avg


@partial(jit, static_argnums=(0, 1, 4))
def update(fnet, op, params, batch, tx, tx_state):
    r"""
    Applies one step of the training algorithm.
    """
    loss, grads, Sigma_avg, Sigma_jac_avg = backward(fnet, op, params, batch)
    updates, opt_state = tx.update(grads, tx_state)
    params = apply_updates(params, updates)
    return params, loss, opt_state, Sigma_avg, Sigma_jac_avg


def run(op, dataset, MLP, hyper):
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
    box_max = hyper["box_max"]
    box_min = hyper["box_min"]

    # Build layers list
    layers = [ndim]
    for i in range(num_layers-1):
        layers.append(num_hidden)
    layers.append(neig)

    # Initialize things
    net_init, net_apply = MLP(layers)
    params = net_init(random.PRNGKey(0))
    exp_lr = exponential_decay(lr, transition_steps=1000, decay_rate=0.9)
    tx = adam(exp_lr)
    tx_state = tx.init(params)
    fnet = jit(partial(fnet_box, net_apply, box_min, box_max))

    iterator = iter(dataset)
    beta = 1.0
    itercount = count()
    losses = zeros(num_iters)
    evals = zeros((num_iters, neig))

    if verbosity >= 0:
        print("Network Shape:")
        for i, (W, b) in enumerate(params):
            print('{}: W.shape = {}, b.shape = {}'.format(i, W.shape, b.shape))
        print("")

    # Initialize moving averages
    x = next(iterator)
    fnet_eval = fnet(params, x)
    backprop = jacfwd(fnet)(params, x)
    Sigma_avg = np.ones(neig)
    Sigma_jac_avg = tree_map(lambda x: tensordot(fnet_eval.T, x, 1), backprop)

    pbar = trange(num_iters)
    for _ in pbar:
        counter = next(itercount)
        x = next(iterator)
        beta = beta if counter > 0 else 1.0
        batch = x, Sigma_avg, Sigma_jac_avg, beta
        results = update(fnet, op, params, batch, tx, tx_state)
        params, loss, tx_state, Sigma_avg, Sigma_jac_avg = results
        eval, _ = eigen(fnet, op, Sigma_avg, beta, params, x)
        losses[counter] = loss
        evals[counter] = eval
        pbar.set_postfix({'Loss': loss})

    fixed_eigen = partial(eigen, fnet, op, Sigma_avg, beta)
    return params, fixed_eigen, losses, evals
