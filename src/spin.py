from jax import numpy as np
import numpy
from jax import tree_map, tree_multimap, random, jit, vmap, jacfwd
import itertools
from functools import partial
from tqdm import trange
import optax


def forward(fnet, op, params, x, avrgs, beta):
    r"""
    Performs the feed forward over the neural network. Returns the evaluated
    function, the inverse cholesky decomposition, the covariance and lambda
    matrices (see the paper for details), the operator (eg. laplacian) of the
    function evaluated for the input, and the average of sigma (see paper).
    """
    batch_size = x.shape[0]
    fnet_eval = fnet(params, x)
    op_eval = op(fnet, params, x)
    Sigma_avg, _ = avrgs  # Get previous avg
    Sigma = np.dot(fnet_eval.T, fnet_eval)/batch_size  # Calc new Sigma
    Sigma_avg = (1.0 - beta) * Sigma_avg + beta * Sigma  # Update avg
    L_inv = np.linalg.inv(np.linalg.cholesky(Sigma_avg))
    Pi = np.dot(op_eval.T, fnet_eval)/batch_size
    Lambda = np.dot(L_inv, np.dot(Pi, L_inv.T))
    return fnet_eval, op_eval, L_inv, Pi, Lambda, Sigma_avg


def eigen(fnet, op, params, x, avrgs, beta):
    r"""
    Computes both the eigenvalues and eigenfunctions. The eigenvalues are the
    diagonal elements of $\Lambda$ and the eigenfunctions are recovered by
    multiplying the output of the neural network by the inverse transpose
    cholesky. For more details see section A of the supplementary materials of
    the paper.
    """
    outputs = forward(fnet, op, params, x, avrgs, beta)
    fnet_eval, _, L_inv, _, Lambda, _ = outputs
    return np.diag(Lambda), np.matmul(fnet_eval, L_inv.T)


@partial(jit, static_argnums=(0, 1))
def backward(fnet, op, params, batch):
    r"""
    Compute the loss function and the gradient for learning. Acts similarly to
    jax.value_and_grad() but instead uses masked gradient. It also returns the
    moving averages. Masked gradient is implemented here (Refer to eq. 14).
    """
    x, avrgs, beta = batch
    _, Sigma_jac_avg = avrgs
    batch_size = x.shape[0]
    outputs = forward(fnet, op, params, x, avrgs, beta)
    fnet_eval, op_eval, L_inv, _, Lambda, Sigma_avg = outputs
    loss = np.sum(np.diag(Lambda))  # Sum of the eigenvalues
    backprop = vmap(jacfwd(fnet), in_axes=(None, 0))(params, x)

    L_diag = np.diag(np.diag(L_inv))

    # \frac{\partial tr(\Lambda)}{\partial \Sigma}
    Sigma_grad = -np.matmul(L_inv.T, np.triu(np.dot(Lambda, L_diag)))

    # \frac{\partail tr(\Lambda){\partial \Pi}}
    Pi_grad = np.dot(L_inv.T, L_diag)

    # frac{\partail \Sigma}{\partial \theta}
    Sigma_jac = tree_map(lambda x: np.tensordot(fnet_eval.T, x, 1), backprop)
    Sigma_jac_avg = tree_multimap(lambda x, y: (1. - beta) * x + beta * y,
                                  Sigma_jac_avg, Sigma_jac)

    # \frac{\partial tr(\Lambda)}{\partial \theta}
    grads = tree_multimap(
        lambda x, y: -1.0*(
            np.tensordot(np.dot(Pi_grad.T, op_eval.T), x, ([0, 1], [1, 0]))
            + np.tensordot(Sigma_grad.T, y, ([0, 1], [1, 0])))/batch_size,
        backprop, Sigma_jac_avg)

    return loss, grads, Sigma_avg, Sigma_jac_avg


@partial(jit, static_argnums=(0, 1, 4))
def update(fnet, op, params, batch, tx, tx_state):
    r"""
    Applies one step of the training algorithm.
    """
    loss, grads, Sigma_avg, Sigma_jac_avg = backward(fnet, op, params, batch)
    updates, opt_state = tx.update(grads, tx_state)
    params = optax.apply_updates(params, updates)
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
    exp_lr = optax.exponential_decay(lr, transition_steps=1000, decay_rate=0.9)
    params = net_init(random.PRNGKey(0))
    tx = optax.adam(exp_lr)
    tx_state = tx.init(params)

    @jit
    def fnet(params, x):
        r"""
        Defines the function over the network. This is needed to enforce
        boundary conditions for the PDE, but also to apply a mask over the
        function to mitigate blow-up for certain domains.
        """
        logits = net_apply(params, x)
        mask = 1.0
        if len(x.shape) == 2:
            for i in range(x.shape[1]):
                slice = x[:, i]
                mask *= -np.minimum((slice - box_min)*(slice - box_max), 0)
                # mask *= np.maximum(-slice**2 + box_max*slice , 0)
            mask = np.expand_dims(mask, -1)
        elif len(x.shape) == 1:
            for slice in x:
                mask *= -np.minimum((slice - box_min)*(slice - box_max), 0)
                # mask *= np.maximum(-slice**2 + box_max*slice , 0)
        return mask*logits

    iterator = iter(dataset)
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

    sigma_avg = np.ones(neig)
    init_sample = next(iterator)
    u = fnet(params, init_sample)
    theta_grad = jacfwd(fnet)(params, init_sample)
    sigma_jac_avg = tree_map(lambda x: np.tensordot(u.T, x, 1), theta_grad)
    avrgs = (sigma_avg, sigma_jac_avg)

    # Main training loop
    pbar = trange(num_iters)
    for _ in pbar:
        counter = next(itercount)
        x = next(iterator)
        beta = beta if counter > 0 else 1.0
        batch = x, avrgs, beta
        results = update(fnet, op, params, batch, tx, tx_state)
        params, loss, tx_state, avrgs = results
        evals, _ = eigen(fnet, op, params, x, avrgs, beta)
        loss_log[counter] = loss
        evals_log[counter] = evals
        pbar.set_postfix({'Loss': loss})

    logs = (loss_log, evals_log)
    return params, avrgs, beta, fnet, logs
