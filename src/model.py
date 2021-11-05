import torch
import torch.nn as nn
import numpy as np

#write code for pytorch model here
def make_network(x, hid, ws, bs, apply_boundary, lim, custom_softplus=False):
  """Constructs network and loss function.

  Args:
    x: Input to the network.
    hid: List of shapes of the hidden layers of the networks.
    ws: List of weights of the network.
    bs: List of biases of the network.
    apply_boundary: If true, force network output to be zero at boundary
    lim: The limit of the network, if apply_boundary is true.

  Returns:
    Output of multi-layer perception network.
  """
  # Somthing along the lines Wx + b



  return None



# This is analagous to _add_max in SpIN
# not sure what the math does but I think it does the same thing
def add_bounds(x, y, lim):
    """
    Adds the boundary conditions
    Forces the wavefunction to be zero outside the range [-lim, lim]
    """
    mask = 1

    for i in range(list(x.shape)[1]):
        mask *= torch.maximum((torch.sqrt(2 * lim**2 - x[:, i]**2) - lim) / lim, torch.tensor(0))

    return torch.unsqueeze(mask,  -1)*y


class SpectralNetwork(object):
  """Class that constructs operators for SpIN and includes training loop."""

  def __init__(self, operator, network, data, params,
               decay=0.0, use_pfor=True, per_example=False):
  # Basing this on the tensorflow code
    """Creates all ops and variables required to train SpIN.

    Args:
      operator: The linear operator to diagonalize.
      network: A function that returns the network applied on the output.
      data: A TensorFlow op for the input to the spectral inference network.
      params: The trainable parameters of the model built by 'network'.
      decay (optional): The decay parameter for the moving average of the
        network covariance and Jacobian.
      use_pfor (optional): If true, use the parallel_for package to compute
        Jacobians. This is often faster but has higher memory overhead.
      per_example (optional): If true, computes the Jacobian of the network
        output covariance using a more complicated but often faster method.
        This interacts badly with anything that uses custom_gradients, so needs
        to be avoided for some code branches.
    """
  def _moving_average(self, x, c):
    """Creates moving average operation.

    Args:
      x: The tensor or list of tensors of which to take a moving average.
      c: The decay constant of the moving average, between 0 and 1.
        0.0 = the moving average is constant
        1.0 = the moving averge has no memory

    Returns:
      ma: Moving average variables.
      ma_update: Op to update moving average.
    """
    def _training_update(self,
                       sigma,
                       pi,
                       params,
                       decay=0.0,
                       use_pfor=False,
                       features=None,
                       jac=None):
    """Makes gradient and moving averages.

    Args:
      sigma: The covariance of the outputs of the network.
      pi: The matrix of network output covariances multiplied by the linear
        operator to diagonalize. See paper for explicit definition.
      params: The trainable parameters.
      decay (optional): The decay parameter for the moving average of the
        network covariance and Jacobian.
      use_pfor (optional): If true, use the parallel_for package to compute
        Jacobians. This is often faster but has higher memory overhead.
      features (optional): The output features of the spectral inference
        network. Only necessary if per_example=True.
      jac (optional): The Jacobian of the network. Only necessary if
        per_example=True.

    Returns:
      loss: The loss function for SpIN - the sum of eigenvalues.
      gradients: The approximate gradient of the loss using moving averages.
      eigvals: The full array of eigenvalues, rather than just their sum.
      chol: The Cholesky decomposition of the covariance of the network outputs,
        which is needed to demix the network outputs.
    """
    return None