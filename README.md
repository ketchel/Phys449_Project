# Spectral Inference Networks
This repo aims to implement spectral inference networks for the purpose of
learning function operators for use in solving PDEs. By using Jax's powerful
automatic differentiation system and XLA jit-compiling, we can implement
operators like the laplacian without resorting to matrix-representation
approximations like finite-differences.

## Structure and Design

The design philosophy was inspired by jax and as such a functional approach was
taken to maximize the usage of pure functions and minimize global variables.
Most of the code is in the `src` directory and their purposes are as follows:

| File        | Description                                   |
| ---------   | --------------------------------------------- |
| model       | Main logic. Implements SpINs                  |
| nn_gen      | Neural network implementation                 |
| data_gen    | Pytorch Dataloader implementation             |
| params      | Defines hyperparameters in a dict             |
| results     | Logic to save results and parameters          |
| util        | Defines operators and sets cpu/gpu            |

## Dependencies

- jax
- jaxlib
- optax
- matplotlib
- tqdm
- pytorch
- functools
- itertools

## Running

Default hyperparameters are included in `main.py`. To run the defaults, you can
simply run

```bash
python main.py
```

However if you want to change __any__ of the hyperparameters you must supply the
program with a full list of hyperparameters as a json file. To run the program
with user-defined hyperparameters, run

```bash
python main.py --param "path/to/params.json"
```

## Hyperparameters

__Be warned that if you supply your own json file you must supply every
hyperparameter!__. Here is a table of all hyperparameters:

|Parameter    | Description                                   |
| ---------   | --------------------------------------------- |
| lr          | learning rate                                 |
| box_min     | minimum of the domain (in all dimensions)     |
| box_max     | maximum of the domain (in all dimensions)     |
| ndim        | dimension number (only supports 1 or 2 now)   |
| neig        | number of eigenfunctions to find              |
| num_iters   | max iterations                                |
| num_layers  | number of hidden layers                       |
| num_hidden  | number of hidden nodes per hidden layer       |
| batch_size  | batch size                                    |
| results     | directory to save results to                  |

## Acknowledgements

You can find the origional tensorflow code
[here](https://github.com/deepmind/spectral_inference_networks) and the paper
[on arXiv](https://arxiv.org/abs/1806.02215v3)
