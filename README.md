# Spectral Inference Networks Project
This is a github repo for our project on spectral inference netwroks.

Run using
```bash
python main.py
```

# Hyperparameters

Default hyperparameters are included in `main.py`. For user-defined parameters,
you can specify them in a separate json file like so:

```bash
python main.py --param "path/to/params.json"
```

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
