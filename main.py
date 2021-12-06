from src.nn_gen import MLP
from src.params import get_params
from src.util import get_operator
from src.data_gen import DataGenerator
from src.spin import run
from src.results import save_results


def main():
    hyper = get_params()
    op = get_operator(hyper)
    dataset = DataGenerator(hyper)
    results = run(op, dataset, MLP, hyper)
    save_results(op, results, hyper)


if __name__ == '__main__':
    main()
