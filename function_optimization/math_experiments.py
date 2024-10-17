import argparse
import pathlib

import pandas
from tqdm import tqdm

import utils
import function_optimization
import lrs

ROOT_DIR = pathlib.Path(__file__).parent.parent
OUTPUT_PATH = ROOT_DIR / 'results' / 'math'


class SummarizeBest:
    def __init__(self, ignore_keys=None):
        self.__df = pandas.DataFrame(columns=['function', 'n', 'optimizer', 'seed', 'best', 'last'])
        self.__new_runs = []
        self.ignore_keys = [] if ignore_keys is None else ignore_keys

    def add_run(self, function, n, optimizer, seed, step_size, history, **additional_kwargs):
        self.__new_runs.append({**additional_kwargs, 'function': function, 'n': n,
                                'optimizer': function_optimization.gradient_label(optimizer, self.ignore_keys),
                                'lr': step_size, 'seed': seed, 'best': history.y.min(),
                                'last': history.y.iloc[-1]})

    def update_dataframe(self):
        self.__df = pandas.concat([self.__df, pandas.DataFrame(self.__new_runs)])
        self.__new_runs = []

    def get_table(self, function=None, n=None, optimizer=None, seed=None, best=False):
        self.update_dataframe()
        query = ' and '.join(f'{key}=={repr(value)}' for key, value in {
            'function': function, 'n': n, 'optimizer': optimizer, 'seed': seed}.items() if value is not None)
        data = self.__df.query(query) if query else self.__df
        value_column = 'best' if best else 'last'
        index = [col for col in ['function', 'optimizer', 'lr'] if col not in self.ignore_keys]
        return data.pivot_table(values=value_column, index=index, columns=['n'], sort=False)

    def get_df(self):
        self.update_dataframe()
        return self.__df

    def to_csv(self, path):
        self.__df.to_csv(path, index=False)


def run_math_experiments(function_name, dims, gradient_approaches, lr_csv, seeds=None, num_seeds=1, max_steps=None,
                         early_stopping_patience=50, start=None, summary_ignore_keys=None, save_to_path=None):
    seeds = list(range(num_seeds)) if seeds is None else seeds
    results = []
    summary = SummarizeBest(summary_ignore_keys)
    function, starting_position_fn, default_max_steps = function_optimization.get_function_config(function_name, start)
    max_steps = max_steps or default_max_steps

    with tqdm(total=len(dims) * len(gradient_approaches) * len(seeds)) as progressbar:
        for n in dims:
            starting_position = starting_position_fn(n)
            for gradient_computation in gradient_approaches:
                step_size = lrs.get_math_lr(lr_csv, function_name, n, gradient_computation)
                for seed in seeds:
                    gradient_computation.set_seed(seed)
                    history = function_optimization.optimize(function, starting_position, max_steps, step_size,
                                                             gradient_computation, early_stopping_patience)

                    history['n'] = n
                    history['_optimizer_label_short'] = function_optimization.gradient_label(
                        gradient_computation, summary_ignore_keys)
                    history['_optimizer_label_long'] = function_optimization.gradient_label(gradient_computation)
                    results.append(history)
                    summary.add_run(function_name, n, gradient_computation, seed, step_size, history)
                    progressbar.update(1)

    results = pandas.concat(results).reset_index(drop=True)
    results['_function'] = function_name

    with pandas.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000,
                               'display.float_format', '{:.1e}'.format, 'display.max_colwidth', 80):
        print(summary.get_table())

    if save_to_path:
        print(f'Writing results to {save_to_path}')
        results.to_csv(utils.append_suffix(save_to_path, '__all.csv'), index=False)
        summary.to_csv(utils.append_suffix(save_to_path, '__summary.csv'))

    return results, summary


def experiments_for_function(function_name, ns=None, ks=None, num_seeds=5, best_lrs=None, max_steps=None):
    best_lrs = best_lrs or lrs.BEST_LRS_CSV['math']
    ks = ks or [2, 4, 16]
    ns = ns or [2**i for i in range(11)]  # 1, 2, 4,..., 1024

    gradient_approaches = [
        function_optimization.get_algorithm('bp')(),
        function_optimization.get_algorithm(gradient_type='fg', k=1, aggregation='mean')(),
    ] + [function_optimization.get_algorithm(gradient_type='fg', k=k, aggregation=agg)()
         for k in ks for agg in ['mean', 'orthogonal_projection']]

    output_path = utils.construct_output_path(
        output_path=OUTPUT_PATH, output_name=function_name, experiment_id='math_experiments')
    run_math_experiments(function_name, ns, gradient_approaches, lr_csv=best_lrs, num_seeds=num_seeds,
                         save_to_path=output_path, summary_ignore_keys=['lr'], max_steps=max_steps)


def experiments_custom_tangents(function_name, ns=None, ks=None, num_seeds=5, best_lrs=None, max_steps=None):
    best_lrs = best_lrs or lrs.BEST_LRS_CSV['math_custom_tangents']
    ks = ks or [16, 64]
    ns = ns or [64]

    gradient_approaches = [function_optimization.get_algorithm(gradient_type='fg', k=k, aggregation=agg,
                                                               tangent_sampler=tangent_sampler)()
                           for k in ks for agg in ['mean', 'orthogonal_projection']
                           for tangent_sampler in [f'specific_angle_{angle}' for angle in [15, 30, 45, 60, 75, 90]]]

    output_path = utils.construct_output_path(
            output_path=OUTPUT_PATH, output_name=function_name, experiment_id='custom_tangents')
    run_math_experiments(function_name, ns, gradient_approaches, lr_csv=best_lrs, num_seeds=num_seeds,
                         save_to_path=output_path, summary_ignore_keys=['lr'], max_steps=max_steps)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda _: print('No subparser given'))
    parser.add_argument('--best_lrs', type=pathlib.Path)
    parser.add_argument('--function', default='styblinski-tang', type=str,
                        choices=['rosenbrock', 'styblinski-tang', 'sphere'])
    parser.add_argument('-k', type=int, nargs='+')
    parser.add_argument('-n', type=int, nargs='+')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--max_steps', type=int)
    subparsers = parser.add_subparsers()

    paper_experiments_parser = subparsers.add_parser('math_experiments')
    paper_experiments_parser.set_defaults(func=lambda config: experiments_for_function(
        config.function, config.n, config.k, config.num_seeds, config.best_lrs, config.max_steps))

    paper_experiments_parser = subparsers.add_parser('custom_tangents')
    paper_experiments_parser.set_defaults(func=lambda config: experiments_custom_tangents(
        config.function, config.n, config.k, config.num_seeds, config.best_lrs, config.max_steps))

    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_args()
    cli_args.func(cli_args)
