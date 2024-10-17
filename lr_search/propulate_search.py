import argparse
import collections
import datetime
import logging
import math
import pathlib
import random
import shutil
import statistics
import sys
import uuid
from typing import Tuple

import numpy as np
import pandas
import propulate
import propulate.propagators
import torch
from mpi4py import MPI

import function_optimization
import nn_training

PROPULATE_CKPT_PATH = pathlib.Path(__file__).parent / 'propulate_checkpoints'


class FinalLossAwareStaticSurrogate(propulate.surrogate.StaticSurrogate):
    def __init__(self, patience=0, max_steps=None, verbose=False, margin_offset=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience: int = patience  # optional patience, if > 0, allow patience worse steps before pruning
        self.steps_without_improvement: int = 0  # cancels when self.steps_without_improvement > self.patience
        self.logger = logging.getLogger('propulate.FinalLossAwareStaticSurrogate')
        self.verbose = verbose
        self.margin_offset = margin_offset  # uses as temporary fix for negative loss values by shifting the zero line
        self.max_steps = max_steps or 0
        self.current_run = np.full(self.max_steps, np.nan)
        self.logger.debug(f'Using {type(self).__name__} with margin {self.margin} and patience {self.patience}.')

    def start_run(self, ind: propulate.population.Individual) -> None:
        super().start_run(ind)
        # initialize current_run with nan instead of zero, otherwise the zeros might be interpreted as actual loss
        # values reached by the run if it is shorter than the baseline (e.g. stopped because of internal early stopping
        # or reaching nan values).
        self.current_run = np.full(max(self.max_steps, self.baseline.size), np.nan)
        # Additionally reset the patience counter
        self.steps_without_improvement = 0

    def stop_criterion(self, step, loss) -> Tuple[bool, str]:
        # Cancel is only allowed after the first complete run.
        reason = ''
        if self.first_run:
            return False, reason

        # Consider cancel if current run is outside margin of baseline AND final baseline is better than current loss
        # This additional second condition ensures runs aren't stopped too early due to a suboptimal baseline (e.g. a
        # baseline with a too large learning rate that improves quickly at the beginning of training but diverges later)
        #
        # note that baseline may contain fewer stats than the current run, use the last baseline value in that case
        baseline_at_step = self.baseline[step] if step < len(self.baseline) else self.baseline[-1]
        # use offset for negative losses (e.g. styblinski-tang)
        # as a simple multiplicative margin always points away from zero
        margin = (baseline_at_step - self.margin_offset) / self.margin + self.margin_offset
        is_outside_margin = margin < loss
        final_baseline_is_better = self.baseline[-1] < loss

        if is_outside_margin and final_baseline_is_better:
            self.steps_without_improvement += 1
            reason = f'{loss=:.2e}, baseline={baseline_at_step:.2e} -> margin {margin:.2e} ({self.margin}) ' \
                     f'patience at {self.steps_without_improvement} / {self.patience}'

            # only actually prune the run if the patience has run out
            if self.steps_without_improvement > self.patience:
                self.steps_without_improvement = 0
                return True, reason
        else:
            # if the run is within the margin, reset the patience
            self.steps_without_improvement = 0

        return False, reason

    def append_loss_to_current_run(self, loss):
        # Append loss to current run.
        # insert current loss at current index, either by writing directly to that index (if the array is long enough)
        if self.synthetic_id < len(self.current_run):
            self.current_run[self.synthetic_id] = loss
        else:  # or by appending to the array. in which case it is expected to contain all previous values
            assert len(self.current_run) == self.synthetic_id
            self.current_run = np.append(self.current_run, loss)
        self.synthetic_id += 1  # finally increment the current step counter

    def cancel(self, loss: float) -> bool:
        """
        Cancel the current run if the loss is outside the margin of the baseline AND worse than the final margin loss.

        Parameters
        ----------
        loss : float
            The next interim loss of the current run.

        Returns
        -------
        bool
            True if the current run is cancelled, False otherwise.
        """
        # Append loss to current run.
        self.append_loss_to_current_run(loss)

        # Check if the run should be pruned and log potential reasons
        stop, reason = self.stop_criterion(self.synthetic_id - 1, loss)
        if reason and (self.verbose or stop):
            self.logger.debug(reason + (f' -> Pruning run in step {self.synthetic_id - 1}' if stop else ''))
        return stop

    def __replace_baseline(self, other_run: np.ndarray) -> None:
        """
        Replace the baseline with the given run if the final loss is better, or if there is no prior run.
        Additionally, it prunes all NaN values off the end of the array

        Parameters
        ----------
        other_run : np.ndarray
            The loss series of the incoming run.
        """
        # skip all nan runs
        if np.isnan(other_run).all():
            return

        index_of_last_actual_value = np.argwhere(~np.isnan(other_run)).max()
        other_run = other_run[:index_of_last_actual_value + 1]  # cut off trailing nans

        # Always take first available baseline
        if self.first_run:
            self.baseline = other_run.copy()
            self.first_run = False
            return

        # After that, only update baseline if the new final loss is better (i.e. SMALLER) than baseline.
        if self.baseline[-1] > other_run[-1]:
            self.logger.debug(
                f'Replacing surrogate: new final loss {other_run[-1]:.2g} < old final loss {self.baseline[-1]:.2g}.')
            self.baseline = other_run.copy()

    def update(self, loss: float) -> None:
        """
        Replace the baseline with the current run if the final loss is better, or if there is no prior run.

        Parameters
        ----------
        loss : float
            The (unused) final loss of the current run.
        """
        self.logger.debug('Starting update.')
        self.__replace_baseline(self.current_run)

    def merge(self, data: np.ndarray) -> None:
        """
        Replace the baseline with the incoming run if the final loss is better, or if there is no prior run.

        Parameters
        ----------
        data : np.ndarray
            The loss series of the incoming run.
        """
        self.logger.debug('Starting merge.')
        self.__replace_baseline(data)


def get_lr_from_param_dict(param_dict):
    if 'lr' in param_dict:
        return param_dict['lr']
    if 'lr_log10' in param_dict:
        return 10 ** param_dict['lr_log10']
    raise ValueError(f'No lr found in {param_dict}.')


def run_propulate(loss_fn, limits, seed=0, comm=MPI.COMM_WORLD, population_size=None, generations=10,
                  surrogate_margin=0.5, top_n=2, checkpoint_path='propulate_checkpoints', logging_interval=1,
                  log_level=logging.DEBUG, stdout_log_level=logging.INFO, log_file=None, propagator_kwargs=None,
                  surrogate_kwargs=None):
    propulate.utils.set_logger_config(level=log_level, log_file=log_file, log_to_stdout=True, log_rank=True,
                                      colors=True)
    logging.getLogger('propulate').handlers[0].setLevel(stdout_log_level)

    population_size = population_size or comm.size * 2
    rng = random.Random(seed + comm.rank)  # Separate random number generator for optimization.
    propagator_kwargs = propagator_kwargs or {}
    propagator = propulate.utils.get_default_propagator(pop_size=population_size, limits=limits, rng=rng,
                                                        **propagator_kwargs)
    surrogate_kwargs = surrogate_kwargs or {}
    surrogate_factory = lambda: FinalLossAwareStaticSurrogate(
        margin=surrogate_margin, **surrogate_kwargs) if surrogate_margin else None
    propulator = propulate.Propulator(loss_fn=loss_fn, propagator=propagator, propulate_comm=comm,
                                      generations=generations, checkpoint_path=checkpoint_path, rng=rng,
                                      surrogate_factory=surrogate_factory)

    # Run optimization and print summary of results.
    propulator.propulate(logging_interval=logging_interval)
    top_n_individuals = propulator.summarize(top_n=top_n)

    if comm.rank == 0:
        result_logger = logging.getLogger('propulate.results')
        print_overview_of_best_results(propulator.population, print_fn=result_logger.info)
        if log_file:
            with open(log_file, 'r') as f:
                num_pruned = len([line for line in f.readlines() if 'PRUNING' in line])
            result_logger.info(f'{num_pruned} / {len(propulator.population)} pruned')

    return top_n_individuals


def print_overview_of_best_results(population, eps=0.1, print_fn=print):
    sorted_individual = sorted(population, key=lambda x: x.loss)
    min_loss = sorted_individual[0].loss
    print_fn(f'Top LRs after {len(population)} evaluations, best loss found {min_loss}:')
    for individual in sorted_individual:
        print_fn(f'LR {get_lr_from_param_dict(individual.mapping):8.2g}\t loss {individual.loss:8.2g}')
        if individual.loss > min_loss * (1. + eps):
            break


def print_population_overview(population):
    lr_to_loss = {}
    lrs = []
    for individual in population:
        lr = get_lr_from_param_dict(individual.mapping)
        lrs.append(lr)
        if lr not in lr_to_loss:
            lr_to_loss[lr] = []
        lr_to_loss[lr].append(individual.loss)

    print(f'LRs tested:')
    for lr, count in collections.Counter(lrs).most_common():
        print(f'{lr:8.2g} {f"({count})":>6} loss: {", ".join(f"{loss:8.2g}" for loss in set(lr_to_loss[lr]))}')


def lr_search_with_propulate(loss_fn, limits, search_label, population_size=8, num_evaluations=32, surrogate_margin=0.5,
                             comm=MPI.COMM_WORLD, delete_old_checkpoints=False, surrogate_kwargs=None):
    checkpoint_path = PROPULATE_CKPT_PATH / search_label / 'checkpoints'
    if comm.rank == 0 and delete_old_checkpoints and checkpoint_path.exists():
        print(f'Deleting old checkpoint directory {checkpoint_path} to overwrite with new checkpoints.')
        shutil.rmtree(checkpoint_path)
    if comm.rank == 0:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    log_file = checkpoint_path.parent / f'propulate_log__{datetime.datetime.today().strftime("%Y-%m-%d--%H-%M-%S")}.log'
    log_file = comm.bcast(log_file)
    if comm.rank == 0:
        print(f'Logging to {log_file.absolute()}')

    if limits[0] > limits[1]:
        raise ValueError(f'Limits must be passed as (min, max) but for {tuple(limits)}, {limits[0]} > {limits[1]}')
    limits = {'lr_log10': tuple(limits)}  # search log space, convert internally to actual LR

    generations = math.ceil(num_evaluations / comm.size)
    run_propulate(loss_fn, limits, population_size=population_size, generations=generations,
                  surrogate_margin=surrogate_margin, checkpoint_path=checkpoint_path, log_file=log_file, comm=comm,
                  surrogate_kwargs=surrogate_kwargs)


def get_function_starting_position_and_max_steps(function_name, n):
    function, starting_position_fn, max_steps = function_optimization.get_function_config(function_name)
    starting_position = starting_position_fn(n)
    return function, starting_position, max_steps


def optimize_math(function, starting_position, max_steps, gradient_computation, step_size, early_stopping_patience,
                  yield_every=1):
    optimizer = function_optimization.GradientDescent(step_size, gradient_computation, starting_position, function,
                                                      early_stopping_patience)
    yield_every = yield_every or (1 if max_steps < 250 else max_steps // 100)

    for i in range(max_steps):
        optimizer.step()
        if i % yield_every == 0:
            yield np.inf if np.isnan(optimizer.current_value) else optimizer.current_value

        reason = optimizer.check_stop_criteria()
        if reason:
            print(reason)
            break
    return np.inf if np.isnan(optimizer.current_value) else optimizer.current_value


def get_surrogate_kwargs(cli_args, max_steps):
    surrogate_kwargs = {}
    if cli_args.surrogate_patience is not None:
        patience = cli_args.surrogate_patience
        if patience < 1:
            patience *= max_steps
        surrogate_kwargs['patience'] = int(patience)
    return surrogate_kwargs


def approach_label(gradient_approach, k=None):
    return f'{gradient_approach}' + ('' if gradient_approach == 'bp' else f'_k={k}')


def lr_search_for_math(cli_args):
    # setup task
    function, starting_position, max_steps = get_function_starting_position_and_max_steps(cli_args.function, cli_args.n)
    early_stopping_patience = 100
    task_label = f'{cli_args.function}_n={cli_args.n:04d}'

    # setup gradient approach
    gradient_type = 'bp' if cli_args.gradient == 'bp' else 'fg'
    aggregation = {'fg-mean': 'mean', 'frog': 'orthogonal_projection'}.get(cli_args.gradient, None)
    gradient_approach = function_optimization.get_algorithm(gradient_type, aggregation=aggregation, k=cli_args.k,
                                                            tangent_sampler=cli_args.tangent_sampler)

    # define loss fn
    def loss_fn(param_dict):
        yield from optimize_math(function, starting_position, max_steps, gradient_approach(),
                                 get_lr_from_param_dict(param_dict), early_stopping_patience)

    search_label = f'math/{task_label}/{approach_label(cli_args.gradient, cli_args.k)}'
    if cli_args.tangent_sampler != 'normal':
        search_label += '__' + cli_args.tangent_sampler
    surrogate_kwargs = get_surrogate_kwargs(cli_args, max_steps)
    if cli_args.function == 'styblinski-tang':
        global_minimum = function_optimization.get_global_minimum(cli_args.function, cli_args.n)
        surrogate_kwargs['margin_offset'] = global_minimum
    lr_search_with_propulate(loss_fn, cli_args.limits, search_label, cli_args.population_size, cli_args.num_evaluations,
                             cli_args.surrogate_margin, delete_old_checkpoints=cli_args.delete_old_checkpoints,
                             surrogate_kwargs=surrogate_kwargs)


class NNArgs:
    def __init__(self, gradient, dataset, model, model_width=None, num_directions=1, seed=0, epochs=200, device=None):
        self.configfile = nn_training.parse_config.DEFAULT_CONFIG_FILES

        self.gradient_computation = gradient.split('-')[0]  # from 'bp', 'fg-mean', 'frog' to 'bp', 'fg', 'frog'
        self.perturbation_mode = 'node'
        self.fg_computation_mode = 'sim'
        self.num_directions = num_directions

        self.device = device

        self.optimizer = 'plain_sgd'
        self.epochs = epochs
        self.seed = seed
        self.initial_lr = None  # this should be set manually for each loss_fn

        self.dataset_root = pathlib.Path(__file__).parent.parent / 'data'  # repo_root/data
        self.dataset = dataset

        self.model = model
        self.model_hidden_size = model_width

    def get_task_label(self):
        model = self.model
        if model == 'fc':
            model += f'_w={self.model_hidden_size}'
        return f'{model}__{self.dataset}'

    def __getattr__(self, item):
        # returns None for all other attributes that have not been defined
        return None


def get_device(device_type, comm=MPI.COMM_WORLD):
    if device_type == 'cuda':
        assert torch.cuda.is_available()
        assigned_gpu = comm.rank % torch.cuda.device_count()
        return f'cuda:{assigned_gpu}'
    return device_type


def lr_search_for_nn(cli_args):
    # ---------- Prepare configs ----------
    nn_args = NNArgs(cli_args.gradient, cli_args.dataset, cli_args.model, cli_args.model_width, cli_args.k,
                     epochs=cli_args.epochs, device=cli_args.device)

    search_label = f'nns/{nn_args.get_task_label()}/{approach_label(cli_args.gradient, cli_args.k)}'
    surrogate_kwargs = get_surrogate_kwargs(cli_args, nn_args.epochs)

    comm = MPI.COMM_WORLD
    results_dir = PROPULATE_CKPT_PATH / search_label / 'run_results'
    if comm.rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)

    loss_fn_logger = logging.getLogger('propulate.loss_fn')

    def loss_fn(param_dict):
        # ---------- Prepare nn config ----------
        lr = get_lr_from_param_dict(param_dict)
        nn_args.initial_lr = lr
        config = nn_training.parse_config.Configuration(nn_args)

        # ---------- Prepare nn training ----------
        torch.manual_seed(nn_args.seed)
        device = torch.device(get_device(config.device, comm=comm))
        device_specific_kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}

        # initialize dataloaders (need only train and val loader)
        train_loader, _, val_loader, _ = nn_training.datasets.get_dataloaders(config, device_specific_kwargs)

        # initialize model and optimizer
        max_epochs = config.get('epochs', datatype=int)
        model = nn_training.models.get_model(config).to(device)
        optimizer, lr_scheduler = nn_training.train.create_optimizer_and_lr_schedule_from_config(
            model, config, max_epochs)

        # setup forward gradient configuration
        num_directions = config.get_config('gradient_computation').getint('num_directions', None)
        fg_computation_mode = config.get_config('gradient_computation').get('fg_computation_mode')
        print(f'{num_directions=}', file=sys.stderr)
        print(f'gradient config: {dict(config.get_config("gradient_computation"))}', file=sys.stderr)
        simulated_fg = fg_computation_mode == 'sim'

        # model setup: move to device and initialize activity shapes for activity perturbation
        model.to(device)
        if isinstance(model, (nn_training.fg_models.ActivityPerturbedForwardGradientModel,
                              nn_training.simfg_models.ActivityPerturbedSimFGModel)):
            model.eval()
            model.update_activity_shapes(next(iter(train_loader))[0].shape)

        # create trainer object
        trainer = nn_training.train.Trainer(model, device, train_loader, None, val_loader, None, optimizer,
                                            lr_scheduler, config.gradient_computation, num_directions, simulated_fg,
                                            track_additional_stats=False)
        early_stopping = nn_training.train.EarlyStopping(10, 1e-6)

        lr_label = f'{lr:.7f}'.replace('.', '_')
        results_file = results_dir / f'validation_losses__lr_{lr_label}__{str(uuid.uuid4())[:8]}.csv'
        with open(results_file, mode='a') as f:
            f.write(f'epoch,validation_loss\n')

        loss_fn_logger.info(f'Starting training for {lr}.\nWriting epoch results to: {results_file}')

        # ---------- Actual training ----------
        for epoch in range(max_epochs):
            trainer.train_epoch()
            val_loss, _ = trainer.evaluate(val_loader)
            yield np.inf if np.isnan(val_loss) else val_loss

            with open(results_file, mode='a') as f:
                f.write(f'{epoch},{val_loss}\n')

            if np.isnan(val_loss):
                loss_fn_logger.debug(f'Nan val loss after {epoch} epochs, stopping.')
                break

            if early_stopping.early_stop(val_loss):
                loss_fn_logger.debug(f'No improvement over {early_stopping.min_delta} after '
                                     f'{early_stopping.steps_without_improvement} steps, stopping.')
                break

        # NOTE: the following code is only executed if the training was not aborted early by the surrogate model
        # however, if aborted early, the LR was inferior anyway
        loss_fn_logger.info(f'LR: {nn_args.initial_lr:.7f} -- Final loss {val_loss:.4f} after {epoch + 1:3d} epochs.')

    # ---------- Start propulate ----------
    lr_search_with_propulate(loss_fn, cli_args.limits, search_label, cli_args.population_size, cli_args.num_evaluations,
                             cli_args.surrogate_margin, delete_old_checkpoints=cli_args.delete_old_checkpoints,
                             surrogate_kwargs=surrogate_kwargs)


def print_lr_statistics(individuals):
    close_to_min_lrs = [get_lr_from_param_dict(ind.mapping) for ind in individuals]
    tag_lrs = [(min(close_to_min_lrs), "min"), (max(close_to_min_lrs), "max"),
               (min(close_to_min_lrs, key=lambda x: abs(x - statistics.mean(close_to_min_lrs))), "mean"),
               (min(close_to_min_lrs, key=lambda x: abs(x - statistics.median(close_to_min_lrs))), "median")]

    for ind in individuals:
        lr = get_lr_from_param_dict(ind.mapping)
        tags = [tag for _lr, tag in tag_lrs if lr == _lr]
        print(f'loss={ind.loss:.2e}, {lr=:.6f}, {", ".join(tags)}')


def extract_best_lr(cli_args):
    task = cli_args.task

    def search_label_math(function, n, gradient_approach, k=None, tangent_sampler='normal'):
        gradient_type = 'bp' if gradient_approach == 'bp' else 'fg'
        aggregation = {'fg-mean': 'mean', 'frog': 'orthogonal_projection'}.get(gradient_approach, None)
        optimizer = function_optimization.get_algorithm(gradient_type, aggregation=aggregation, k=k,
                                                        tangent_sampler=tangent_sampler)()

        config = {'function': function, 'n': n, 'gradient_approach': gradient_approach, 'k': k,
                  'optimizer': function_optimization.gradient_label(optimizer, ignore_keys=['scaling_correction'])}
        task_label = f'{function}_n={n:04d}'
        approach = gradient_approach
        if gradient_approach != 'bp':
            approach += f'_n={k}' if cli_args.old_label else f'_k={k}'

        label = f'math/{task_label}/{approach}' + ('' if tangent_sampler == 'normal' else f'__{tangent_sampler}')
        return label, config

    def search_label_nn(model, dataset, gradient_approach, k=None):
        config = {'model': model, 'dataset': dataset, 'gradient_approach': gradient_approach, 'k': k}
        return f'nns/{model}__{dataset}/{approach_label(gradient_approach, k)}', config

    approaches = [('bp', None), ('fg-mean', 1), ('fg-mean', 2), ('fg-mean', 4), ('fg-mean', 16),
                  ('frog', 2), ('frog', 4), ('frog', 16)]

    if task == 'math':
        functions = ['rosenbrock', 'styblinski-tang', 'sphere']
        ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        search_configs = dict(search_label_math(function, n, *approach)
                              for function in functions for n in ns for approach in approaches)
    elif task == 'math_custom_tangents':
        tangent_samplers = [f'specific_angle_{angle}' for angle in [15, 30, 45, 60, 75, 90]] + [
            f'varying_length_{scale}' for scale in [2**i for i in range(6)]]
        fg_approaches = [('fg-mean', 16), ('fg-mean', 64)]
        frog_approaches = [('frog', 16), ('frog', 64)]
        ns = [64, 1024]
        search_configs = dict(search_label_math('styblinski-tang', n, *approach, tangent_sampler)
                              for tangent_sampler in tangent_samplers for approach in fg_approaches for n in ns)
        search_configs_frog = dict(search_label_math('styblinski-tang', n, *approach)
                                   for approach in frog_approaches for n in ns)
        search_configs = {**search_configs, **search_configs_frog}

    elif task == 'fc-nn':
        models = ['fc_w=256', 'fc_w=1024', 'fc_w=4096']
        search_configs = dict(search_label_nn(model, 'mnist', *approach)
                              for model in models for approach in approaches)
    elif task == 'sota-nn':
        models = ['resnet18', 'vit']
        datasets = ['mnist', 'cifar10', 'svhn']
        search_configs = dict(search_label_nn(model, dataset, *approach)
                              for model in models for dataset in datasets for approach in approaches)
    else:
        raise ValueError(f'Invalid {task=}')

    results = []
    for search_label, search_config in search_configs.items():
        checkpoint_path = PROPULATE_CKPT_PATH / search_label / 'checkpoints'
        if not checkpoint_path.exists():
            print(f'No checkpoint for {search_label}, skipping.')
            continue
        propulator = propulate.Propulator(loss_fn=lambda x: None, propagator=None, rng=random.Random(0),
                                          checkpoint_path=checkpoint_path)
        sorted_individual = sorted(propulator.population, key=lambda x: x.loss)
        min_loss = sorted_individual[0].loss

        margin = min_loss * cli_args.margin
        individuals_close_to_min_loss = sorted([ind for ind in sorted_individual if ind.loss <= min_loss + margin],
                                               key=lambda ind: (ind.loss, get_lr_from_param_dict(ind.mapping)))

        if len(individuals_close_to_min_loss) > 1:
            print(f'\n{search_label}:')
            if cli_args.verbose:
                print_lr_statistics(individuals_close_to_min_loss)
            close_to_min_lrs = [get_lr_from_param_dict(ind.mapping) for ind in individuals_close_to_min_loss]
            selected_ind = min(individuals_close_to_min_loss, key=lambda ind: abs(
                get_lr_from_param_dict(ind.mapping) - statistics.mean(close_to_min_lrs)))

            print(f'{len(individuals_close_to_min_loss)} lrs close to min loss ({min_loss=}, '
                  f'including all <= {min_loss + margin}). Selecting mean {selected_ind}')
        else:
            selected_ind = sorted_individual[0]
        results.append({**search_config, 'lr': get_lr_from_param_dict(selected_ind.mapping), 'loss': selected_ind.loss})

    pandas.DataFrame(results).to_csv(cli_args.outfile, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform LR Search with Propulate')
    parser.set_defaults(func=lambda _: print('No subparser given'))
    parser.add_argument('--limits', nargs=2, default=[-6., 0.], type=float)
    parser.add_argument('--num_evaluations', default=32, type=int)
    parser.add_argument('--population_size', default=8, type=int)
    parser.add_argument('--surrogate_margin', default=0.5, type=float)
    parser.add_argument('--surrogate_patience', default=0.05, type=float,
                        help='Set the patience of allowed worse steps before canceling a run. Either a fixed number of '
                             'steps (when ≥ 1) or relative to the maximum step count (when < 1).')
    parser.add_argument('--delete_old_checkpoints', action='store_true',
                        help='Delete old checkpoints dir for this run before starting a new run.')
    # gradient options
    parser.add_argument('--gradient', type=str, default='bp', choices=['bp', 'fg-mean', 'frog'])
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('--tangent_sampler', type=str, default='normal')
    subparsers = parser.add_subparsers()

    # math optimization options
    math_parser = subparsers.add_parser('math')
    math_parser.add_argument('--function', type=str, choices=['rosenbrock', 'styblinski-tang', 'sphere'], required=True)
    math_parser.add_argument('-n', type=int, required=True)
    math_parser.add_argument('--early_stopping_patience', default=100, type=int)
    math_parser.set_defaults(func=lambda config: lr_search_for_math(config))

    # nn training options
    nn_parser = subparsers.add_parser('nn')
    nn_parser.add_argument('--device', type=str, default='cpu')
    nn_parser.add_argument('--model', type=str, choices=['fc', 'lenet5', 'resnet18', 'vit', 'mlpmixer'])
    nn_parser.add_argument('--model_width', type=int, help='Overwrite hidden size of fc model.')
    nn_parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'svhn'])
    nn_parser.add_argument('--epochs', type=int, default=200, help='Overwrite default number of epochs')
    nn_parser.set_defaults(func=lambda config: lr_search_for_nn(config))

    # extract best lrs options
    extract_best_parser = subparsers.add_parser('extract_best')
    extract_best_parser.add_argument('--task', type=str, choices=['math', 'math_custom_tangents', 'fc-nn', 'sota-nn'],
                                     required=True)
    extract_best_parser.add_argument('--outfile', type=pathlib.Path, required=True)
    extract_best_parser.add_argument('--old_label', action='store_true')
    extract_best_parser.add_argument('--verbose', action='store_true')
    extract_best_parser.add_argument('--margin', type=float, default=0.05)
    extract_best_parser.set_defaults(func=lambda config: extract_best_lr(config))

    _cli_args = parser.parse_args()
    _cli_args.func(_cli_args)
