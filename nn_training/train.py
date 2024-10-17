import contextlib
import math
import os
import sys

import numpy as np
import pandas
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.profiler import record_function

import nn_training
from nn_training import datasets, models, parse_config
import utils

torch.set_printoptions(precision=2, linewidth=150)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.steps_without_improvement = 0
        self.min_validation_loss = torch.inf

    def reset(self):
        self.steps_without_improvement = 0
        self.min_validation_loss = torch.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:  # new minimum, reset counter
            self.min_validation_loss = validation_loss
            self.steps_without_improvement = 0
        else:  # no improvement, increment counter, check patience and potentially break
            self.steps_without_improvement += 1
            if self.steps_without_improvement >= self.patience:
                return True
        return False


class Trainer:
    def __init__(self, model, device, train_loader, train_test_loader, val_loader, test_loader, optimizer, lr_scheduler,
                 gradient_computation, num_directions=None, simulated_fg=False, track_additional_stats=True,
                 early_stopping_patience=10, early_stopping_min_delta=0):
        self.stats = []
        self.epoch_id = 0
        self.track_additional_stats = track_additional_stats

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.train_test_loader = train_test_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_directions = num_directions
        self.early_stopping = EarlyStopping(early_stopping_patience, early_stopping_min_delta)

        self.timers = {}
        # timers that are called multiple times during an epoch, for the epoch-wise results, we thus want to include
        # their mean and reset them after each epoch
        self._timers_aggregated_by_epoch = {"create_tangents", "forward_and_jvps", "compute_forward_gradients",
                                            "step", "forward", "backward"}
        # timers that should not be included in the epoch time measures, for now, this is just the 'train' timer
        # which covers the entire training (i.e. all epochs).
        self._non_epoch_timers = {"train"}

        if gradient_computation in ['bp']:
            self.train_epoch = self.train_epoch_backprop
        elif gradient_computation in ['fg', 'frog']:
            self.train_epoch = self.train_epoch_simulated_fg if simulated_fg else self.train_epoch_forward_gradient
        else:
            raise ValueError(f'Gradient computation {gradient_computation} currently not supported.')

    @contextlib.contextmanager
    def time_function(self, label):
        # a simple wrapper to combine the timing and profiling context-managers to a single context-manager
        if label not in self.timers:
            self.timers[label] = utils.Timer(
                label, output_format="mean {mean_elapsed_time:.2g}s, total {total_elapsed_time:.2g}s ({count})")
        with self.timers[label], record_function(label):
            yield

    def reset_epoch_wise_timers(self):
        for key in self._timers_aggregated_by_epoch:
            if key in self.timers:
                self.timers[key].reset()

    def get_epoch_time_measurements(self, key_prefix='', key_suffix=''):
        def get_measurement(key):
            # use total value for timers called multiple times during an epoch
            if key in self._timers_aggregated_by_epoch:
                return self.timers[key].total_elapsed_time
            # use last value (i.e. the one for the current epoch) for timers called once per epoch
            return self.timers[key].last_elapsed_time

        return {f'{key_prefix}{key}{key_suffix}': get_measurement(key) for key in self.timers
                if key not in self._non_epoch_timers}

    def print_timers(self):
        max_key_length = max(len(key) for key in self.timers)
        print('\n'.join([f'{key:<{max_key_length}}: {timer}' for key, timer in self.timers.items()]))

    def get_time(self, label, kind='last'):
        if kind == 'last':
            return self.timers[label].last_elapsed_time
        elif kind == 'mean':
            return self.timers[label].mean_elapsed_time
        elif kind == 'total':
            return self.timers[label].total_elapsed_time
        else:
            raise ValueError(f'Invalid time {kind=}.')

    def train_epoch_forward_gradient(self):
        # TODO: reuse tangents for whole epoch or only per batch? per batch probably makes more sense
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            with self.time_function("create_tangents"):
                tangents = self.model.create_tangents(self.num_directions)

            jvps = []
            with self.time_function("forward_and_jvps"):
                for i in range(self.num_directions):
                    with fwAD.dual_level():
                        self.model.set_tangent(tangents, i)
                        output = self.model(data)
                        loss = F.cross_entropy(output, target)
                        jvps.append(fwAD.unpack_dual(loss).tangent)
            with self.time_function("compute_forward_gradients"):
                self.model.compute_gradients(torch.stack(jvps), tangents)

            with self.time_function("step"):
                self.optimizer.step()
                self.lr_scheduler.step()
            print(f'\rTrain loss in batch {batch_idx:4d}: {loss:6.3f} jvp: {torch.stack(jvps).sum():6.3f}',
                  end='\r', flush=True, file=sys.stderr)
            if loss.isnan():
                print(f'Aborting epoch early, nan values.', file=sys.stderr)
                break

            if self.track_additional_stats:
                self.stats.append({
                    'epoch': self.epoch_id, 'batch': batch_idx, 'jvp': torch.stack(jvps).sum().item(),
                    **compute_stats(tangents, 'tangent'),
                    **compute_stats({name: param.grad for name, param in self.model.actual_parameters.items()},
                                    'gradient')})
        self.model.sync_weights()

    def train_epoch_simulated_fg(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # standard forward and backward pass
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            with self.time_function("forward"):
                output = self.model(data)
                loss = F.cross_entropy(output, target)
            with self.time_function("backward"):
                loss.backward()

            # compute FG gradients
            with self.time_function("create_tangents"):
                tangents = self.model.create_tangents(self.num_directions)
            with self.time_function("compute_forward_gradients"):
                self.model.compute_gradients(tangents=tangents)

            # perform optimization step
            with self.time_function("step"):
                self.optimizer.step()
                self.lr_scheduler.step()
            print(f'\rTrain loss in batch {batch_idx:4d}: {loss:6.3f}', end='\r', flush=True, file=sys.stderr)
            if loss.isnan():
                print(f'Aborting epoch early, nan values.', file=sys.stderr)
                break

            # collect statistics, currently not tracking jvps, would need to return jvps from within the model
            if self.track_additional_stats:
                self.stats.append({
                    'epoch': self.epoch_id, 'batch': batch_idx, **compute_stats(tangents, 'tangent'),
                    **compute_stats({name: param.grad for name, param in self.model.actual_parameters.items()},
                                    'gradient')})

    def train_epoch_backprop(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            with self.time_function("forward"):
                output = self.model(data)
                loss = F.cross_entropy(output, target)
            with self.time_function("backward"):
                loss.backward()
            if self.track_additional_stats:
                self.stats.append({'epoch': self.epoch_id, 'batch': batch_idx, **compute_stats(
                    {name: param.grad for name, param in self.model.named_parameters()}, 'gradient')})
            with self.time_function("step"):
                self.optimizer.step()
                self.lr_scheduler.step()
            if loss.isnan():
                print(f'Aborting epoch early, nan values.', file=sys.stderr)
                break

    def evaluate(self, dataloader):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                prediction = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        loss /= len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        return loss, accuracy

    def train(self, max_epochs, progress_bar=True, break_at_loss=10):
        with self.time_function("train"):
            return self._train(max_epochs, progress_bar, break_at_loss)

    def _train(self, max_epochs, progress_bar=True, break_at_loss=10):
        self.early_stopping.reset()
        self.model.to(self.device)
        if isinstance(self.model, (nn_training.fg_models.ActivityPerturbedForwardGradientModel,
                                   nn_training.simfg_models.ActivityPerturbedSimFGModel)):
            self.model.eval()
            self.model.update_activity_shapes(next(iter(self.train_loader))[0].shape)
        if isinstance(self.model, nn_training.fg_models.ForwardGradientBaseModel):
            print(self.model.summarize(), file=sys.stderr)
        results = []
        for epoch in tqdm.trange(max_epochs + 1, ncols=100, disable=not progress_bar):
            self.reset_epoch_wise_timers()
            self.epoch_id = epoch

            if epoch > 0:
                with self.time_function("train_epoch"):
                    self.train_epoch()

            with self.time_function("evaluate_epoch"):
                train_loss, train_accuracy = self.evaluate(self.train_test_loader
                                                           if self.val_loader else self.train_loader)
                val_loss, val_accuracy = self.evaluate(self.val_loader) if self.val_loader else (np.nan, np.nan)
                test_loss, test_accuracy = self.evaluate(self.test_loader) if self.test_loader else (np.nan, np.nan)

            epoch_results = {'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy,
                             'val_loss': val_loss, 'val_accuracy': val_accuracy,
                             'test_loss': test_loss, 'test_accuracy': test_accuracy, 'time__train_epoch_s': 0,
                             # note: time__train_epoch_s is overwritten by get_epoch_time_measurements for epoch > 0
                             **self.get_epoch_time_measurements('time__', '_s')}
            results.append(epoch_results)

            epoch_report = 'Epoch {epoch:>4d},    Train: {train_accuracy:> 7.2%} (Acc), {train_loss:> 5.3f} (Loss), '
            if self.val_loader:
                epoch_report += '   Val: {val_accuracy:> 7.2%} (Acc), {val_loss:> 5.3f} (Loss), '
            if self.test_loader:
                epoch_report += '   Test: {test_accuracy:> 7.2%} (Acc), {test_loss:> 5.3f} (Loss), '
            epoch_report += ('   Time: {time__train_epoch_s:.2f}s (Train), {time__evaluate_epoch_s:.2f}s (Eval)'
                             '   LR: {lr:.2g}')

            print(epoch_report.format(**epoch_results, lr=self.optimizer.param_groups[0]['lr']))

            loss_for_stop_criterion = ([loss for loss in [val_loss, test_loss, train_loss]
                                        if not math.isnan(loss)] + [np.nan])[0]
            if break_at_loss and (loss_for_stop_criterion >= break_at_loss or math.isnan(loss_for_stop_criterion)):
                print(f'Stopping in {epoch=} at {loss_for_stop_criterion=}')
                break

            if self.early_stopping.early_stop(loss_for_stop_criterion):
                print(f'No improvement over {self.early_stopping.min_delta} after '
                      f'{self.early_stopping.steps_without_improvement} steps, stopping.')
                break

        stats_df = pandas.DataFrame(self.stats).reset_index() if self.track_additional_stats else None
        return pandas.DataFrame(results), stats_df


def compute_stats(parameters, label):
    results = {f'{label}_sum': 0., f'{label}_norm': 0., f'{label}_abs_mean': 0.}
    element_count = 0
    for param in parameters.values():
        results[f'{label}_sum'] += param.sum().item()
        results[f'{label}_norm'] += torch.dot(param.flatten(), param.flatten()).item()
        results[f'{label}_abs_mean'] += param.abs().sum().item()
        element_count += len(param.flatten())
    results[f'{label}_abs_mean'] /= element_count
    results[f'{label}_mean'] = results[f'{label}_sum'] / element_count
    results[f'{label}_norm'] = math.sqrt(results[f'{label}_norm'])
    results[f'{label}_dim'] = element_count
    return results


def create_lr_scheduler(lr_schedule, optimizer, max_epochs, milestones=(0.5, 0.75), gamma=0.1):
    if lr_schedule == 'step':
        milestones = [int(milestone * max_epochs) for milestone in milestones]  # convert from percentage to epochs
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    elif lr_schedule in ['cosine', 'cosine_annealing']:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif lr_schedule in ['plateau', 'reduce_on_plateau']:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif lr_schedule in ['decay', 'baydin', 'fg']:  # LR decay from Baydin et al.
        return optim.lr_scheduler.LambdaLR(optimizer, lambda i: torch.e ** (-i * 1e-4))
    else:
        return optim.lr_scheduler.LambdaLR(optimizer, lambda i: 1)  # constant, no change


def create_optimizer_and_lr_schedule(model, optimizer_type, initial_lr, lr_schedule, max_epochs, momentum, nesterov,
                                     weight_decay, decoupled_weight_decay=False):
    if optimizer_type == 'Adam':
        if decoupled_weight_decay:
            optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, nesterov=nesterov,
                              weight_decay=weight_decay)
    else:
        raise ValueError(f'Invalid {optimizer_type=}')
    scheduler = create_lr_scheduler(lr_schedule, optimizer, max_epochs, milestones=(0.5, 0.75), gamma=0.1)
    return optimizer, scheduler


def create_optimizer_and_lr_schedule_from_config(model, config, max_epochs):
    optimizer_config = config.get_config('optimizer')
    kwargs = {
        'optimizer_type': optimizer_config.get('optimizer_type', 'SGD'),
        'initial_lr': optimizer_config.getfloat('initial_lr'),
        'lr_schedule': optimizer_config.get('lr_schedule', 'constant'),
        'max_epochs': max_epochs,
        'momentum': optimizer_config.getfloat('momentum', 0.),
        'nesterov': optimizer_config.getboolean('nesterov', False),
        'weight_decay': optimizer_config.getfloat('weight_decay', 0.),
        'decoupled_weight_decay': optimizer_config.getboolean('decoupled_weight_decay', False),
    }
    return create_optimizer_and_lr_schedule(model, **kwargs)


def train(config, print_model=False, track_additional_stats=False):
    torch.manual_seed(config.get('seed', datatype=int))
    device = torch.device(config.device)
    device_specific_kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}

    train_loader, train_test_loader, val_loader, test_loader = datasets.get_dataloaders(
        config, device_specific_kwargs, with_val_set=True)

    epochs = config.get('epochs', datatype=int)
    model = models.get_model(config).to(device)
    optimizer, lr_scheduler = create_optimizer_and_lr_schedule_from_config(model, config, epochs)
    if print_model:
        print(model, file=sys.stderr)

    num_directions = config.get_config('gradient_computation').getint('num_directions', None)
    fg_computation_mode = config.get_config('gradient_computation').get('fg_computation_mode')
    simulated_fg = fg_computation_mode == 'sim'
    trainer = Trainer(model, device, train_loader, train_test_loader, val_loader, test_loader, optimizer, lr_scheduler,
                      config.gradient_computation, num_directions, simulated_fg, track_additional_stats)
    with record_function("trainer.train"):
        results, stats = trainer.train(config.get('epochs', datatype=int), config.progress_bar)

    results['k'] = num_directions
    results['seed'] = config.get('seed', datatype=int)
    results['fg_computation_mode'] = fg_computation_mode
    for key, value in model.model_config.items():
        results[key] = str(value)

    utils.save_dataframe(results, config)
    if track_additional_stats:
        path = utils.append_suffix(config.output_path, '_stats.csv')
        print(f'Saving to {path}', file=sys.stderr)
        stats.to_csv(path, index=False)

    return results, stats


def main():
    parser = parse_config.parse_cli_args()
    parser.add_argument('--print_model', action='store_true')
    parser.add_argument('--track_additional_stats', action='store_true')
    parser.add_argument('--deterministic', action='store_true',
                        help='Whether to activate torch.use_deterministic_algorithms for exact reproducibility.')
    parser.add_argument('--profile', action='store_true',
                        help='Use torch profiler to profile the training. Warning: profiling and processing the results'
                             ' can add a significant overhead to the training run. It is advisable to keep the number'
                             ' of epochs low.')
    args = parser.parse_args()
    if args.deterministic:
        # torch deterministic requires CUBLAS_WORKSPACE_CONFIG=:4096:8
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)

    config = parse_config.Configuration(args)
    config.save()

    with utils.TorchProfiler(disable=not args.profile):
        _, stats = train(config, args.print_model)


if __name__ == '__main__':
    print(f'{torch.__version__=}', file=sys.stderr)
    main()
