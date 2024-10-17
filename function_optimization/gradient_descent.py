import numpy as np
import pandas
import torch
import tqdm.auto

import utils


class GradientDescent:
    def __init__(self, step_size, gradient_computation, initial_position, func, early_stopping_patience=0):
        self.step_size = step_size
        self.gradient_computation = gradient_computation
        self.steps_without_improvement = 0
        self.early_stopping_patience = early_stopping_patience

        self.func = func
        self.initial_position = initial_position
        self.current_position = torch.tensor(self.initial_position, dtype=torch.float)
        self.current_value = self.func(self.current_position)
        self.last_value = np.nan

    def get_config(self):
        return {'step_size': self.step_size, 'early_stopping_patience': self.early_stopping_patience}

    def reset(self):
        self.steps_without_improvement = 0
        self.current_position = torch.tensor(self.initial_position, dtype=torch.float)
        self.current_value = self.func(self.current_position)
        self.last_value = np.nan

    def set_step_size(self, step_size):
        self.step_size = step_size

    def step(self):
        # compute gradient
        gradient = self.gradient_computation.compute_gradient(self.func, self.current_position)
        # step and update position
        self.current_position = self.current_position - self.step_size * gradient

        # update last and current value
        self.last_value = self.current_value
        self.current_value = self.func(self.current_position)

        return self.current_position, self.current_value

    def check_stop_criteria(self):
        reason = ''
        # stop on nan
        if self.current_position.isnan().any().item():
            return 'NaN values'

        # early stopping
        if self.current_value >= self.last_value:
            self.steps_without_improvement += 1
        else:
            self.steps_without_improvement = 0
        if 0 < self.early_stopping_patience <= self.steps_without_improvement:
            reason = f'{self.steps_without_improvement} steps without improvement'
        return reason


def optimize_make_dataframe(history, optimizer, gradient_computation, steps, include_position):
    history_df = pandas.DataFrame([{key: step[key] for key in ['y', 'step_duration_s']} for step in history])
    if include_position:
        positions = pandas.DataFrame(np.asarray([step[0].numpy() for step in history]),
                                     columns=[f'x{i}' for i in range(len(optimizer.starting_position))])
        history_df = pandas.concat([history_df, positions], axis=1)
    history_df['time_s'] = history_df['step_duration_s'].cumsum()
    history_df.index.name = 'step'
    history_df = history_df.reset_index()

    # config columns start with an underscore
    if include_position:
        history_df['_starting_position'] = str(tuple(optimizer.starting_position.tolist()))
    history_df['_steps'] = steps

    for key, value in optimizer.get_config().items():
        history_df[f'_optimizer_{key}'] = str(value)

    for key, value in gradient_computation.get_config().items():
        if key == 'directions':
            pretty_directions = utils.apply_to_list(value, lambda x: f'{x:g}')
            history_df[f'_optimizer_{key}'] = str(pretty_directions).replace("'", "")
        else:
            history_df[f'_optimizer_{key}'] = str(value)
    return history_df


def optimize(func, starting_position, steps, step_size, gradient_computation, early_stopping_patience=0,
             debug=False, progress_bar=False, progress_bar_kwargs=None, history_include_position=False):
    history = []
    optimizer = GradientDescent(step_size, gradient_computation, starting_position, func, early_stopping_patience)
    step_timer = utils.Timer()

    def update_history(step_duration=0):
        history.append({'x': optimizer.current_position if history_include_position else None,
                        'y': optimizer.current_value.item(),
                        'step_duration_s': step_duration})

    update_history()

    for i in tqdm.trange(steps, disable=not progress_bar, **(progress_bar_kwargs or {})):
        with step_timer:
            optimizer.step()
        update_history(step_timer.last_elapsed_time)

        reason = optimizer.check_stop_criteria()
        if reason:
            if debug:
                print(f'{reason}, stopping computation in step {i + 1}/{steps}')
            break

    history_df = optimize_make_dataframe(history, optimizer, gradient_computation, steps, history_include_position)
    return history_df
