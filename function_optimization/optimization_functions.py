import abc
import dataclasses
import typing

import numpy as np
import torch

import utils


@dataclasses.dataclass
class OptimizationFunction:
    max_input_dimension = None

    input_dimension: int = 2
    output_dimension: int = 1
    domain: np.ndarray = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    name: str = ''

    starting_position: tuple = (0, 0)
    steps: int = 1000
    best_lr_true_gradient: float = 0.1
    best_lr_approx_gradient: float = 0.1
    lrs_to_compare: typing.Iterable[float] = (0.1, 0.01, 0.001)

    plotting_limits: np.ndarray = np.array([[-10, 10], [-10, 10]])
    plotting_ylim: np.ndarray = np.array([-np.inf, np.inf])
    equal_axis: bool = False
    plotting_contour_scale: str = 'linear'
    plotting_zoomed_x_max: float = 100
    plotting_zoomed_y_max: float = 1

    def get_best_lr(self, optimizer):
        if optimizer in ['Backpropagation', 'ForwardModeAD']:
            return self.best_lr_true_gradient
        elif optimizer in ['ForwardGradient', 'FixedDirectionForwardGradient']:
            return self.best_lr_approx_gradient
        raise ValueError(f'Unknown optimizer {optimizer}')

    @property
    def xlim(self):
        return self.plotting_limits[0]

    @property
    def ylim(self):
        return self.plotting_limits[1]

    @property
    def minima(self):
        return None

    @property
    def minimum_value(self):
        return None if self.minima is None else self(self.minima[0])

    @abc.abstractmethod
    def apply(self, x):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            x = args[0] if isinstance(args[0], torch.Tensor) else torch.tensor(args[0])
            return self.apply(x)
        elif len(args) == self.input_dimension:
            return self.apply(torch.tensor(args))
        else:
            raise ValueError(f'Invalid number of arguments, expected 1 or {self.input_dimension} but got {len(args)}.')


@dataclasses.dataclass
class Beale(OptimizationFunction):
    max_input_dimension = 2
    name: str = 'Beale'
    best_lr_true_gradient: float = 0.01
    best_lr_approx_gradient: float = 0.01

    plotting_limits: np.ndarray = np.array([[0, 3.5], [-0.6, 0.6]])
    plotting_ylim: np.ndarray = np.array([0, 3])
    plotting_zoomed_x_max: float = 300
    plotting_zoomed_y_max: float = 0.5

    @property
    def minima(self):
        return [(3, 0.5)]

    def apply(self, x):
        return ((1.5 - x[0] + x[0] * x[1]) ** 2 +
                (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 +
                (2.625 - x[0] + x[0] * x[1] ** 3) ** 2)


@dataclasses.dataclass
class Plane(OptimizationFunction):
    name: str = 'Plane'
    slope: float = -1.0

    plotting_limits: np.ndarray = np.array([[-2, 2], [-2, 2]])
    plotting_ylim: np.ndarray = np.array([0, 4])
    equal_axis: bool = True

    def apply(self, x):
        return (x * self.slope).sum()


@dataclasses.dataclass
class Sphere(OptimizationFunction):
    name: str = 'Sphere'

    starting_position: tuple = (-1.5, -1)
    lrs_to_compare: typing.Iterable[float] = (1, 0.1, 0.01, 0.001)

    plotting_limits: np.ndarray = np.array([[-2, 2], [-2, 2]])
    plotting_ylim: np.ndarray = np.array([0, 4])
    equal_axis: bool = True

    @property
    def minima(self):
        return [tuple([0] * self.input_dimension)]

    def apply(self, x):
        return (x ** 2).sum()


@dataclasses.dataclass
class Rosenbrock(OptimizationFunction):
    name: str = 'Rosenbrock'

    a: float = 1
    b: float = 100

    starting_position: tuple = (-1, 0)
    steps: int = 15000
    best_lr_true_gradient: float = 1e-3
    best_lr_approx_gradient: float = 5e-4
    lrs_to_compare: typing.Iterable[float] = (2.5e-3, 1e-3, 7.5e-4, 5e-4)

    plotting_limits: np.ndarray = np.array([[-1.2, 1.2], [-0.4, 1.3]])
    plotting_ylim: np.ndarray = np.array([0, 3])

    @property
    def minima(self):
        return [tuple([1] * self.input_dimension)]

    def apply(self, x):
        return (self.b * (x[1:] - x[:-1] ** 2) ** 2 + (self.a - x[:-1]) ** 2).sum()


@dataclasses.dataclass
class EggCrate(OptimizationFunction):
    max_input_dimension = 2
    # Global minimum: at (0, 0) with value 0
    name: str = 'Egg-Crate'

    starting_position: tuple = (-4, -2)
    best_lr_true_gradient: float = 0.075  # oscillates around the global minimum
    best_lr_approx_gradient: float = 0.01  # does not reach the global minimum
    lrs_to_compare: typing.Iterable[float] = (0.001, 0.01, 0.075)

    plotting_limits: np.ndarray = np.array([[-5, 5], [-5, 5]])
    plotting_ylim: np.ndarray = np.array([0, 100])

    @property
    def minima(self):
        return [(0, 0)]

    def apply(self, x):
        return x[0] ** 2 + x[1] ** 2 + 25 * (torch.sin(x[0]) ** 2 + torch.sin(x[1]) ** 2)


@dataclasses.dataclass
class Ackley(OptimizationFunction):
    max_input_dimension = 2
    # Global minimum: at (0, 0) with value 0
    name: str = 'Ackley'

    starting_position: tuple = (-20, 0)
    steps: int = 2500
    best_lr_true_gradient: float = 0.8  # oscillates around the global minimum
    best_lr_approx_gradient: float = 0.075  # oscillates around the global minimum
    lrs_to_compare: typing.Iterable[float] = (0.01, 0.075, 0.1, 0.8)

    plotting_limits: np.ndarray = np.array([[-30, 30], [-30, 30]])
    plotting_ylim: np.ndarray = np.array([0, 25])
    equal_axis: bool = True

    @property
    def minima(self):
        return [(0, 0)]

    def apply(self, x):
        return (-20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
                - torch.exp(0.5 * (torch.cos(2 * torch.pi * x[0]) + torch.cos(2 * torch.pi * x[1]))) + torch.e + 20)


@dataclasses.dataclass
class Forest(OptimizationFunction):
    max_input_dimension = 2
    # Global minimum: at (0, 0) with value 0
    name: str = 'Forest Function'
    starting_position: tuple = (-4, -4)
    best_lr_true_gradient: float = 0.05  # reaches the global minimum, oscillates slightly
    best_lr_approx_gradient: float = 0.01  # reaches a local minima
    lrs_to_compare: typing.Iterable[float] = (0.005, 0.01, 0.05, 0.1)

    plotting_limits: np.ndarray = np.array([[-5, 5], [-5, 5]])
    plotting_ylim: np.ndarray = np.array([0, 25])

    @property
    def minima(self):
        return [(0, 0)]

    def apply(self, x):
        return (abs(x[0]) + abs(x[1])) * torch.exp(-torch.sin(x[0] ** 2) + torch.sin(x[1] ** 2))


@dataclasses.dataclass
class GoldsteinPrice(OptimizationFunction):
    max_input_dimension = 2
    # 4 local minima, global minimum: at (0, -1) with value 3
    name: str = 'Goldstein and Price'
    starting_position: tuple = (-1, -1)
    best_lr_true_gradient: float = 0.00025  # reaches the global minimum
    best_lr_approx_gradient: float = 0.0001  # reaches the global minimum
    lrs_to_compare: typing.Iterable[float] = (0.0001, 0.00025, 0.0005)

    plotting_limits: np.ndarray = np.array([[-2, 2], [-3, 1]])
    plotting_ylim: np.ndarray = np.array([3, 1e4])
    equal_axis: bool = True

    @property
    def minima(self):
        return [(0, -1)]

    def apply(self, x):
        return ((1 + (x[0] + x[1] + 1) ** 2 *
                (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) *
                (30 + (2 * x[0] - 3 * x[1]) ** 2
                 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)))


@dataclasses.dataclass
class LevyMontalvo(OptimizationFunction):
    max_input_dimension = 2
    # many local minima, global minimum: at (1, 1) with value 0
    name: str = 'Levy and Montalvo N.1'

    starting_position: tuple = (-8, -8)
    best_lr_true_gradient: float = 0.165  # oscillates around the global minimum
    best_lr_approx_gradient: float = 0.05  # reaches the global minimum
    lrs_to_compare: typing.Iterable[float] = (0.01, 0.05, 0.165)

    plotting_limits: np.ndarray = np.array([[-10, 10], [-10, 10]])
    plotting_ylim: np.ndarray = np.array([0, 50])

    @property
    def minima(self):
        return [(1, 1)]

    def apply(self, x):
        x_bar = 1 + 0.25 * (x[0] - 1)
        y_bar = 1 + 0.25 * (x[1] - 1)
        return torch.pi / 2 * (
                10 * torch.sin(torch.pi * x_bar) ** 2
                + (x_bar - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * y_bar) ** 2)
                + (y_bar - 1) ** 2)


@dataclasses.dataclass
class StyblinskiTang(OptimizationFunction):
    # Global minimum: at (-2.903534, -2.903534) with value −78.33236
    name: str = 'Styblinski-Tang'

    starting_position: tuple = (1, 0)
    best_lr_true_gradient: float = 0.05  # reaches a local minimum
    best_lr_approx_gradient: float = 0.03  # escapes the first local minimum but also the global minimum
    lrs_to_compare: typing.Iterable[float] = (0.01, 0.03, 0.05, 0.1)

    plotting_limits: np.ndarray = np.array([[-5, 5], [-5, 5]])
    plotting_ylim: np.ndarray = np.array([-80, 0])

    @property
    def minima(self):
        return [tuple([-2.903534] * self.input_dimension)]

    def apply(self, x):
        return 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x).sum()


@dataclasses.dataclass
class ThreeHumpCamel(OptimizationFunction):
    max_input_dimension = 2
    # 3 local minima, global minimum: at (0, 0) with value 0
    name: str = 'Three-Hump Camel'

    starting_position: tuple = (-4, -4)
    best_lr_true_gradient: float = 0.005
    best_lr_approx_gradient: float = 0.001
    lrs_to_compare: typing.Iterable[float] = (0.001, 0.005, 0.0075, 0.01)
    steps: int = 5000

    plotting_limits: np.ndarray = np.array([[-5, 5], [-5, 5]])
    plotting_ylim: np.ndarray = np.array([0, 25])
    plotting_contour_scale: str = 'log10'

    @property
    def minima(self):
        return [(0, 0)]

    def apply(self, x):
        return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + 1 / 6 * x[0] ** 6 + x[0] * x[1] + x[1] ** 2


function_by_name = {utils.path_friendly_string(function_class.name): function_class
                    for function_class in utils.get_subclasses(OptimizationFunction)}


def get_function_by_name(name):
    return function_by_name[utils.path_friendly_string(name)]


start_positions = {
    '0': lambda n: [0 for _ in range(n)],
    '-0.5': lambda n: [-0.5 for _ in range(n)],
    '-+0.5': lambda n: [0.5 if i % 2 else -0.5 for i in range(n)],
    '-10': lambda n: [-1] + [0 for _ in range(n - 1)],
    '-1': lambda n: [-1 for _ in range(n)],
}

function_configs = {
    'rosenbrock': {
        'start': '-10',
        'max_steps': 25000,
        'global_minimum': lambda n: 0,
    },
    'styblinski-tang': {
        'start': '0',
        'max_steps': 1000,
        'global_minimum': lambda n: -39.16599 * n,
    },
    'sphere': {
        'start': '-1',
        'max_steps': 1000,
        'global_minimum': lambda n: 0,
    },
}


def get_global_minimum(function_name, n):
    if function_name not in function_configs:
        raise ValueError(f'No config available for function {function_name}. '
                         f'Available functions are {function_configs.keys()}')
    return function_configs[function_name]['global_minimum'](n)


def get_function_config(function_name, start=None):
    function = get_function_by_name(function_name)()
    start = start or function_configs[function_name]['start']
    starting_position_fn = start_positions[start]
    max_steps = function_configs[function_name]['max_steps']
    return function, starting_position_fn, max_steps
