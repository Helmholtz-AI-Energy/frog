import math
import warnings

import torch
import torch.autograd.forward_ad as fwAD

import forward_gradient


class GradientBase:
    is_deterministic = True
    _vars_to_print = ['seed']

    def __init__(self, seed=0):
        self.seed = seed
        self.random_numer_generator = torch.Generator()
        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = seed
        self.random_numer_generator.manual_seed(self.seed)

    def get_config(self):
        return {'type': type(self).__name__, 'seed': self.seed}

    def __str__(self):
        return f'{self.__class__.__name__}({", ".join([f"{key}={vars(self)[key]}" for key in self._vars_to_print])})'

    def __repr__(self):
        return str(self)

    def compute_gradient(self, func, position):
        pass


class Backpropagation(GradientBase):
    def __init__(self, seed=0):
        super().__init__(seed)

    def compute_gradient(self, func, position):
        # compute gradients with backpropagation
        position = position.clone()
        position.requires_grad = True
        position.grad = None
        value = func(position)  # forward pass
        value.backward()  # backward pass to compute gradients value wrt the position
        return position.grad


class ForwardModeAD(GradientBase):
    def __init__(self, seed=0):
        super().__init__(seed)

    @staticmethod
    def directional_derivative(func, position, direction):
        with fwAD.dual_level():
            # create dual tensor: associate primal with tangent
            dual_input = fwAD.make_dual(position, direction)
            dual_output = func(dual_input)
            # return directional derivative
            return fwAD.unpack_dual(dual_output).tangent

    @staticmethod
    def forward_gradient(directions_v, directional_derivatives, agg='mean', fallback=None, **kwargs):
        try:
            return forward_gradient.forward_gradient(directions_v, directional_derivatives, agg, **kwargs)
        except RuntimeError as e:
            if fallback and fallback != agg:
                print(e)
                print(f'Falling back to {fallback}')
                return ForwardModeAD.forward_gradient(
                    directions_v, directional_derivatives, agg=fallback, fallback=None)
            else:
                raise e

    def directional_derivatives_to_gradient(self, directions, directional_derivatives):
        # combine directional derivatives to the gradient. As the directions are the standard basis vectors, the
        # directional derivatives are the partial derivatives, making up the gradient together.
        return directional_derivatives

    def get_directions(self, input_dimension):
        return torch.eye(input_dimension)

    def compute_gradient(self, func, position):
        # current_position is the primal, standard basis vectors as the tangent
        directions_v = self.get_directions(len(position))

        # compute directional gradients in the directions v_i and assemble the gradient
        directional_derivatives = torch.tensor([self.directional_derivative(func, position, direction)
                                                for direction in directions_v], dtype=torch.float32)
        gradient = self.directional_derivatives_to_gradient(directions_v, directional_derivatives)
        return gradient


class ForwardGradient(ForwardModeAD):
    _vars_to_print = ['seed', 'num_directions', 'normalize_tangents', 'gradient_aggregation', 'direction_distribution',
                      'ensure_linear_independence']

    def __init__(self, seed=0, num_directions=1, gradient_aggregation='sum', normalize_tangents=False,
                 direction_distribution='normal', ensure_linear_independence=None, scaling_correction=False):
        super().__init__(seed)
        self.normalize_tangents = normalize_tangents
        self.num_directions = num_directions
        self.gradient_aggregation = gradient_aggregation
        self.direction_distribution = direction_distribution
        self.scaling_correction = scaling_correction

        if ensure_linear_independence is None:
            # the check is not too expensive for small k and the risk of linearly dependent tangents is infinitesimal
            # for larger n. While we don't have access to n here, we can use k as a stand-in (even if we don't skip
            # unnecessary experiments where k > n, we cannot select independent tangents in that case anyway)
            self.ensure_linear_independence = (gradient_aggregation == 'orthogonal_projection'
                                               and num_directions < 32 and direction_distribution == 'rademacher')
        else:
            self.ensure_linear_independence = ensure_linear_independence

    def get_config(self):
        return {**super().get_config(), 'num_directions': self.num_directions,
                'gradient_aggregation': self.gradient_aggregation, 'normalize_tangents': self.normalize_tangents,
                'direction_distribution': self.direction_distribution, 'scaling_correction': self.scaling_correction,
                'ensure_linear_independence': self.ensure_linear_independence}

    def expected_tangent_length(self, n):
        if self.normalize_tangents:
            return 1

        if self.direction_distribution == 'normal':
            return math.sqrt(n)

        raise ValueError(f'{self.direction_distribution} currently not supported for tangent length prediction.')

    def directional_derivatives_to_gradient(self, directions, directional_derivatives):
        # forward gradient: approximate actual gradient by
        #   - multiplying each directional derivative with its direction v to obtain the forward gradient
        #     (reshape directional_derivatives to (num_directions, 1) with [:, None] for row-wise multiplication)
        #   - averaging over the forward gradients for all directions v
        return self.forward_gradient(directions, directional_derivatives, agg=self.gradient_aggregation,
                                     scaling_correction=self.scaling_correction,
                                     expected_tangent_length_fn=self.expected_tangent_length)

    def get_directions(self, input_dimension, max_depth=10):
        resample = max_depth if self.ensure_linear_independence and self.num_directions <= input_dimension else 0
        postprocessing = []
        if self.normalize_tangents:
            postprocessing.append(forward_gradient.normalize)

        return forward_gradient.sample_tangents(
            (self.num_directions, input_dimension), sampler=self.direction_distribution, postprocessing=postprocessing,
            generator=self.random_numer_generator, resample=resample, device='cpu')


class FixedDirectionForwardGradient(ForwardModeAD):
    _vars_to_print = ['seed', 'directions_v']

    def __init__(self, directions, seed=0, gradient_aggregation='sum'):
        super().__init__(seed)
        self.directions_v = torch.tensor(directions, dtype=torch.float)
        self.gradient_aggregation = gradient_aggregation

    def get_config(self):
        return {**super().get_config(), 'directions': self.directions_v.tolist()}

    def directional_derivatives_to_gradient(self, directions, directional_derivatives):
        # forward gradient: approximate actual gradient by
        #   - multiplying each directional derivative with its direction v to obtain the forward gradient
        #     (reshape directional_derivatives to (num_directions, 1) with [:, None] for row-wise multiplication)
        #   - averaging over the forward gradients for all directions v
        return self.forward_gradient(directions, directional_derivatives, agg=self.gradient_aggregation)

    def get_directions(self, input_dimension):
        return self.directions_v


def get_algorithm(gradient_type, k=1, tangent_sampler='normal', normalize=False, orthonormalize=False,
                  aggregation=None, scaling_correction=False, independent_tangents=None):
    if gradient_type == 'bp':
        return lambda **kwargs: Backpropagation(**kwargs)
    elif gradient_type in ['fg', 'forward_gradient']:
        if orthonormalize:
            if aggregation is not None and aggregation != 'orthogonal_projection':
                warnings.warn(f'Setting {orthonormalize=} overwrites {aggregation=} with \'orthogonal_projection\'')
            aggregation = 'orthogonal_projection'
        fg_kwargs = {'num_directions': k, 'gradient_aggregation': aggregation, 'normalize_tangents': normalize,
                     'direction_distribution': tangent_sampler, 'ensure_linear_independence': independent_tangents,
                     'scaling_correction': scaling_correction}
        fg_kwargs = {key: value for key, value in fg_kwargs.items() if value is not None}
        return lambda **kwargs: ForwardGradient(**{**fg_kwargs, **kwargs})
    else:
        raise ValueError(f'Invalid {gradient_type=}.')


def gradient_label(optimizer, ignore_keys=None):
    ignore_keys = [] if ignore_keys is None else ignore_keys
    if isinstance(optimizer, ForwardGradient):
        config = {'k': optimizer.num_directions, 'agg': optimizer.gradient_aggregation,
                  'normalize': optimizer.normalize_tangents, 'sampler': optimizer.direction_distribution,
                  'scaling_correction': optimizer.scaling_correction}
        return f'FG({",".join(f"{key}={value}" for key, value in config.items() if key not in ignore_keys)})'
    else:
        return type(optimizer).__name__
