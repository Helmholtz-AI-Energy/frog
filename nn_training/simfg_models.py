import abc

import numpy as np
import prefixed
import torch
from torch import nn

import nn_training.fg_models


class SimulatedForwardGradientModel(nn_training.fg_models.ForwardGradientBaseModel, abc.ABC):
    """
    Base model for simulated forward gradients.
    Handles computation of both local and global jvps and forward gradients from these jvps.

    Training workflow:
    - for each batch (x, y*):
      - forward as with BP: loss = L(model(x), y*)
      - compute true gradients with BP: loss.backward()
      - compute the forward gradients from the true gradients using model.compute_gradients(...)
        this performs the following steps:
        - create tangents if none were given: tangents = model.create_tangents(k)
        - compute forward gradients: forward_gradients = model.compute_forward_gradients(tangents)
          Note: this *uses* the true gradient .grad computed in the backward pass
        - set gradients to forward gradients: model.overwrite_gradients(forward_gradients)
          Note: this *overwrites* the .grad of the parameters with the forward gradients
      - continue as if the gradients had been computed by BP
    """
    def __init__(self, model, local_jvps=False, global_jvp_agg='sum', **kwargs):
        super(SimulatedForwardGradientModel, self).__init__(model, **kwargs)
        self._local_jvps = local_jvps
        self.global_jvp_agg = global_jvp_agg
        self.model_config = {**self.model_config, 'fg_model__local_jvps': local_jvps,
                             'fg_model__global_jvp_agg': global_jvp_agg}

    def compute_gradients(self, tangents=None, num_directions=None):
        # input sanitation
        if tangents is None and num_directions is None:
            raise ValueError(f'Either tangents or num_directions need to be provided to compute forward gradients.')
        elif tangents is not None and num_directions is not None and next(
                iter(tangents.values())).shape[0] != num_directions:
            raise RuntimeWarning('Received both tangents dict and num_directions but number of tangents '
                                 f'{iter(tangents.values()).shape[0]} does not match {num_directions=}.'
                                 'Will continue with given tangents and ignore num_directions.')

        # compute tangents if none are given
        if tangents is None:
            tangents = self.create_tangents(num_directions)

        # compute and set the forward gradients
        forward_gradients = self.compute_forward_gradients(tangents)
        self.overwrite_gradients(forward_gradients)

    @abc.abstractmethod
    def compute_forward_gradients(self, tangents):
        pass

    @abc.abstractmethod
    def overwrite_gradients(self, forward_gradients):
        pass

    @staticmethod
    def gradient_to_jvps(gradient, tangents):
        """Compute jacobi vector products VT∇f for k tangents VT (shape k x ...) and a gradient ∇f."""
        flat_tangent = tangents.view(tangents.shape[0], -1)  # VT (k x n)
        flat_gradient = gradient.view(flat_tangent.shape[1], -1)  # ∇f (n x batch-size)
        return torch.matmul(flat_tangent, flat_gradient)  # jvps VT∇f

    @classmethod
    def local_jvps(cls, gradients, tangents):
        """Compute jvps across multiple partial gradients and return as list."""
        return [cls.gradient_to_jvps(gradient, tangent) for gradient, tangent in zip(gradients, tangents)]

    @classmethod
    def global_jvps(cls, gradients, tangents, agg='sum'):
        """
        Compute jvps across multiple partial gradients and their tangents (e.g. multiple layers in a network), then
        aggregate them to global jvps using the given aggregation method.

        Using sum to aggregate the partial jvps corresponds to the jvp over the entire gradient using the concatenated
        tangents (with zero tangents for parameters not represented in the given gradient-tangent pairs) since
        the matrix multiplication (A1 | A2 | A3 | ... ) * (B1 | B2 | B3 | ... )T = ∑i Ai * BiT.
        """
        local_jvps = cls.local_jvps(gradients, tangents)
        aggregations = {
            'sum': sum
        }
        if agg not in aggregations:
            raise ValueError(f'Invalid aggregation method {agg}. Valid methods are {list(aggregations.keys())}.')

        return aggregations[agg](local_jvps)

    def gradients_to_forward_gradients(self, true_gradients, tangents):
        """
        Compute forward gradients from a list of true gradients and tangents.
        The true gradients are a list of partial gradients, e.g. multiple layers in a network. The tangents are a list
        of the same length of k corresponding tangents per partial gradient as tensor of shape k x ...
        Returns a list of forward gradients using either global or local jvps.
        """
        if self._local_jvps:
            jvps = self.local_jvps(true_gradients, tangents)
            return [self._compute_forward_gradient(partial_tangents, local_jvps)
                    for partial_tangents, local_jvps in zip(tangents, jvps)]
        else:  # global jvps
            # compute global jvps
            jvps = self.global_jvps(true_gradients, tangents, self.global_jvp_agg)

            # concat flat partial tangents to global tangents
            k = len(jvps)
            flat_partial_tangents = [partial_tangents.view(k, -1) for partial_tangents in tangents]
            global_tangents = torch.cat(flat_partial_tangents, dim=1)

            # compute FG on global tangents (especially important for frog to orthogonalize across the entire network)
            global_forward_gradient = self._compute_forward_gradient(global_tangents, jvps)

            # undo the tangent concatenation and split back into partial gradients of the correct shape
            start_indices = torch.tensor([0] + [t.shape[1] for t in flat_partial_tangents]).cumsum(0).tolist()
            indices = [(start, end) for start, end in zip(start_indices, start_indices[1:])]
            return [global_forward_gradient[start:end].view_as(true_grad)
                    for (start, end), true_grad in zip(indices, true_gradients)]


class WeightPerturbedSimFGModel(SimulatedForwardGradientModel):
    def __init__(self, model, **kwargs):
        super(WeightPerturbedSimFGModel, self).__init__(model, **kwargs)

    def create_tangents(self, num_directions):
        # note: as of python 3.7, dict preserves the insertion order, i.e., the order of tangents (via
        # param_names_and_shapes.values()) should remain consistent with param_names_and_shapes.keys().
        param_names_and_shapes = {name: param.shape
                                  for name, param in self.model.named_parameters() if param.requires_grad}
        tangents = self._create_tangents(num_directions, param_names_and_shapes.values())
        return dict(zip(param_names_and_shapes.keys(), tangents))

    def compute_forward_gradients(self, tangents):
        """
        Compute forward gradients for all parameters given their tangents.
        Return as dict parameter name -> forward gradient
        """
        true_gradients = {name: param.grad for name, param in self.model.named_parameters() if name in tangents}
        # make sure all gradients exist
        for name, gradient in true_gradients.items():
            if gradient is None:
                raise RuntimeError(f'Trying to compute forward gradient for parameter {name} but .grad is None. '
                                   'Did you run backward()?')

        param_order = list(tangents.keys())
        forward_gradients = self.gradients_to_forward_gradients([true_gradients[param] for param in param_order],
                                                                [tangents[param] for param in param_order])
        return {param: forward_gradient.view(true_gradients[param].shape)
                for param, forward_gradient in zip(param_order, forward_gradients)}

    def overwrite_gradients(self, forward_gradients, set_remainder_to_zero=False):
        for name, param in self.model.named_parameters():
            if name in forward_gradients:
                param.grad = forward_gradients[name]
            elif set_remainder_to_zero and param.grad is not None:
                param.grad.zero_()


class ActivityPerturbationLayer(nn.Module):
    def __init__(self, inner_module):
        super(ActivityPerturbationLayer, self).__init__()
        self.inner_module = inner_module
        self.__requires_grad = None
        self.input = None
        self.activity = None

    def summarize(self, with_params=False):
        summarize_layer = str if with_params else lambda layer: layer._get_name()
        try:
            return '-'.join([summarize_layer(layer) for layer in self.inner_module])
        except TypeError:
            return summarize_layer(self.inner_module)

    @property
    def activity_size(self):
        return None if self.activity is None else np.prod(self.activity.shape[1:])

    @property
    def weight_size(self):
        return sum([np.prod(param.shape) for param in self.parameters() if param.requires_grad])

    @property
    def requires_grad(self):
        # this layer requires grad if any of its parameters requires grad
        # the value is cached for later checks and should be changed via the corresponding setter
        if self.__requires_grad is None:
            self.__requires_grad = any(param.requires_grad for param in self.parameters())
        return self.__requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad=True):
        # set requires_grad for this layer and all it's contained parameters.
        # Same as requires_grad_ except that is also sets the requires_grad property.
        self.__requires_grad = requires_grad
        self.requires_grad_(requires_grad)

    def forward(self, x):
        # standard forward pass on the inner module
        # save activity to generate tangents of the appropriate shape and access the true activity gradient
        self.activity = self.inner_module(x)
        if torch.is_grad_enabled() and self.requires_grad:
            # computing the FG of the activity via simulation requires the actual gradient of the activity
            # since it is a non-leaf tensor, this would normally not be retained until after the backward pass
            self.activity.retain_grad()
            # save layer input to compute FG-based parameter gradients with layer-local backward pass
            self.input = x.detach()
        return self.activity

    def compute_gradients_from_activity_gradient(self, activity_gradient):
        # reset gradients to replace them with forward gradients
        for param in self.parameters():
            param.grad = None

        # redo forward pass on detached input (to limit the backward pass to this layer only)
        activity = self.inner_module(self.input)

        # redo backward pass within this layer seeded with the given activity_gradient
        # this automatically sets the gradients of this layer's parameters
        activity.backward(activity_gradient)


class ActivityPerturbedSimFGModel(SimulatedForwardGradientModel):
    def __init__(self, model, **kwargs):
        layers = model.layers if hasattr(model, 'layers') else model.children()
        self.blocks = [ActivityPerturbationLayer(layer) for layer in layers]
        wrapped_model = nn.Sequential(*self.blocks)
        wrapped_model.model_config = model.model_config
        super(ActivityPerturbedSimFGModel, self).__init__(wrapped_model, **kwargs)

    def create_tangents(self, num_directions):
        # Note: in contrast to weight perturbation, this depends on the shape of the input x and can thus only be
        # performed after the forward pass. If the shape does not change throughout training, it could also be moved
        # outside the training loop but would still require one prior forward pass to initialize the shapes.
        try:
            block_shapes = [block.activity.shape[1:] for block in self.blocks]
            return self._create_tangents(num_directions, block_shapes)
        except AttributeError as e:
            raise RuntimeError('Cannot create tangents without prior forward pass to initialize activity shapes. '
                               f'Causes: {e}.')

    def summarize(self, with_layer_params=False):
        block_summaries = [block.summarize(with_layer_params) for block in self.blocks]
        name_width = max(50, *[len(summary) for summary in block_summaries]) + 5

        def format_block_stats(summary, weights, activity):
            factor = prefixed.Float(weights / activity)
            weights, activity = prefixed.Float(weights), prefixed.Float(activity)
            return (f'{summary:<{name_width}}'
                    f'    #weights: {weights:6.1h}'
                    f'    #activations: {activity:6.1h}'
                    f'    weight/activation factor: {factor:6.1e}')
        block_lines = [
            format_block_stats('  ' + block.summarize(with_layer_params), block.weight_size, block.activity_size)
            for block, summary in zip(self.blocks, block_summaries)]
        total_weight_size = sum([block.weight_size for block in self.blocks])
        total_activity_size = sum([block.activity_size for block in self.blocks])
        total_sizes = format_block_stats('Total', total_weight_size, total_activity_size)
        return '\n'.join([self.__class__.__name__] + block_lines + ['-' * len(total_sizes), total_sizes])

    def compute_forward_gradients(self, tangents):
        """
        Compute forward gradients for all block activations given their tangents.
        Return as list of forward gradients (one entry per block, ordered sequentially)
        """
        true_gradients = [block.activity.grad for block in self.blocks]
        # make sure all gradients exist
        if any([grad is None for grad in true_gradients]):
            raise RuntimeError(f'Trying to compute forward gradient but some activity.grad is None. '
                               'Did you run backward()?')

        true_gradients = [grad.movedim(0, -1) for grad in true_gradients]  # move batch dimension from 0 to last
        forward_gradients = self.gradients_to_forward_gradients(true_gradients, tangents)
        return [grad.movedim(-1, 0) for grad in forward_gradients]  # move batch dimension back to 0 from last

    def overwrite_gradients(self, forward_gradients, set_remainder_to_zero=False):
        if set_remainder_to_zero:
            # set gradients of all parameters to zero, those with forward gradients will be overwritten in the next step
            for name, param in self.model.named_parameters().items():
                if name not in forward_gradients and param.grad is not None:
                    param.grad.zero_()

        # compute weight gradients with local backward steps
        for block, activity_gradient in zip(self.blocks, forward_gradients):
            block.compute_gradients_from_activity_gradient(activity_gradient)

    def update_activity_shapes(self, input_shape):
        self(torch.randn(input_shape, device=self.device))
