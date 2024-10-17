import abc
import math

import numpy as np
import torch
from torch import nn
import torch.autograd.forward_ad as fwAD

import forward_gradient


class ForwardGradientBaseModel(nn.Module, abc.ABC):
    def __init__(self, model, aggregation_mode='mean', tangent_sampler='normal', aggregation_kwargs=None,
                 generator=None, seed=None, tangent_postprocessing=None, normalize_tangents=False,
                 tangent_sampling_kwargs=None, global_tangent=True, scaling_correction=False, **kwargs):
        super(ForwardGradientBaseModel, self).__init__()
        self.model = model

        self.aggregation_mode = aggregation_mode
        self.aggregation_kwargs = {} if aggregation_kwargs is None else aggregation_kwargs

        self.model_config = {
            'fg_model__cls': self.__class__, 'fg_model__aggregation_mode': self.aggregation_mode,
            'fg_model__tangent_sampler': tangent_sampler, 'fg_model__aggregation_kwargs': str(self.aggregation_kwargs),
            'fg_model__tangent_postprocessing': tangent_postprocessing,
            'fg_model__normalize_tangents': normalize_tangents,
            'fg_model__tangent_sampling_kwargs': tangent_sampling_kwargs, 'fg_model__global_tangent': global_tangent,
            'fg_model__scaling_correction': scaling_correction,
            **self.model.model_config}

        if scaling_correction:
            expected_tangent_length_fns = {'normal': lambda n: math.sqrt(n)}
            expected_tangent_length_fn = (lambda n: 1) if normalize_tangents \
                else expected_tangent_length_fns[tangent_sampler]
            self.aggregation_kwargs = {**self.aggregation_kwargs, 'scaling_correction': True,
                                       'expected_tangent_length_fn': expected_tangent_length_fn}

        self.tangents = {}

        self.seed = seed
        self.generators_by_device = {}
        if generator is not None:
            self.generators_by_device[str(generator.device)] = generator

        if normalize_tangents:
            if tangent_postprocessing is None:
                tangent_postprocessing = [forward_gradient.normalize]
            else:
                print(f'Warning: both {tangent_postprocessing=} and {normalize_tangents=} passed to {self.__class__},'
                      'will append normalization to existing postprocessing steps. Make sure this is intentional.')
                tangent_postprocessing.append(forward_gradient.normalize)

        self.tangent_sampler = lambda shape: forward_gradient.sample_tangents(
            shape, sampler=tangent_sampler, postprocessing=tangent_postprocessing, generator=self.generator,
            device=self.device, **({} if tangent_sampling_kwargs is None else tangent_sampling_kwargs))
        self.global_tangent = global_tangent

    def summarize(self, with_layer_params=False):
        return self.__class__.__name__

    def add_generator_for_device(self, device_name):
        generator = torch.Generator(device=device_name)
        generator.manual_seed(self.seed)
        self.generators_by_device[device_name] = generator
        return generator

    @property
    def generator(self):
        device_name = str(self.device)
        if device_name not in self.generators_by_device:
            self.add_generator_for_device(device_name)
        return self.generators_by_device[device_name]

    def forward(self, x):
        return self.model(x)

    @property
    def device(self):
        return next(self.parameters()).device

    def _create_global_tangents(self, num_directions, partial_tangent_shapes):
        flat_partial_shapes = [np.prod(shape) for shape in partial_tangent_shapes]
        global_tangent = self.tangent_sampler((num_directions, sum(flat_partial_shapes)))

        start_indices = torch.tensor([0, *flat_partial_shapes]).cumsum(0).tolist()
        return [global_tangent[:, start:end].view([num_directions, *shape])
                for start, end, shape in zip(start_indices, start_indices[1:], partial_tangent_shapes)]

    def _create_local_tangents(self, num_directions, partial_tangent_shapes):
        return [self.tangent_sampler((num_directions, *shape)) for shape in partial_tangent_shapes]

    def _create_tangents(self, num_directions, partial_tangent_shapes):
        if self.global_tangent:
            return self._create_global_tangents(num_directions, partial_tangent_shapes)
        else:
            return self._create_local_tangents(num_directions, partial_tangent_shapes)

    @abc.abstractmethod
    def create_tangents(self, num_directions):
        pass

    def _compute_forward_gradient(self, tangents, jvps):
        return forward_gradient.forward_gradient(tangents, jvps, self.aggregation_mode,
                                                 **self.aggregation_kwargs)


class ForwardGradientModel(ForwardGradientBaseModel, abc.ABC):
    def __init__(self, model, **kwargs):
        super(ForwardGradientModel, self).__init__(model, **kwargs)

    def sync_weights(self):
        pass

    @abc.abstractmethod
    def set_tangent(self, tangents, tangent_index=0):
        pass

    @abc.abstractmethod
    def compute_gradients(self, jvps, tangents):
        pass

    def set_gradients(self, forward_gradients):
        for name, gradient in forward_gradients.items():
            submodule_name, parameter_name = name.rsplit('.', 1)
            submodule = self.get_submodule(submodule_name)
            getattr(submodule, parameter_name).grad = gradient

    def compute_forward_gradients(self, jvps, tangents):
        # TODO: orthogonalize across the entire network (not just the partial tangents)
        return {name: self._compute_forward_gradient(partial_tangents, jvps)
                for name, partial_tangents in tangents.items()}


class WeightPerturbedForwardGradientModel(ForwardGradientModel):
    def __init__(self, model, **kwargs):
        super(WeightPerturbedForwardGradientModel, self).__init__(model, **kwargs)
        # replace the parameters in the model with buffers, so they can be converted to dual tensors during training
        # store both the original parameters and the buffers in a dict for easy access by parameter name
        self.actual_parameters = self.convert_parameters_to_tensors()
        self.model.requires_grad_(False)

    def sync_weights(self):
        for name, param in self.actual_parameters.items():
            submodule_name, parameter_name = name.rsplit('.', 1)
            submodule = self.get_submodule(submodule_name)
            setattr(submodule, parameter_name, param)

    def _apply(self, fn, recurse=True):
        super(WeightPerturbedForwardGradientModel, self)._apply(fn, recurse)

        for key, param in self.actual_parameters.items():
            self.actual_parameters[key] = fn(param)
        self.sync_weights()
        return self

    def convert_parameters_to_tensors(self):
        named_parameters = dict(self.named_parameters())
        actual_parameters = {}
        for name, param in named_parameters.items():
            actual_parameters[name] = param.detach()  # store detached tensor, should reuse the same storage
            submodule_name, parameter_name = name.rsplit('.', 1)
            submodule = self.get_submodule(submodule_name)
            # delete parameter in submodule (otherwise we cannot overwrite it with a (dual) tensor)
            # should not affect the detached version
            delattr(submodule, parameter_name)
            setattr(submodule, parameter_name, actual_parameters[name])  # add the parameter again as tensor
        return actual_parameters

    def parameters(self, recurse=True):
        return iter(self.actual_parameters.values())

    def create_tangents(self, num_directions):
        # note: as of python 3.7, dict preserves the insertion order, i.e., the order of tangents (via
        # param_names_and_shapes.values()) should remain consistent with param_names_and_shapes.keys().
        param_names_and_shapes = {name: param.shape for name, param in self.actual_parameters.items()}
        tangents = self._create_tangents(num_directions, param_names_and_shapes.values())
        return dict(zip(param_names_and_shapes.keys(), tangents))

    def set_tangent(self, tangent, index=None):
        for name, param_tangent in tangent.items():
            param_tangent = param_tangent if index is None else param_tangent[index]

            submodule_name, parameter_name = name.rsplit('.', 1)
            submodule = self.get_submodule(submodule_name)
            setattr(submodule, parameter_name, fwAD.make_dual(self.actual_parameters[name], param_tangent))

    def set_gradients(self, forward_gradients):
        for name, gradient in forward_gradients.items():
            self.actual_parameters[name].grad = gradient

    def compute_gradients(self, jvps, tangents):
        forward_gradients = self.compute_forward_gradients(jvps, tangents)
        self.set_gradients(forward_gradients)


class ActivityPerturbationLayer(nn.Module):
    def __init__(self, inner_module):
        super(ActivityPerturbationLayer, self).__init__()
        self.inner_module = inner_module
        self.tangent = None
        self.activity = None

    def forward(self, x):
        if fwAD._current_level >= 0:
            self.activity = self.inner_module(x.detach())
            return fwAD.make_dual(self.activity, self.tangent.expand(*self.activity.shape))
        else:
            return self.inner_module(x)

    def compute_gradients_from_activity_gradient(self, activity_gradient):
        self.activity.backward(activity_gradient.expand(*self.activity.shape))


class ActivityPerturbedForwardGradientModel(ForwardGradientModel):
    def __init__(self, model, **kwargs):
        self.is_wrapped = [bool(list(layer.parameters())) for layer in model.layers]
        self.wrapped_layer_ids = [i for i, is_wrapped in enumerate(self.is_wrapped) if is_wrapped]
        wrapped_model = nn.Sequential(*[ActivityPerturbationLayer(layer) if wrap else layer
                                        for layer, wrap in zip(model.layers, self.is_wrapped)])
        wrapped_model.model_config = model.model_config

        super(ActivityPerturbedForwardGradientModel, self).__init__(wrapped_model, **kwargs)
        self.tangent_shapes = {layer_id: None for layer_id in self.wrapped_layer_ids}

    def update_activity_shapes(self, input_shape):
        x = torch.randn(input_shape)
        for layer_id, layer in enumerate(self.model):
            x = layer(x)
            if self.is_wrapped[layer_id]:
                self.tangent_shapes[layer_id] = x[0].shape

    def create_tangents(self, num_directions):
        # note: as of python 3.7, dict preserves the insertion order, i.e., the order of tangents (via
        # self.tangent_shapes.values()) should remain consistent with self.tangent_shapes.keys().
        tangents = self._create_tangents(num_directions, self.tangent_shapes.values())
        return dict(zip(self.tangent_shapes.keys(), tangents))

    def set_tangent(self, tangents, tangent_index=0):
        for layer_id, tangent in tangents.items():
            tangent = tangent if tangent_index is None else tangent[tangent_index]
            self.model[layer_id].tangent = tangent

    def compute_gradients(self, jvps, tangents):
        # TODO: orthogonalize across the entire network (not just the partial tangents)
        forward_gradients = self.compute_forward_gradients(jvps, tangents)
        for layer_id, activity_forward_gradient in forward_gradients.items():
            self.model[layer_id].compute_gradients_from_activity_gradient(activity_forward_gradient)
