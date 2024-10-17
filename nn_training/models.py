import math
import re

import torch
from torch import nn
import torchvision.models

import nn_training.fg_models
import nn_training.simfg_models
import nn_training.vision_transformer


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.model_config = {'model__name': 'lenet5'}
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(256, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_classes)
        self.layers = nn.Sequential(self.conv1, self.conv2, self.fc1, self.fc2, self.fc3)

    def forward(self, x):
        return self.layers(x)


class FCNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers, activation=nn.ReLU):
        super(FCNet, self).__init__()
        self.model_config = {'model__name': 'fc', 'model__width': hidden_size, 'model__depth': num_hidden_layers,
                             'model__activation': str(activation)}
        if num_hidden_layers < 1:
            self.layers = [nn.Sequential(nn.Flatten(), nn.Linear(input_size, output_size))]
        else:
            self.layers = [[nn.Flatten(), nn.Linear(input_size, hidden_size), activation()]]
            self.layers += [[nn.Linear(hidden_size, hidden_size), activation()] for _ in range(num_hidden_layers - 1)]
            self.layers += [[nn.Linear(hidden_size, output_size)]]
            self.layers = [nn.Sequential(*layer) for layer in self.layers]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class ConvNet(nn.Module):
    def __init__(self, input_shape, config_string, layer_sep='-', block_sep='--', activation=nn.ReLU):
        super(ConvNet, self).__init__()
        self.model_config = {'model__name': 'convnet', 'model__config_string': config_string,
                             'model__activation': str(activation)}
        self.activation = activation
        self.layers = self.from_string(input_shape, config_string, layer_sep, block_sep)

    def parse_layer(self, layer_description, input_shape):
        # Supported layers and parameters:
        # - FC<out_features>
        fc_pattern = r'FC(?P<out_features>\d+)'
        # - Conv<out_channels>,<kernel_size>(,<stride=1>,<padding=0>) (last two are optional)
        conv_pattern = (r'Conv(?P<out_channels>\d+),(?P<kernel_size>\d+)'
                        r'(,(?P<stride>\d+))?(,(?P<padding>\d+))?')
        # - Pool<kernel_size>,<stride>(,padding=0) (padding is optional)
        max_pool_pattern = r'Pool(?P<kernel_size>\d+),(?P<stride>\d+)(,(?P<padding>\d+))?'
        # - BN
        norm_pattern = r'BN'
        # - Activation functions: ReLU, act (uses self.activation)
        activation_pattern = r'(act|ReLU)'
        # - 'Flatten' or 'F'
        flatten_pattern = r'(Flatten|F)'

        layer_kwargs = {}
        if match := re.match(fc_pattern, layer_description):
            layer_type = nn.Linear
            layer_kwargs = {'in_features': input_shape[-1], **match.groupdict()}
        elif match := re.match(conv_pattern, layer_description):
            layer_type = nn.Conv2d
            layer_kwargs = {'in_channels': input_shape[1], **match.groupdict()}
        elif match := re.match(max_pool_pattern, layer_description):
            layer_type = nn.MaxPool2d
            layer_kwargs = match.groupdict()
        elif match := re.match(norm_pattern, layer_description):
            layer_type = nn.BatchNorm2d
            layer_kwargs = {'num_features': input_shape[1], **match.groupdict()}
        elif re.match(activation_pattern, layer_description):
            if layer_description == 'ReLU':
                layer_type = nn.ReLU
            elif layer_description == 'act':
                layer_type = self.activation
        elif re.match(flatten_pattern, layer_description):
            layer_type = nn.Flatten
        else:
            raise ValueError(f'Cannot parse layer {layer_description}.')
        return layer_type, {key: int(value) for key, value in layer_kwargs.items() if value is not None}

    @staticmethod
    def get_next_shape(layer_type, layer_kwargs, input_shape):
        previous_rng_state = torch.get_rng_state()
        layer = layer_type(**layer_kwargs)
        output = layer(torch.randn(input_shape))
        torch.set_rng_state(previous_rng_state)
        return output.shape

    def from_string(self, input_shape, config_string, layer_sep='-', block_sep='--'):
        # layers are separated by layer_sep
        # blocks are separated by block_sep (optional, to group sequential layers into an additional nn.Sequential)
        blocks = []
        for block_description in config_string.split(block_sep):
            layers_in_block = []
            for layer_description in block_description.split(layer_sep):
                layer_type, layer_kwargs = self.parse_layer(layer_description, input_shape)
                layers_in_block.append(layer_type(**layer_kwargs))
                input_shape = self.get_next_shape(layer_type, layer_kwargs, input_shape)
            blocks.append(nn.Sequential(*layers_in_block))
        return nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)


class ResNet18(nn.Module):
    def __init__(self, input_shape, num_classes, grouping_mode='wide'):
        super(ResNet18, self).__init__()
        self.model_config = {'model__name': 'resnet18', 'model__grouping_mode': grouping_mode}
        resnet18 = torchvision.models.resnet18(num_classes=num_classes)
        channels = input_shape[0]
        if channels != 3:  # if the number of input channels does not match the default 3 replace the first layer
            resnet18.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.grouping_mode = grouping_mode
        if grouping_mode == 'wide':
            self.layers = nn.Sequential(
                nn.Sequential(resnet18.conv1, resnet18.bn1, resnet18.relu, resnet18.maxpool),
                resnet18.layer1,
                resnet18.layer2,
                resnet18.layer3,
                resnet18.layer4,
                nn.Sequential(resnet18.avgpool, nn.Flatten(), resnet18.fc)
            )
        elif grouping_mode == 'fine':
            self.layers = nn.Sequential(
                resnet18.conv1,
                resnet18.bn1,
                resnet18.relu,
                resnet18.maxpool,
                *resnet18.layer1,
                *resnet18.layer2,
                *resnet18.layer3,
                *resnet18.layer4,
                resnet18.avgpool,
                nn.Flatten(),
                resnet18.fc
            )
        else:
            raise ValueError(f'Invalid {grouping_mode=}')

    def forward(self, x):
        return self.layers(x)


def create_resnet18(input_shape, num_classes):
    resnet18 = torchvision.models.resnet18(num_classes=num_classes)
    channels = input_shape[0]
    if channels != 3:  # if the number of input channels does not match the default 3 replace the first layer
        resnet18.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return resnet18


class ViTBase(nn_training.vision_transformer.VisionTransformer):
    def __init__(self, input_shape, num_classes, grouping_mode='block_wise', patch_size=None, num_layers=12,
                 num_heads=12, hidden_dim=768, mlp_dim=3072, **kwargs):
        self.model_config = {
            'model__name': 'vit', 'model__grouping_mode': grouping_mode, 'model__patch_size': patch_size,
            'model__num_layers': num_layers, 'model__num_heads': num_heads, 'model__hidden_dim': hidden_dim,
            'model__mlp_dim': mlp_dim, **kwargs}
        # use torchvision vit_b_16 as baseline
        in_channels = input_shape[0]
        image_size = input_shape[1]
        if patch_size is None:
            patch_size = 4 if image_size < 64 else 16
        super().__init__(num_classes=num_classes, image_size=image_size, patch_size=patch_size, num_layers=num_layers,
                         num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim, in_channels=in_channels,
                         grouping_mode=grouping_mode, **kwargs)


def get_model(config):
    model_type = config.model.lower()
    model_config = config.get_config('model')

    dataset_config = config.get_config('dataset')
    input_shape = torch.Size(map(int, dataset_config.get('input_shape').split('x')))
    num_classes = dataset_config.getint('num_classes')

    if model_type == 'fc':
        model = FCNet(math.prod(input_shape), num_classes, model_config.getint('hidden_size'),
                      model_config.getint('num_hidden_layers'))
    elif model_type == 'lenet5':
        model = LeNet5(num_classes)
    elif model_type == 'resnet18':
        model = ResNet18(input_shape, num_classes)
    elif model_type == 'vit':
        vit_config = config.get_multiple_from_config(
            model_config, [['num_layers', int], ['num_heads', int], ['hidden_dim', int], ['mlp_dim', int],
                           ['replicate_torchvision', bool, False], ['grouping_mode']])
        model = ViTBase(input_shape, num_classes, **vit_config)
    elif model_type == 'mlpmixer':
        default_patch_size = 4 if input_shape[1] < 64 else 16
        mlpmixer_config = config.get_multiple_from_config(
            model_config, [['hidden_dim', int], ['token_mlp_dim', int], ['channel_mlp_dim', int], ['num_blocks', int],
                           ['patch_size', int, default_patch_size], ['dropout', float], ['grouping_mode']])
        model = nn_training.mlp_mixer.MLPMixer(input_shape, num_classes=num_classes, **mlpmixer_config)
    else:
        raise ValueError(f'Model {model_type} is currently not supported.')

    if config.gradient_computation in ['fg', 'frog']:
        fg_config = config.get_config('gradient_computation')

        fg_computation_mode = fg_config.get('fg_computation_mode')
        bool_config_keys = ['normalize_tangents', 'global_tangent', 'scaling_correction']
        str_config_keys = ['aggregation_mode', 'tangent_sampler']
        model_kwargs = {'model': model, 'seed': fg_config.getint('seed') + 1,
                        **{key: fg_config.getboolean(key) for key in bool_config_keys if key in fg_config},
                        **{key: fg_config.get(key) for key in str_config_keys if key in fg_config}}

        fg_model = {
            ('fwad', 'weight'): nn_training.fg_models.WeightPerturbedForwardGradientModel,
            ('fwad', 'node'): nn_training.fg_models.ActivityPerturbedForwardGradientModel,
            ('sim', 'weight'): nn_training.simfg_models.WeightPerturbedSimFGModel,
            ('sim', 'node'): nn_training.simfg_models.ActivityPerturbedSimFGModel,
        }
        fg_mode = (fg_computation_mode, config.perturbation_mode)
        print(fg_mode)
        
        if fg_mode not in fg_model:
            raise ValueError(f'Unknown FG mode {fg_mode}. Valid modes are {list(fg_model.keys())}.')
        model = fg_model[fg_mode](**model_kwargs)

    return model
