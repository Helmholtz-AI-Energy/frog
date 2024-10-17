import abc
import re

from torch import nn
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    def __init__(self, input_shape, patch_size, output_dimension):
        super().__init__()
        assert len(input_shape) == 3, f'Expected input_shape as channels x height x width but got {input_shape}.'
        self.channels, self.image_height, self.image_width = input_shape
        self.patch_size = patch_size
        self.output_dimension = output_dimension

        assert (self.image_height % self.patch_size) == 0 and (self.image_width % self.patch_size) == 0, \
            f'{self.image_height}x{self.image_width} image not divisible by patch size {self.patch_size}.'
        self.num_patches = (self.image_height // self.patch_size) * (self.image_width // self.patch_size)

        self.split_into_patches = Rearrange('batch c (h p1) (w p2) -> batch (h w) (p1 p2 c)',
                                            p1=self.patch_size, p2=self.patch_size)
        self.linear_projection = nn.Linear((self.patch_size ** 2) * self.channels, self.output_dimension)

    def forward(self, x):
        x = self.split_into_patches(x)
        x = self.linear_projection(x)
        return x


class MLPBlock(nn.Module, abc.ABC):
    def __init__(self, outer_dim, inner_dim, dropout=0., activation=nn.GELU):
        super().__init__()
        self.outer_dim = outer_dim
        self.inner_dim = inner_dim

        self.layers = nn.Sequential(
            self.fully_connected(self.outer_dim, self.inner_dim),
            activation(),
            nn.Dropout(dropout),
            self.fully_connected(self.inner_dim, self.outer_dim),
            nn.Dropout(dropout)
        )

    @staticmethod
    @abc.abstractmethod
    def fully_connected(in_features, out_features):
        pass

    def forward(self, x):
        return self.layers(x)


class TokenMixingViaConv(MLPBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def fully_connected(in_features, out_features):
        return nn.Conv1d(in_features, out_features, kernel_size=1)


class TokenMixingViaSwapAxes(MLPBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            # swap axes from batch size x num patches x hidden dimension to batch size x hidden dimension x num patches
            Rearrange('b p d -> b d p'),
            *self.layers,
            # swap axes back to batch size x num patches x hidden dimension
            Rearrange('b d p -> b p d')
        )

    @staticmethod
    def fully_connected(in_features, out_features):
        return nn.Linear(in_features, out_features)


class ChannelMixing(MLPBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def fully_connected(in_features, out_features):
        return nn.Linear(in_features, out_features)


class MixerLayer(nn.Module):
    def __init__(self, outer_dim, num_patches, token_mlp_dim, channel_mlp_dim, dropout=0., activation=nn.GELU):
        super().__init__()
        self.outer_dim = outer_dim

        self.norm1 = nn.LayerNorm(self.outer_dim)
        self.token_mixing = TokenMixingViaSwapAxes(num_patches, token_mlp_dim, dropout, activation)
        self.norm2 = nn.LayerNorm(self.outer_dim)
        self.channel_mixing = ChannelMixing(self.outer_dim, channel_mlp_dim, dropout, activation)

    def forward(self, x):
        x = self.token_mixing(self.norm1(x)) + x
        x = self.channel_mixing(self.norm2(x)) + x
        return x


class ClassifierHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        self.pre_head_layer_norm = nn.LayerNorm(self.dim)
        self.global_average_pooling = Reduce('b n c -> b c', 'mean')
        self.fc_classifier_head = nn.Linear(self.dim, num_classes) if self.num_classes else nn.Identity()
        self.layers = nn.Sequential(self.pre_head_layer_norm, self.global_average_pooling, self.fc_classifier_head)

    def forward(self, x):
        return self.layers(x)


class MLPMixer(nn.Module):
    def __init__(self, input_shape, patch_size, hidden_dim, token_mlp_dim, channel_mlp_dim, num_blocks, num_classes,
                 dropout=0., activation=nn.GELU, grouping_mode="wide"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_mlp_dim = token_mlp_dim
        self.channel_mlp_dim = channel_mlp_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        self.model_config = {
            'model__name': 'mlpmixer', 'model__grouping_mode': grouping_mode, 'model__patch_size': patch_size,
            'model__hidden_dim': hidden_dim, 'model__token_mlp_dim': token_mlp_dim,
            'model__channel_mlp_dim': channel_mlp_dim, 'model__num_blocks': num_blocks, 'model__dropout': dropout,
            'model__activation': activation}

        self.patch_embedding = PatchEmbedding(input_shape, patch_size, self.hidden_dim)
        self.mixer_layers = [MixerLayer(self.hidden_dim, self.patch_embedding.num_patches, token_mlp_dim,
                                        channel_mlp_dim, dropout, activation)
                             for _ in range(self.num_blocks)]
        self.classifier_head = ClassifierHead(self.hidden_dim, self.num_classes)

        if grouping_mode == 'block_wise':
            self.layers = nn.Sequential(self.patch_embedding, *self.mixer_layers, self.classifier_head)
        elif match := re.match(r'wide_(?P<layers_per_group>\d+)', grouping_mode):
            layers_per_group = int(match.group('layers_per_group'))
            grouped_layers = [self.mixer_layers[i:i+layers_per_group] for i in range(0, num_blocks, layers_per_group)]
            # prepend embedding to first group and append head to last group
            grouped_layers[0] = [self.patch_embedding] + grouped_layers[0]
            grouped_layers[-1] = grouped_layers[-1] + [self.classifier_head]
            self.layers = nn.Sequential(*[nn.Sequential(*group) for group in grouped_layers])
        elif grouping_mode == 'fine':
            raise NotImplementedError(f'{grouping_mode=} not yet implemented.')
        else:
            raise ValueError(f'Invalid {grouping_mode=}.')

    def forward(self, x):
        return self.layers(x)
