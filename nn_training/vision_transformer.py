# This file has been adapted from torchvision.models.vision_transformer which
# has been released under the following license:
#
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import re
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models.vision_transformer import MLPBlock, ConvStemConfig


class PatchEmbedding(nn.Module):
    # extracted from torchvision.models.vision_transformer.VisionTransformer
    def __init__(self, image_size: int, patch_size: int, hidden_dim: int, in_channels: int = 3,
                 conv_stem_configs: Optional[List[ConvStemConfig]] = None, initialize=True):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = in_channels
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        self.seq_length = (image_size // patch_size) ** 2
        if initialize:
            self.initialize()

    def initialize(self):
        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

    def forward(self, x):
        # this is a copy of torchvision.models.vision_transformer.VisionTransformer._process_input
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x


class ClassToken(nn.Module):
    # extracted from torchvision.models.vision_transformer.VisionTransformer
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor):
        # Expand the class token to the full batch
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        # Prepend class token to patch sequence
        return torch.cat([batch_class_token, x], dim=1)


class PositionalEmbedding(nn.Module):
    # refactored from torchvision.models.vision_transformer.Encoder
    def __init__(self, seq_length: int, hidden_dim: int):
        super().__init__()
        # Note that batch_size is on the first dim because we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT

    def forward(self, input: torch.Tensor):
        return input + self.pos_embedding


class EncoderAttentionBlock(nn.Module):
    # extracted from torchvision.models.vision_transformer.EncoderBlock
    def __init__(self, num_heads: int, hidden_dim: int, dropout: float, attention_dropout: float,
                 norm_layer: Callable[..., torch.nn.Module]):
        super().__init__()
        self.num_heads = num_heads
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        return x + input


class EncoderMLPBlock(nn.Module):
    # extracted from torchvision.models.vision_transformer.EncoderBlock
    def __init__(self, hidden_dim: int, mlp_dim: int, dropout: float, norm_layer: Callable[..., torch.nn.Module]):
        super().__init__()
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        x = self.ln_2(input)
        x = self.mlp(x)
        return x + input


class EncoderBlock(nn.Module):
    # refactored from torchvision.models.vision_transformer.EncoderBlock
    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.num_heads = num_heads
        self.attention_block = EncoderAttentionBlock(num_heads, hidden_dim, dropout, attention_dropout, norm_layer)
        self.mlp_block = EncoderMLPBlock(hidden_dim, mlp_dim, dropout, norm_layer)
        self.layers = nn.Sequential(OrderedDict(attention_block=self.attention_block, mlp_block=self.mlp_block))

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        return self.layers(input)


class ViTHeads(nn.Module):
    # extracted from torchvision.models.vision_transformer.VisionTransformer
    def __init__(self, hidden_dim: int, num_classes: int = 1000, representation_size: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.representation_size = representation_size

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def forward(self, x: torch.Tensor):
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return self.heads(x)


class VisionTransformer(nn.Module):
    # refactored from torchvision.models.vision_transformer.VisionTransformer
    def __init__(self, image_size: int, patch_size: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int,
                 dropout: float = 0.0, attention_dropout: float = 0.0, num_classes: int = 1000,
                 in_channels: int = 3, representation_size: Optional[int] = None,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 conv_stem_configs: Optional[List[ConvStemConfig]] = None,
                 grouping_mode='block_wise', replicate_torchvision: bool = True):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        # Process input: patch embedding, add class token, positional embedding and dropout
        if replicate_torchvision:
            # replicate the reinitialization from torchvision ViT for alternative in channel dimensions
            self.patch_embedding = PatchEmbedding(image_size, patch_size, hidden_dim, 3, conv_stem_configs,
                                                  initialize=False)
        else:
            self.patch_embedding = PatchEmbedding(image_size, patch_size, hidden_dim, in_channels, conv_stem_configs)

        self.class_token = ClassToken(hidden_dim)
        self.seq_length = self.patch_embedding.seq_length + 1

        self.pos_embedding = PositionalEmbedding(self.seq_length, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Encoder consisting of num_layers encoder blocks
        self.encoder_layers = [EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
                               for _ in range(num_layers)]

        # Process output: final layer norm and heads
        self.layer_norm = norm_layer(hidden_dim)
        self.heads = ViTHeads(hidden_dim, num_classes, representation_size)
        self.process_output = nn.Sequential(self.layer_norm, self.heads)

        if replicate_torchvision:
            # replicate the reinitialization from torchvision ViT for alternative in channel dimensions
            self.patch_embedding.initialize()
            if in_channels != 3:
                self.patch_embedding = PatchEmbedding(image_size, patch_size, hidden_dim, in_channels,
                                                      conv_stem_configs)

        self.process_input = nn.Sequential(self.patch_embedding, self.class_token, self.pos_embedding, self.dropout)
        self.layers = nn.Sequential(self.process_input, *self.encoder_layers, self.process_output)

        if grouping_mode == 'block_wise':
            self.layers = nn.Sequential(self.process_input, *self.encoder_layers, self.process_output)
        elif match := re.match(r'wide_(?P<encoder_blocks_per_group>\d+)', grouping_mode):
            encoder_blocks_per_group = int(match.group('encoder_blocks_per_group'))
            grouped_encoder_blocks = [self.encoder_layers[i:i+encoder_blocks_per_group]
                                      for i in range(0, num_layers, encoder_blocks_per_group)]
            # prepend input processing to first group and append output processing to last group
            grouped_encoder_blocks[0] = [self.process_input] + grouped_encoder_blocks[0]
            grouped_encoder_blocks[-1] = grouped_encoder_blocks[-1] + [self.process_output]
            self.layers = nn.Sequential(*[nn.Sequential(*group) for group in grouped_encoder_blocks])
        elif grouping_mode == 'fine':
            raise NotImplementedError(f'{grouping_mode=} not yet implemented.')
        else:
            raise ValueError(f'Invalid {grouping_mode=}.')

    def forward(self, x: torch.Tensor):
        return self.layers(x)
