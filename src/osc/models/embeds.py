"""
Positional embeddings and query tokens for objects.
"""
from typing import Tuple

import numpy as np
import timm.models
import torch
from torch import nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim, dropout=0.0):
        super(PositionalEmbedding, self).__init__()
        self.embed = nn.Parameter(torch.zeros((num_embeddings, embed_dim)))
        self.drop = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        timm.models.layers.trunc_normal_(self.embed, std=0.02)

    def forward(self, x):
        return self.drop(x + self.embed)

    def extra_repr(self) -> str:
        return ", ".join(
            [
                str(tuple(self.embed.shape[:-1])),
                f"dim={self.embed.shape[-1]}",
                f"drop={self.drop.p}",
            ]
        )


class PositionalEmbeddingNd(nn.Module):
    def __init__(self, embed_shape: Tuple[int, ...], embed_dim: int, dropout=0.0):
        super(PositionalEmbeddingNd, self).__init__()
        self.embed_shape = tuple(embed_shape)
        self.embed_dim = embed_dim
        self.embed = nn.Parameter(torch.zeros((*embed_shape, embed_dim)))
        self.dropout = dropout
        self.init_weights()

    def init_weights(self):
        timm.models.layers.trunc_normal_(self.embed, std=0.02)

    def forward(self, x):
        embed = self.embed
        # If x is flattened, flatten the embedding too
        if x.shape[-self.embed.dim() : -1] != self.embed_shape:
            embed = self.embed.reshape(-1, self.embed_dim)
        return torch.dropout(x + embed, self.dropout, self.training)

    def extra_repr(self) -> str:
        return ", ".join(
            [
                str(self.embed_shape),
                f"dim={self.embed.shape[-1]}",
                f"dropout={self.dropout}",
            ]
        )


def create_positional_embedding(shape: Tuple[int, ...], dim: int, flatten: bool):
    if flatten:
        shape = (np.prod(shape),)
    embed = torch.zeros((*shape, dim))
    timm.models.layers.trunc_normal_(embed, std=0.02)
    return embed
