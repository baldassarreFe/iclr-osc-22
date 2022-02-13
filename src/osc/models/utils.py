"""
Misc modules.
"""

from typing import Tuple

import einops.layers.torch
from torch import nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        activation=nn.GELU,
        hidden_bias=True,
        out_bias=True,
        dropout=0.0,
    ):
        super().__init__()
        if hidden_features is None:
            hidden_features = in_features
        if out_features is None:
            out_features = in_features
        if not isinstance(dropout, Tuple):
            dropout = (dropout, dropout)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=hidden_bias)
        self.act = activation()
        self.drop1 = nn.Dropout(dropout[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=out_bias)
        self.drop2 = nn.Dropout(dropout[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


global_avg_pool = einops.layers.torch.Reduce("B N C -> B C", "mean")
global_max_pool = einops.layers.torch.Reduce("B N C -> B C", "max")
