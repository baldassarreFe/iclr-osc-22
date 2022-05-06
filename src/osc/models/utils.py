"""
Misc modules.
"""

from typing import Callable, Generator, List, Tuple, Union

import einops.layers.torch
import torch
from torch import Tensor
from torch import nn as nn

from osc.utils import normalize_sum_to_one


class MLP(nn.Module):
    _ACTIVATIONS = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
    }

    def __init__(
        self,
        in_features: int,
        hidden_mult: float = 1.0,
        out_features: int = None,
        activation: Union[str, Callable[[], nn.Module]] = "gelu",
        hidden_bias=True,
        out_bias=True,
        dropout=0.0,
    ):
        """2-layer MLP.

        Args:
            in_features:
            hidden_mult: multiplier for ``in_in_features``
            out_features: defaults to  ``in_features``
            activation:
            hidden_bias:
            out_bias:
            dropout:
        """
        super().__init__()
        if out_features is None:
            out_features = in_features
        if isinstance(activation, str):
            activation = self._ACTIVATIONS[activation]
        if not isinstance(dropout, Tuple):
            dropout = (dropout, dropout)

        hidden_features = round(hidden_mult * in_features)
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


class RegularSoftmax(nn.Module):
    """Regular softmax with dropout."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
        self.register_buffer("negative_inf", torch.tensor(float("-inf")))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0:
            keep = torch.empty_like(x).uniform_().gt(self.p)
            x = torch.where(keep, x, self.negative_inf)
        return self.softmax(x)


class CompetitiveSoftmax(nn.Module):
    """Competitive softmax with dropout."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
        self.register_buffer("negative_inf", torch.tensor(float("-inf")))
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0:
            keep = torch.empty_like(x).uniform_().gt(self.p)
            x = torch.where(keep, x, self.negative_inf)
        x = self.softmax(x)
        return normalize_sum_to_one(x, dim=-1)


def no_weight_decay(module: nn.Module) -> List[str]:
    return list(_recursive_no_weight_decay(module))


def _recursive_no_weight_decay(module: nn.Module) -> Generator[str, None, None]:
    if hasattr(module, "no_weight_decay"):
        yield from module.no_weight_decay()
    yield from (
        ".".join([name, nwd])
        for name, child in module.named_children()
        for nwd in _recursive_no_weight_decay(child)
    )


global_avg_pool = einops.layers.torch.Reduce("B N C -> B C", "mean")
global_max_pool = einops.layers.torch.Reduce("B N C -> B C", "max")
