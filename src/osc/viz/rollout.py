from typing import Callable, Mapping, Sequence, Union

import torch
from einops import reduce
from einops.layers.torch import Reduce
from torch import Tensor

from osc.utils import fill_diagonal_, normalize_sum_to_one


def self_attn_rollout(
    attns: Union[Mapping[str, Tensor], Sequence[Tensor]],
    head_reduction: Union[str, Callable] = "mean",
    adjust_residual=True,
    global_avg_pool=True,
) -> Tensor:
    """Self-attn rollout: how much output token(s) attend to input tokens across layers

    Args:
        attns: dict or list where each entry has shape [B heads Q K]
        head_reduction: 'mean', 'max', or a callable that reduces the head dimension
        adjust_residual: bool, whether to add 0.5 for the self connection
        global_avg_pool: bool, if the output of the final attention layer is avg-pooled
                         into a single vector of features

    Returns:
        Rollout, shape [B Q K] if ``global_avg_pool=False`` else [B K]
    """
    if isinstance(attns, Mapping):
        attns = [attns[k] for k in sorted(attns.keys())]
    if isinstance(attns, Sequence):
        attns = [a.detach() for a in attns]

    # Reduce heads: mean or max
    if head_reduction in {"mean", "max"}:
        head_reduction = Reduce("B h Q K -> B Q K", head_reduction)
    attns = [head_reduction(a) for a in attns]

    # adjust for self-connections
    if adjust_residual:
        attns = [(a + torch.eye(*a.shape[-2:], device=a.device)) / 2 for a in attns]

    rollout = attns[0]
    for layer_idx in range(1, len(attns)):
        rollout = torch.einsum("bij,bjk->bik", attns[layer_idx], rollout)

    # Last layer is global avg pool, i.e. a single query token
    # that attends to all keys with uniform attention
    if global_avg_pool:
        rollout = reduce(rollout, "B Q K -> B K", "mean")

    return rollout


def slot_attn_rollout(
    attns: Union[Mapping[str, Tensor], Sequence[Tensor]], normalize="layer"
) -> Tensor:
    """Slot attention rollout: how much a slot token attends to context tokens.

    Args:
        attns: dict or list where each entry has shape ``[B S K]``
        normalize: `layer` to normalize the rollout at every layer,
            or `all` to normalize at the end only.

    Returns:
        Rollout, shape ``[B S K]``
    """
    if isinstance(attns, Mapping):
        attns = [attns[k] for k in sorted(attns.keys())]
    if isinstance(attns, Sequence):
        attns = [a.detach() for a in attns]
    attns = [normalize_sum_to_one(a, dim=-1) for a in attns]

    if normalize == "layer":
        rollout = attns[0]
        for layer_idx in range(1, len(attns)):
            rollout = normalize_sum_to_one(attns[layer_idx] + rollout, dim=-1)
    elif normalize == "all":
        rollout = torch.sum(torch.stack(attns, dim=0), dim=0)
        rollout = normalize_sum_to_one(rollout, dim=-1)
    else:
        raise ValueError(normalize)
    return rollout


def cross_attn_rollout(attns: Union[Mapping[str, Tensor], Sequence[Tensor]]) -> Tensor:
    """Cross attention rollout: how much an object token attends to context tokens.

    Args:
        attns: dict or list where entries alternate
            self attention with shape ``[B heads S S]``
            and cross attention with shape ``[B heads S K]``.

    Returns:
        Rollout, shape ``[B S K]``
    """
    if len(attns) % 2 != 0:
        raise ValueError("Must be pairs of self and cross attn")
    if isinstance(attns, Mapping):
        attns = list(attns.values())
    if isinstance(attns, Sequence):
        attns = [a.detach().mean(dim=1) for a in attns]

    device = attns[0].device
    B, S, S = attns[0].shape
    B, S, K = attns[1].shape
    id_ss = fill_diagonal_(torch.zeros(B, S, S, device=device), 1.0)
    R_ss = id_ss.clone()
    R_sk = torch.zeros(B, S, K, device=device)

    for attn in attns:
        if attn.shape == (B, S, S):
            # Self attention
            R_ss.add_(torch.bmm(attn, R_ss))
            R_sk.add_(torch.bmm(attn, R_sk))
        elif attn.shape == (B, S, K):
            # Cross attention
            R_ss_bar = normalize_sum_to_one(R_ss - id_ss) + id_ss
            R_sk.add_(torch.bmm(R_ss_bar.transpose(-2, -1), attn))
        else:
            raise ValueError(attn.shape)

    return normalize_sum_to_one(R_sk)


def obj_attn_rollout(attns: Union[Mapping[str, Tensor], Sequence[Tensor]]) -> Tensor:
    if isinstance(attns, Mapping):
        attns = list(attns.values())
    if isinstance(attns, Sequence):
        attns = [a.detach().mean(dim=1) for a in attns]
    # TODO implement obj attention rollout
    return attns[-1]
