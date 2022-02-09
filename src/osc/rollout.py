from typing import Callable, Mapping, Sequence, Union

import einops
import einops.layers.torch
import torch

from osc.utils import normalize_sum_to_one


def self_attn_rollout(
    attns: Union[Mapping[str, torch.Tensor], Sequence[torch.Tensor]],
    head_reduction: Union[str, Callable] = "mean",
    adjust_residual=True,
    global_avg_pool=True,
):
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
        head_reduction = einops.layers.torch.Reduce("B h Q K -> B Q K", head_reduction)
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
        rollout = einops.reduce(rollout, "B Q K -> B K", "mean")

    return rollout


def slot_attn_rollout(attns, normalize="layer"):
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
