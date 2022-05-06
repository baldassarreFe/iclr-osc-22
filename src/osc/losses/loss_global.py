from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor

from osc.models.core_model import ModelOutput
from osc.utils import cos_pairwise, l2_normalize

from .loss_vic import variance_covariance_loss


@torch.jit.script
def cosine_sim_loss(feats: Tensor, projs: Tensor, reduction: str = "mean") -> Tensor:
    """Cosine similarity loss for BYOL and SimSiam.

    For both `feats` and `projs`, the vectors at ``[b, 0, :]`` and ``[b, 1, :]``
    must represent the embeddings of different augmentations of the same image.

    In BYOL, `feats` should come from an offline network updated with momentum.
    In SimSiam, `feats` comes from the same online network that computes `proj`.
    Detach (stop gradient) is always called on the `feats` embeddings.

    The loss is made symmetric w.r.t. the image augmentations.

    The loss is defined as ``(1-cos)/2`` so that it can be minimized to 0.

    Examples:

        BYOL:

        >>> imgs = torch.rand(B, 3, H, W)
        >>> inputs = torch.stack([aug(imgs), aug(imgs)], dim=1).reshape(B * 2, 3, H, W)
        >>> feats_offline = backbone_offline(inputs).reshape(B, 2, -1)
        >>> projs = projection(backbone(inputs)).reshape(B, 2, -1)
        >>> cosine_sim_loss(feats_offline, projs)

        SimSiam:

        >>> imgs = torch.rand(B, 3, H, W)
        >>> inputs = torch.stack([aug(imgs), aug(imgs)], dim=1).reshape(B * 2, 3, H, W)
        >>> feats = backbone(inputs).reshape(B, 2, -1)
        >>> projs = projection(feats).reshape(B, 2, -1)
        >>> cosine_sim_loss(feats, projs)

    Args:
        feats: [B, 2, C] tensor of image features
        projs: [B, 2, C] tensor of projected image features
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss if reduction is 'mean' or 'sum'.
        A matrix of ``[B, 2]`` losses if reduction is 'none'

    """
    B, A, C = feats.shape
    if A != 2:
        raise ValueError(f"Invalid shape {feats.shape}")
    feats = l2_normalize(feats.detach())
    projs = l2_normalize(projs)
    # cos[b, a] = cos(projs[b, a], feats[b, (a + 1) % 2])
    cos = torch.sum(projs * feats.roll(1, dims=1), dim=-1)
    if reduction == "mean":
        cos = cos.mean()
    elif reduction == "sum":
        cos = cos.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction {reduction}")
    return (1 - cos) / 2


@torch.jit.script
def contrastive_loss(
    projs: Tensor, temperature: float = 1.0, reduction: str = "mean"
) -> Tuple[Tensor, Tensor]:
    """Contrastive loss for SimCLR.

    The vectors at ``[b, 0, :]`` and ``[b, 1, :]`` of ``projs`` must represent the
    embeddings of different augmentations of the ``b``-th image.
    The loss can be thought as a classification problem with ``2B-1`` classes where
    the sample at ``[b, 0, :]`` should be classified as the sample at ``[b, 1, :]``
    and viceversa.

    The loss is made symmetric w.r.t. the image augmentations.

    Worst case:
    if all embeddings collapse to the same value, the loss will be ``log(2B-1)``.

    Best case:
    if each pair of images get a unique embedding that is orthogonal to all others,
    the loss will be ``log(exp(1/t) + 2B - 2) - 1/t``.

    Examples:

        SimSiam:

        >>> imgs = torch.rand(B, 3, H, W)
        >>> inputs = torch.stack([aug(imgs), aug(imgs)], dim=1).reshape(B * 2, 3, H, W)
        >>> feats = backbone(inputs.reshape(B, 2, -1)
        >>> projs = projection(feats).reshape(B, 2, -1)
        >>> contrastive_loss(projs)

    Args:
        projs: [B, 2, C] tensor of projected image features
        temperature: temperature scaling
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss if reduction is 'mean' or 'sum'. A vector of ``[B, 2]`` losses
        if reduction is 'none'. Also returns the matching accuracy.
    """
    B, A, C = projs.shape
    device = projs.device
    if A != 2:
        raise ValueError(f"Invalid shape {projs.shape}")
    projs = l2_normalize(projs).reshape(B * 2, C)
    # cos [B*2, B*2]
    cos = cos_pairwise(projs @ projs.mT).div_(temperature).fill_diagonal_(-torch.inf)
    # target: [1, 0, 3, 2, 5, 4, 7, 6, ..., 2B, 2B-1]
    target = torch.arange(B * 2).reshape(B, 2).roll(1, dims=1).reshape(B * 2).to(device)
    loss = torch.nn.functional.cross_entropy(cos, target, reduction=reduction)
    accuracy = loss.detach().argmax(-1).eq(target).float().mean()
    return loss, accuracy


def contrastive_loss_best_worst(
    batch_size: int, temperature: float
) -> Tuple[float, float]:
    """Best and worst case for ``contrastive_loss()``."""
    best = np.log(np.exp(1 / temperature) + 2 * batch_size - 2) - 1 / temperature
    worst = np.log(2 * batch_size - 1)
    return best, worst


def compute_losses_global(
    cfg: DictConfig, output: ModelOutput
) -> (Tensor, Dict[str, Tensor]):
    """Compute all global losses. Always use ``reduction="mean"``."""
    cfg = cfg.losses.l_global
    loss = torch.zeros([], device=output.f_global.device)
    logs: Dict[str, Tensor] = {}

    if cfg.ctr.weight > 0:
        ctr, acc = contrastive_loss(projs=output.p_global, temperature=cfg.ctr.temp)
        loss = loss + cfg.ctr.weight * ctr
        logs["loss/global/ctr"] = ctr
        logs["metric/global/acc"] = acc

    if cfg.sim.weight > 0:
        sim = cosine_sim_loss(feats=output.f_global, projs=output.p_global)
        loss = loss + cfg.sim.weight * sim
        logs["loss/global/sim"] = sim

    # Variance and covariance are always computed, so we can log them.
    # The `A` augmentations are kept distinct and averaged.
    f = rearrange(output.f_global, "B A C -> A B C")
    var, cov = variance_covariance_loss(f)
    loss = loss + cfg.var.weight * var
    loss = loss + cfg.cov.weight * cov
    logs["loss/global/var"] = var
    logs["loss/global/cov"] = cov

    logs["loss/global/total"] = loss

    return loss, logs
