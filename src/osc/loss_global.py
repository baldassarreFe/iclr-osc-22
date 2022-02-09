from typing import Tuple

import numpy as np
import torch
import torch.nn.functional

from osc.utils import cos_pairwise, l2_normalize


@torch.jit.script
def cosine_sim_loss(
    feats: torch.Tensor, projs: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Cosine similarity loss for BYOL and SimSiam.

    For both `feats` and `projs`, the vectors in position ``i`` and ``i + B``
    must represent the embeddings of different augmentations of the same image.

    In BYOL, `feats` should come from an offline network updated with momentum.
    In SimSiam, `feats` comes from the same online network that computes `proj`.
    Detach (stop gradient) is always called on the `feats` embeddings.

    The loss is made symmetric w.r.t. the image augmentations.

    Examples:

        BYOL:

        >>> imgs = torch.rand(4, 3, 256, 256)
        >>> inputs = torch.cat([aug0(imgs), aug1(imgs)], dim=0)
        >>> feats_offline = backbone_offline(inputs)
        >>> feats = backbone(inputs)
        >>> projs = projection(feats)
        >>> cosine_sim_loss(feats_offline, projs)

        SimSiam:

        >>> imgs = torch.rand(4, 3, 256, 256)
        >>> inputs = torch.cat([aug0(imgs), aug1(imgs)], dim=0)
        >>> feats = backbone(inputs)
        >>> projs = projection(feats)
        >>> cosine_sim_loss(feats, projs)

    Args:
        feats: [2B, C] tensor of image features
        projs: [2B, C] tensor of projected image features
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss if reduction is 'mean' or 'sum'.
        A vector of ``2B`` losses if reduction is 'none'

    """
    B = feats.shape[0] // 2
    feats = l2_normalize(feats.detach()).roll(B, dims=0)
    projs = l2_normalize(projs)
    if reduction == "mean":
        # Same as: - (feats * projs).sum(dim=1).mean(dim=0)
        return torch.einsum("bc,bc->", feats, projs).div_(-2 * B)
    elif reduction == "sum":
        # Same as: - (feats * projs).sum(dim=1).sum(dim=0)
        return torch.einsum("bc,bc->", feats, projs).neg_()
    elif reduction == "none":
        # Same as: - (feats * projs).sum(dim=1)
        return torch.einsum("bc,bc->b", feats, projs).neg_()
    raise ValueError(reduction)


@torch.jit.script
def contrastive_loss(
    projs: torch.Tensor, temperature: float = 1.0, reduction: str = "mean"
) -> torch.Tensor:
    """Contrastive loss for SimCLR.

    The vectors in position ``i`` and ``i + B`` of `projs` must
    represent the embeddings of different augmentations of the same image.
    The loss can be thought as a classification problem with ``2B-1`` classes where
    the ``i``-th sample should be classified as the ``i+B``-th sample and viceversa.
    The loss is made symmetric w.r.t. the image augmentations.

    Worst case:
    if all embeddings collapse to the same value, the loss will be ``log(2B-1)``.

    Best case:
    if each image gets an embedding that is orthogonal to all others,
    the loss will be ``log(exp(1/t) + 2B - 2) - 1/t``.

    Examples:

        SimSiam:

        >>> imgs = torch.rand(4, 3, 256, 256)
        >>> inputs = torch.cat([aug0(imgs), aug1(imgs)], dim=0)
        >>> feats = backbone(inputs)
        >>> projs = projection(feats)
        >>> contrastive_loss(projs)

    Args:
        projs: [2B, C] tensor of projected image features
        temperature: temperature scaling
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss if reduction is 'mean' or 'sum'.
        A vector of ``2B`` losses if reduction is 'none'
    """
    B = projs.shape[0] // 2
    cos = cos_pairwise(projs).div_(temperature).fill_diagonal_(-torch.inf)  # [2B, 2B]
    target = torch.arange(2 * B, device=cos.device).roll(B)  # [2B]
    loss = torch.nn.functional.cross_entropy(cos, target, reduction=reduction)
    return loss


def contrastive_loss_best_worst(
    batch_size: int, temperature: float
) -> Tuple[float, float]:
    """Best and worst case for ``contrastive_loss()``."""
    best = np.log(np.exp(1 / temperature) + 2 * batch_size - 2) - 1 / temperature
    worst = np.log(2 * batch_size - 1)
    return best, worst
