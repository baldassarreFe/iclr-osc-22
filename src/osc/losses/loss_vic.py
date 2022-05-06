from typing import Tuple

import torch
import torch.jit
from torch import Tensor

from osc.utils import fill_diagonal

# TODO: consider if it's worth it to use the VIbC formulation
#       https://arxiv.org/pdf/2109.00783.pdf


@torch.jit.script
def variance_covariance_loss(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Variance and covariance of a batch of features.

    Args:
        x: tensor of shape ``[..., N, D]`` with optional leading dimensions.
            Variance and covariance will be computed using for each ``d`` dimension
            using ``N`` samples . The leading dimension(s) will be averaged over.

    Examples:

        Assume that ``x`` contains the ``D``-dimensional global features of a batch
        of ``N`` images, each augmented ``A=2`` times.
        In this case, the loss should be called with an input shape of ``[A, N, D]``.
        Otherwise, the loss would encourage the vectors corresponding to augmentations
        of the same image to be different.

        Assume that ``x`` contains the ``D``-dimensional global features of ``S`` object
        tokens of a batch of ``N`` images, each augmented ``A=2`` times.
        In this case, the loss should be called with an input shape of ``[A, S, N, D]``.
        Otherwise, the loss might encourage objects that appear identical in multiple
        images/augmentations to be different.

    Returns:
        A pair of scalar tensors.
    """
    N, D = x.shape[-2:]
    x = x - x.mean(dim=-2, keepdim=True)

    # cov: bmm([..., N, D].mT, [..., N, D]) -> [..., D, D]
    cov = torch.einsum("...nd, ...nc -> ...dc", x, x) / (N - 1)

    # [..., D, D] -> [..., D] -> scalar
    var = torch.diagonal(cov, dim1=-2, dim2=-1)
    var = torch.relu(1 - torch.sqrt(var + 1e-4))
    var = torch.mean(var)

    # [..., D, D] -> [...] -> scalar
    cov = fill_diagonal(cov, 0.0)
    cov = torch.sum(cov.pow(2), dim=(-2, -1)) / (D * (D - 1))
    cov = torch.mean(cov)

    return var, cov
