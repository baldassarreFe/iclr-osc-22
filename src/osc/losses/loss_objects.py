from typing import Dict, Optional, Tuple

import numpy as np
import scipy.optimize
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.functional import cross_entropy

from osc.models.core_model import ModelOutput
from osc.utils import cos_pairwise, l2_normalize

from .loss_vic import variance_covariance_loss


def matching_contrastive_loss(
    slots: Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Contrastive object-wise loss, all vs. all.

    The vectors at ``[b, 0, :, :]`` and ``[b, 1, :, :]`` of ``slots`` must represent
    ``S`` slot embeddings of shape ``D`` of different augmentations of the b-th image.
    For each image pair ``((b, 0), (b, 1)), the ``S`` embeddings are 1:1 matched using
    linear-sum assignment to produce the targets for a ``2BS-1``-classes classification
    problem. The matching slot represents the positive class, and the remaining
    ``2BS-2`` slots are considered negatives.

    Worst case:
    if all embeddings collapse to the same value, the loss will be ``log(2BS-1)``.

    Best case:
    if each image gets an embedding that is orthogonal to all others,
    the loss will be ``log(exp(1/t) + 2BS - 2) - 1/t``.

    Args:
        slots: ``[B, 2, S, D]`` tensor of projected image features
        temperature: temperature scaling
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss over all samples and slots if reduction is 'mean' or 'sum'.
        A tensor ``[B, 2, S]`` of losses if reduction is 'none'

    Example:
        A batch of ``B=4`` images, augmented twice, each with ``S=3``.
        Note the symmetry of matches along the diagonal.
        The ``X`` represent positive matching targets for the cross entropy loss,
        the ``.`` represent negatives included in the loss (all except diagonal)::

                              img_0       img_1        img_2       img_3
                           ╭─────────╮ ╭─────────╮  ╭─────────╮ ╭─────────╮
                             0     1     0     1      0     1     0     1
                  ╭       ┃  . .│. . X┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
                  │ aug_0 ┃.   .│. X .┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
                  │       ┃. .  │X . .┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
            img_0 │       ╋─────┼─────╋─────┼─────╋─────┼─────╋─────│─────╋
                  │       ┃. . X│  . .┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
                  │ aug_1 ┃. X .│.   .┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
                  ╰       ┃X . .│. .  ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━│━━━━━┃
                  ╭       ┃. . .│. . .┃  . .│. X .┃. . .│. . .┃. . .│. . .┃
                  │ aug_0 ┃. . .│. . .┃.   .│X . .┃. . .│. . .┃. . .│. . .┃
                  │       ┃. . .│. . .┃. .  │. . X┃. . .│. . .┃. . .│. . .┃
            img_1 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃. . .│. . .┃. X .│  . .┃. . .│. . .┃. . .│. . .┃
                  │ aug_1 ┃. . .│. . .┃X . .│.   .┃. . .│. . .┃. . .│. . .┃
                  ╰       ┃. . .│. . .┃. . X│. .  ┃. . .│. . .┃. . .│. . .┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━╋━━━━━┃
                  ╭       ┃. . .│. . .┃. . .│. . .┃  . .│. . X┃. . .│. . .┃
                  │ aug_0 ┃. . .│. . .┃. . .│. . .┃.   .│X . .┃. . .│. . .┃
                  │       ┃. . .│. . .┃. . .│. . .┃. .  │. X .┃. . .│. . .┃
            img_2 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃. . .│. . .┃. . .│. . .┃. X .│  . .┃. . .│. . .┃
                  │ aug_1 ┃. . .│. . .┃. . .│. . .┃. . X│.   .┃. . .│. . .┃
                  ╰       ┃. . .│. . .┃. . .│. . .┃X . .│. .  ┃. . .│. . .┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━│━━━━━┃
                  ╭       ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃  . .│. X .┃
                  │ aug_0 ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃.   .│. . X┃
                  │       ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃. .  │X . .┃
            img_3 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃. . X│  . .┃
                  │ aug_1 ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃X . .│.   .┃
                  ╰       ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃. X .│. .  ┃
    """
    B, A, S, D = slots.shape
    if A != 2:
        raise ValueError(f"Invalid shape {slots.shape}")

    # Full cosine similarity matrix between all slots of all images.
    # cos: [B, 2, S, B, 2, S]
    cos = cos_pairwise(slots)

    # Prepare cross-entropy targets by running linear sum assignment
    # on cosine similarity for each pair of augmented images.
    #
    # Thanks to symmetry w.r.t. the diagonal, matches need to be computed
    # only for the B blocks of size [S, S] that are in the top-right
    # quarter of each [A*S, A*S] block:
    #
    # for b in range(B):
    #     match(cos_pairwise(slots[b, 0], slots[B, 1]))
    #     match(cos_pairwise(slots[b, 1], slots[B, 0])) <- not needed, use argsort
    #
    # The only thing to take care of is to offset the column indices so that
    # they correspond to the desired location in the cos matrix.

    targets = torch.full((B, A, S), fill_value=-1)
    for b in range(B):
        cos_np = cos[b, 0, :, b, 1, :].detach().cpu().numpy()
        # First output is a vector of sorted row idxs [0, 1, ..., S]
        _, cols = scipy.optimize.linear_sum_assignment(cos_np, maximize=True)
        targets[b, 0, :] = torch.from_numpy(cols).add_(S * (A * b + 1))
        targets[b, 1, :] = torch.from_numpy(np.argsort(cols)).add_(S * A * b)

    targets = targets.reshape(B * A * S)
    cos = cos.reshape(B * A * S, B * A * S).div_(temperature).fill_diagonal_(-torch.inf)
    loss = cross_entropy(cos, targets.to(slots.device), reduction=reduction)
    if reduction == "none":
        loss = loss.reshape(B, A, S)

    # Debug
    # probs = cos.detach().reshape(B*A*S, B*A*S).softmax(dim=-1)
    # print(probs.mul(100).int().cpu().numpy())
    # with np.printoptions(linewidth=150, formatter={"bool": ".X".__getitem__}):
    #     onehot = np.zeros(cos.shape, dtype=bool)
    #     onehot[np.arange(len(targets)), targets.cpu().numpy()] = 1
    #     print(onehot)

    return loss


def matching_contrastive_loss_best_worst(
    batch_size: int, num_slots: int, temperature: float
) -> Tuple[float, float]:
    """Best and worst case for ``contrastive_loss()``."""
    best = (
        np.log(np.exp(1 / temperature) + 2 * batch_size * num_slots - 2)
        - 1 / temperature
    )
    worst = np.log(2 * batch_size * num_slots - 1)
    return best, worst


def matching_contrastive_loss_per_img(
    slots: Tensor, temperature: float = 1.0, reduction: str = "mean"
) -> Tensor:
    """Contrastive object-wise loss, only between corresponding images.

    For each image ``b``, the ``S`` slots of the first augmentation are matched with the
    ``S`` slots of the second augmentation and vice-versa. Matching slots are considered
    as positives for a ``S-1``-way classification loss. The remaining ``2S-2`` slots
    from the first and second augmentations are considered as negatives.
    Slots are only matched between one image and its augmented version,
    never within the same image and never with other images.

    If all slot embeddings collapse to the same value, the loss will be ``log(2S-2)``.

    Args:
        slots: ``[B, 2, S, C]`` tensor of projected object features
        temperature: temperature scaling
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss over all samples and slots if reduction is 'mean' or 'sum'.
        A tensor ``[B, 2, S]`` of losses if reduction is 'none'

    Example:
        A batch of ``B=4`` images, augmented twice, each with `K=3` slots.
        The ``X`` represent positive matching targets for the cross entropy loss,
        the ``.`` represent negatives included in the loss (only from the augmented
        image and the image itself, but not the slot itself)::

                              img_0       img_1        img_2       img_3
                           ╭─────────╮ ╭─────────╮  ╭─────────╮ ╭─────────╮
                             0     1     0     1      0     1     0     1
                  ╭       ┃  . .│. . X┃     │     ┃     │     ┃     │     ┃
                  │ aug_0 ┃.   .│. X .┃     │     ┃     │     ┃     │     ┃
                  │       ┃. .  │X . .┃     │     ┃     │     ┃     │     ┃
            img_0 │       ╋─────┼─────╋─────┼─────╋─────┼─────╋─────│─────╋
                  │       ┃. . X│  . .┃     │     ┃     │     ┃     │     ┃
                  │ aug_1 ┃. X .│.   .┃     │     ┃     │     ┃     │     ┃
                  ╰       ┃X . .│. .  ┃     │     ┃     │     ┃     │     ┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━│━━━━━┃
                  ╭       ┃     │     ┃  . .│. X .┃     │     ┃     │     ┃
                  │ aug_0 ┃     │     ┃.   .│X . .┃     │     ┃     │     ┃
                  │       ┃     │     ┃. .  │. . X┃     │     ┃     │     ┃
            img_1 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃     │     ┃. X .│  . .┃     │     ┃     │     ┃
                  │ aug_1 ┃     │     ┃X . .│.   .┃     │     ┃     │     ┃
                  ╰       ┃     │     ┃. . X│. .  ┃     │     ┃     │     ┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━╋━━━━━┃
                  ╭       ┃     │     ┃     │     ┃  . .│. . X┃     │     ┃
                  │ aug_0 ┃     │     ┃     │     ┃.   .│X . .┃     │     ┃
                  │       ┃     │     ┃     │     ┃. .  │. X .┃     │     ┃
            img_2 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃     │     ┃     │     ┃. X .│  . .┃     │     ┃
                  │ aug_1 ┃     │     ┃     │     ┃. . X│.   .┃     │     ┃
                  ╰       ┃     │     ┃     │     ┃X . .│. .  ┃     │     ┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━│━━━━━┃
                  ╭       ┃     │     ┃     │     ┃     │     ┃  . .│. X .┃
                  │ aug_0 ┃     │     ┃     │     ┃     │     ┃.   .│. . X┃
                  │       ┃     │     ┃     │     ┃     │     ┃. .  │X . .┃
            img_3 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃     │     ┃     │     ┃     │     ┃. . X│  . .┃
                  │ aug_1 ┃     │     ┃     │     ┃     │     ┃X . .│.   .┃
                  ╰       ┃     │     ┃     │     ┃     │     ┃. X .│. .  ┃
    """
    B, A, S, D = slots.shape
    if A != 2:
        raise ValueError(f"Invalid shape {slots.shape}")

    # For each image, cosine similarity between all slots of all augmentations.
    # Visualize it as a rectangular matrix of shapes [B*A*S, A*S].
    # cos: [B, (A, S), (A, S)] with entries [:, a, s, a, s] set to -inf
    slots = l2_normalize(slots)
    cos = torch.einsum("bisd, bjtd -> bisjt", slots, slots)
    cos = torch.where(
        torch.eye(A * S, A * S, device=cos.device, dtype=torch.bool)
        .expand(B, -1, -1)
        .reshape(B, A, S, A, S),
        cos.new_tensor(-torch.inf),
        cos,
    )

    # Prepare cross-entropy targets by running linear sum assignment
    # on cosine similarity for each pair of augmented images.
    #
    # Thanks to symmetry w.r.t. the diagonal, matches need to be computed
    # only for the B blocks of size [S, S] that are in the top-right
    # quarter of each [A*S, A*S] block:
    #
    # for b in range(B):
    #     match(cos_pairwise(slots[b, 0], slots[B, 1]))
    #     match(cos_pairwise(slots[b, 1], slots[B, 0])) <- not needed, use argsort
    #
    # The only thing to take care of is to offset the column indices so that
    # they correspond to the desired location in the cos matrix.

    targets = torch.full((B, A, S), fill_value=-1)
    for b in range(B):
        cos_np = cos[b, 0, :, 1, :].detach().cpu().numpy()
        # First output is a vector of sorted row idxs [0, 1, ..., S]
        _, cols = scipy.optimize.linear_sum_assignment(cos_np, maximize=True)
        targets[b, 0, :] = torch.from_numpy(cols).add_(S)
        targets[b, 1, :] = torch.from_numpy(np.argsort(cols))

    targets = targets.reshape(B * A * S)
    cos = cos.reshape(B * A * S, A * S).div_(temperature).fill_diagonal_(-torch.inf)
    loss = cross_entropy(cos, targets.to(slots.device), reduction=reduction)
    if reduction == "none":
        loss = loss.reshape(B, A, S)

    # Debug
    # probs = cos.detach().reshape(B*A*S, A*S).softmax(dim=-1)
    # print(probs.mul(100).int().cpu().numpy())
    # with np.printoptions(linewidth=150, formatter={"bool": ".X".__getitem__}):
    #     onehot = np.zeros(cos.shape, dtype=bool)
    #     onehot[np.arange(len(targets)), targets.cpu().numpy()] = 1
    #     print(onehot)

    return loss


def matching_similarity_loss_per_img(
    f: Tensor, p: Tensor, reduction: str = "mean"
) -> Tensor:
    """Cosine similarity per object, only between corresponding images.

    The ``S`` slots of the image at ``[b, 0]`` are matched with the slots of ``[b, 1]``
    and vice-versa. The loss encourages high cosine similarity between matching pairs.

    The loss is defined as ``(1-cos)/2`` so that it can be minimized to 0.

    Args:
        f: ``[B, 2, S, C]`` tensor of pre-projection object features (stop grad)
        p: ``[B, 2, S, C]`` tensor of post-projection object features
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss over all samples and slots if reduction is 'mean' or 'sum'.
        A tensor ``[B, 2, S]`` of losses if reduction is 'none'

    Example:
        A batch of ``B=4`` images, augmented twice, each with `S=3` slots.
        The ``X`` matching pairs whose cosine similarity will be increased.
        When computing the cosine, the vectors along the column axis are detached
        to prevent gradient propagation::

                              img_0       img_1        img_2       img_3
                           ╭─────────╮ ╭─────────╮  ╭─────────╮ ╭─────────╮
                             0     1     0     1      0     1     0     1
                  ╭       ┃     │    X┃     │     ┃     │     ┃     │     ┃
                  │ aug_0 ┃     │  X  ┃     │     ┃     │     ┃     │     ┃
                  │       ┃     │X    ┃     │     ┃     │     ┃     │     ┃
            img_0 │       ╋─────┼─────╋─────┼─────╋─────┼─────╋─────│─────╋
                  │       ┃    X│     ┃     │     ┃     │     ┃     │     ┃
                  │ aug_1 ┃  X  │     ┃     │     ┃     │     ┃     │     ┃
                  ╰       ┃X    │     ┃     │     ┃     │     ┃     │     ┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━│━━━━━┃
                  ╭       ┃     │     ┃     │  X  ┃     │     ┃     │     ┃
                  │ aug_0 ┃     │     ┃     │X    ┃     │     ┃     │     ┃
                  │       ┃     │     ┃     │    X┃     │     ┃     │     ┃
            img_1 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃     │     ┃  X  │     ┃     │     ┃     │     ┃
                  │ aug_1 ┃     │     ┃X    │     ┃     │     ┃     │     ┃
                  ╰       ┃     │     ┃    X│     ┃     │     ┃     │     ┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━╋━━━━━┃
                  ╭       ┃     │     ┃     │     ┃     │    X┃     │     ┃
                  │ aug_0 ┃     │     ┃     │     ┃     │X    ┃     │     ┃
                  │       ┃     │     ┃     │     ┃     │  X  ┃     │     ┃
            img_2 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃     │     ┃     │     ┃  X  │     ┃     │     ┃
                  │ aug_1 ┃     │     ┃     │     ┃    X│     ┃     │     ┃
                  ╰       ┃     │     ┃     │     ┃X    │     ┃     │     ┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━│━━━━━┃
                  ╭       ┃     │     ┃     │     ┃     │     ┃     │  X  ┃
                  │ aug_0 ┃     │     ┃     │     ┃     │     ┃     │    X┃
                  │       ┃     │     ┃     │     ┃     │     ┃     │X    ┃
            img_3 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃     │     ┃     │     ┃     │     ┃    X│     ┃
                  │ aug_1 ┃     │     ┃     │     ┃     │     ┃X    │     ┃
                  ╰       ┃     │     ┃     │     ┃     │     ┃  X  │     ┃
    """
    B, A, S, D = f.shape
    if A != 2:
        raise ValueError(f"Invalid shape {f.shape}")
    f = l2_normalize(f.detach())
    p = l2_normalize(p)

    # Cosine similarity of all slots in aug 0 with all slots in aug 1,
    # will be used to compute the matching pairs. However, the loss is defined between
    # the detached pre-projection features and the post-projection features.
    # cos_01: [B S T], T=S
    # cos_01[b, s, t] = cos(f[b, 0, s], f[b, 1, t])
    cos_01 = torch.einsum("bsd, btd -> bst", f[:, 0], f[:, 1]).cpu().numpy()

    # Find matches and accumulate cosine similarity of matching pairs
    sim = f.new_zeros((B, A, S))
    for b in range(B):
        # First output is a vector of sorted row indices [0, 1, ..., S]
        _, cols = scipy.optimize.linear_sum_assignment(cos_01[b, :, :], maximize=True)

        # Top-right quadrant of a [AS, AS] block in the example matrix above
        sim[b, 0, :] += torch.sum(p[b, 0, :] * f[b, 1, cols], dim=-1)

        # Bottom-left quadrant, use argsort to get the symmetric indices
        cols_t = np.argsort(cols)
        sim[b, 1, :] += torch.sum(p[b, 1, :] * f[b, 0, cols_t], dim=-1)

    if reduction == "sum":
        sim = sim.sum()
    elif reduction == "mean":
        sim = sim.mean()

    return (1 - sim) / 2


def matching_similarity_loss_small_to_large(
    f_large: Tensor, p_small: Tensor, reduction: str = "mean"
) -> Tuple[Tensor, Tensor]:
    """Matching cosine loss between objects from small crops to objects in large crops.

    Symbols:

        - ``B`` batch size
        - ``Al`` number of large crops
        - ``As`` number of small crops
        - ``L`` number of objects for each large crop
        - ``S`` number of objects for each small crop

    Args:
        f_large: ``[B, Al, L, C]`` tensor of pre-projection object features (stop grad)
        p_small: ``[B, As, S, C]`` tensor of post-projection object features
        reduction: 'mean', 'sum', or 'none'. If 'none' it returns a ``[B, Al, As]``
            tensor of losses which might contain ``NaN`` if no reciprocal matches
            were found for that pair of images.

    Returns:

    """
    S = p_small.shape[-2]
    f_large = l2_normalize(f_large.detach())
    p_small = l2_normalize(p_small)

    # Each entry at [b, a_small, a_large] is a [S, L] matrix of similarities
    cos = torch.einsum("bi sd, bj ld -> bij sl", p_small, f_large)

    # For each row (small obj), value and idx of the most similar column (teacher obj)
    # values: [B, A_small, A_large, S]
    # idx_small: [B, A_small, A_large, S]
    loss, idx_small = cos.max(-1)
    loss = (1 - loss) / 2

    # For each column (teacher obj), the index of the most similar row (small object)
    # idx_teacher: [B, A_small, A_large, L]
    idx_teacher = cos.argmax(-2)

    # For each row (small object), is the best-matching large object a reciprocal?
    # reciprocal: [B, A_small, A_large, S]
    reciprocal = torch.eq(
        idx_teacher.gather(index=idx_small, dim=-1), torch.arange(S, device=loss.device)
    )

    # For each entry [b, a_small, a_large], ratio of reciprocal small -> teacher matches
    ratio = reciprocal.float().mean(-1)

    # For each entry [b, a_small, a_large], avg similarity of small -> teacher matches.
    # Can contain NaN entries if no matches are found.
    # loss: [B, A_small, A_large]
    loss = torch.where(reciprocal, loss, loss.new_tensor(torch.nan))
    loss = loss.nanmean(-1)

    if reduction == "mean":
        loss = loss.nanmean()
        ratio = ratio.mean()
    elif reduction == "sum":
        loss = loss.nansum()
        ratio = ratio.sum()
    return loss, ratio


def compute_losses_objects(
    cfg: DictConfig,
    out_large: ModelOutput,
    out_small: Optional[ModelOutput] = None,
) -> (Tensor, Dict[str, Tensor]):
    """Compute all object losses. Always use ``reduction="mean"``."""
    cfg = cfg.losses.l_objects
    loss = torch.zeros([], device=out_large.f_slots.device)
    logs: Dict[str, Tensor] = {}

    if cfg.ctr_all.weight > 0:
        ctr_all = matching_contrastive_loss(
            slots=out_large.p_slots,
            temperature=cfg.ctr_all.temp,
            reduction="mean",
        )
        loss = loss + cfg.ctr_all.weight * ctr_all
        logs["loss/objects/ctr_all"] = ctr_all

    if cfg.ctr_img.weight > 0:
        ctr_img = matching_contrastive_loss_per_img(
            slots=out_large.p_slots,
            temperature=cfg.ctr_img.temp,
            reduction="mean",
        )
        loss = loss + cfg.ctr_img.weight * ctr_img
        logs["loss/objects/ctr_img"] = ctr_img

    if cfg.sim_img.weight > 0:
        sim_img = matching_similarity_loss_per_img(
            f=out_large.f_slots,
            p=out_large.p_slots,
            reduction="mean",
        )
        loss = loss + cfg.sim_img.weight * sim_img
        logs["loss/objects/sim_img"] = sim_img

    if cfg.sim_small.weight > 0:
        sim_small, ratio = matching_similarity_loss_small_to_large(
            f_large=out_large.f_slots,
            p_small=out_small.p_slots,
            reduction="mean",
        )
        loss = loss + cfg.sim_small.weight * sim_small
        logs["loss/objects/sim_small"] = sim_small
        logs["metric/objects/reciprocal"] = ratio

    # Variance and covariance are always computed, so we can log them.
    # Compute variance and covariance of S slots for each [b, a] pair
    var, cov = variance_covariance_loss(out_large.f_slots)
    loss = loss + cfg.var.weight * var
    loss = loss + cfg.cov.weight * cov
    logs["loss/objects/var"] = var
    logs["loss/objects/cov"] = cov

    logs["loss/objects/total"] = loss

    return loss, logs
