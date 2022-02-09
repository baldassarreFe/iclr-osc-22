from typing import Tuple

import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F

from osc.utils import cos_pairwise, l2_normalize


def matching_contrastive_loss(
    slots: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Contrastive object-wise loss, all vs. all.

    The vectors in position ``i`` and ``i + B`` of `projs` must represent ``K``
    embeddings of different augmentations of the same image. For each ``(i, i+B)``
    image pair, the ``K`` embeddings are 1:1 matched using linear-sum assignment
    to produce the targets for a ``2BK-1``-classes classification problem.
    Except for the matching slot, all ``2BK-2`` slots of all images are considered
    negative samples for the loss.

    If all slot embeddings collapse to the same value, the loss will be ``log(2BK-1)``.

    Worst case:
    if all embeddings collapse to the same value, the loss will be ``log(2BK-1)``.

    Best case:
    if each image gets an embedding that is orthogonal to all others,
    the loss will be ``log(exp(1/t) + 2BK - 2) - 1/t``.

    Args:
        slots: [2B, K, C] tensor of projected image features
        temperature: temperature scaling
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss over all samples and slots if reduction is 'mean' or 'sum'.
        A vector ``2BK`` of losses if reduction is 'none'

    Example:
        A batch of ``B=4`` images, augmented twice, each with `K=3` slots.
        The ``X`` represent positive matching targets for the cross entropy loss,
        the ``.`` represent negatives included in the loss (all except diagonal)::

                                    aug_0                    aug_1
                           -----------------------  -----------------------
                             0     1     2     3      0     1     2     3
                  |       [  . .|. . .|. . .|. . .||. . X|. . .|. . .|. . .]
                  | img_0 [.   .|. . .|. . .|. . .||. X .|. . .|. . .|. . .]
                  |       [. .  |. . .|. . .|. . .||X . .|. . .|. . .|. . .]
                  |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [. . .|  . .|. . .|. . .||. . .|X . .|. . .|. . .]
                  | img_1 [. . .|.   .|. . .|. . .||. . .|. X .|. . .|. . .]
                  |       [. . .|. .  |. . .|. . .||. . .|. . X|. . .|. . .]
            aug_0 |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [. . .|. . .|  . .|. . .||. . .|. . .|. . X|. . .]
                  | img_2 [. . .|. . .|.   .|. . .||. . .|. . .|X . .|. . .]
                  |       [. . .|. . .|. .  |. . .||. . .|. . .|. X .|. . .]
                  |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [. . .|. . .|. . .|  . .||. . .|. . .|. . .|. . X]
                  | img_3 [. . .|. . .|. . .|.   .||. . .|. . .|. . .|. X .]
                  |       [. . .|. . .|. . .|. .  ||. . .|. . .|. . .|X . .]
                          [=====|=====|=====|============|=====|=====|=====]
                  |       [. . X|. . .|. . .|. . .||  . .|. . .|. . .|. . .]
                  | img_0 [. X .|. . .|. . .|. . .||.   .|. . .|. . .|. . .]
                  |       [X . .|. . .|. . .|. . .||. .  |. . .|. . .|. . .]
                  |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [. . .|X . .|. . .|. . .||. . .|  . .|. . .|. . .]
                  | img_1 [. . .|. X .|. . .|. . .||. . .|.   .|. . .|. . .]
                  |       [. . .|. . X|. . .|. . .||. . .|. .  |. . .|. . .]
            aug_1 |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [. . .|. . .|. X .|. . .||. . .|. . .|  . .|. . .]
                  | img_2 [. . .|. . .|. . X|. . .||. . .|. . .|.   .|. . .]
                  |       [. . .|. . .|X . .|. . .||. . .|. . .|. .  |. . .]
                  |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [. . .|. . .|. . .|. . X||. . .|. . .|. . .|  . .]
                  | img_3 [. . .|. . .|. . .|. X .||. . .|. . .|. . .|.   .]
                  |       [. . .|. . .|. . .|X . .||. . .|. . .|. . .|. .  ]
    """
    A = 2
    B = slots.shape[0] // A
    K = slots.shape[1]

    # Full cosine similarity matrix between all slots of all images.
    # cos: [2, B, K, 2, B, K]
    slots = slots.reshape(A * B * K, slots.shape[2])
    cos = cos_pairwise(slots).reshape(A, B, K, A, B, K)

    # Prepare cross-entropy targets by running linear sum assignment
    # on cosine similarity for each pair of augmented images.
    #
    # Thanks to symmetry w.r.t. the diagonal, matches need to be computed
    # only for the B blocks of size [K, K] in the diagonal of the top-right
    # quarter of the cosine matrix:
    #   for i in range(B):
    #       match(cos_pairwise(slots[i], slots[i+B]))
    # instead of 2B times:
    #   for i in range(2*B):
    #       match(cos_pairwise(slots[i % B], slots[(i+B) % B]))

    targets = np.full(A * B * K, fill_value=-1, dtype=int)
    for b in range(B):
        cos_np = cos[0, b, :, 1, b, :].detach().cpu().numpy()
        # First output is a vector of sorted row idxs [0, 1, ..., K]
        _, cols = scipy.optimize.linear_sum_assignment(cos_np, maximize=True)
        targets[b * K : (b + 1) * K] = (B + b) * K + cols
        targets[(B + b) * K : (B + b + 1) * K] = b * K + np.argsort(cols)
    targets = torch.from_numpy(targets).to(slots.device)

    cos = cos.reshape(A * B * K, A * B * K).div_(temperature).fill_diagonal_(-torch.inf)
    loss = F.cross_entropy(cos, targets, reduction=reduction)

    # Debug
    # with np.printoptions(linewidth=150, formatter={"bool": ".X".__getitem__}):
    #     probs = cos.detach().softmax(dim=-1).mul_(100).int().cpu().numpy()
    #     print(probs)
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
    slots: torch.Tensor, temperature: float = 1.0, reduction: str = "mean"
) -> torch.Tensor:
    """Contrastive object-wise loss, only between corresponding images.

    The ``K`` slots of the ``i``-th image are matched with the slots of the ``i+B``-th
    image and vice versa. For each slot, the contrastive image considers the matching
    slot in the corresponding image as positive. The other ``K-1`` slots in the
    corresponding image as negatives, as well as the other ``K-1`` slots in the original
    image. Slots are only matched between one image and its augmented version,
    never within the same image and never with other images.

    If all slot embeddings collapse to the same value, the loss will be ``log(2K-2)``.

    Args:
        slots: [2B, K, C] tensor of projected image features
        temperature: temperature scaling
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss over all samples and slots if reduction is 'mean' or 'sum'.
        A vector ``2BK`` of losses if reduction is 'none'

    Example:
        A batch of ``B=4`` images, augmented twice, each with `K=3` slots.
        The ``X`` represent positive matching targets for the cross entropy loss,
        the ``.`` represent negatives included in the loss (only from the augmented
        image and the image itself, but not the slot itself)::

                                    aug_0                    aug_1
                           -----------------------  -----------------------
                             0     1     2     3      0     1     2     3
                  |       [  . .|     |     |     ||. . X|     |     |     ]
                  | img_0 [.   .|     |     |     ||. X .|     |     |     ]
                  |       [. .  |     |     |     ||X . .|     |     |     ]
                  |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [     |  . .|     |     ||     |X . .|     |     ]
                  | img_1 [     |.   .|     |     ||     |. X .|     |     ]
                  |       [     |. .  |     |     ||     |. . X|     |     ]
            aug_0 |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [     |     |  . .|     ||     |     |. . X|     ]
                  | img_2 [     |     |.   .|     ||     |     |X . .|     ]
                  |       [     |     |. .  |     ||     |     |. X .|     ]
                  |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [     |     |     |  . .||     |     |     |. . X]
                  | img_3 [     |     |     |.   .||     |     |     |. X .]
                  |       [     |     |     |. .  ||     |     |     |X . .]
                          [=====|=====|=====|============|=====|=====|=====]
                  |       [. . X|     |     |     ||  . .|     |     |     ]
                  | img_0 [. X .|     |     |     ||.   .|     |     |     ]
                  |       [X . .|     |     |     ||. .  |     |     |     ]
                  |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [     |X . .|     |     ||     |  . .|     |     ]
                  | img_1 [     |. X .|     |     ||     |.   .|     |     ]
                  |       [     |. . X|     |     ||     |. .  |     |     ]
            aug_1 |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [     |     |. X .|     ||     |     |  . .|     ]
                  | img_2 [     |     |. . X|     ||     |     |.   .|     ]
                  |       [     |     |X . .|     ||     |     |. .  |     ]
                  |       [-----+-----+-----+------------+-----+-----+-----]
                  |       [     |     |     |. . X||     |     |     |  . .]
                  | img_3 [     |     |     |. X .||     |     |     |.   .]
                  |       [     |     |     |X . .||     |     |     |. .  ]
    """
    # slots: [2B K C] -> [2 B K C]
    A = 2
    B = slots.shape[0] // A
    K = slots.shape[1]
    slots = l2_normalize(slots).reshape(A, B, K, slots.shape[2])

    # cos_self0: [B K K] (each image with aug=0 with itself, ignore diagonal)
    cos_self0 = torch.einsum("bkc,blc->bkl", slots[0], slots[0]).div_(temperature)
    cos_self0 = torch.where(
        torch.eye(K, K, device=slots.device, dtype=torch.bool).expand(B, -1, -1),
        cos_self0.new_tensor(-torch.inf),
        cos_self0,
    )

    # cos_self1: [B K K] (each image with aug=1 with itself, ignore diagonal)
    cos_self1 = torch.einsum("bkc,blc->bkl", slots[1], slots[1]).div_(temperature)
    cos_self1 = torch.where(
        torch.eye(K, K, device=slots.device, dtype=torch.bool).expand(B, -1, -1),
        cos_self1.new_tensor(-torch.inf),
        cos_self1,
    )

    # cos_aug: [B K K] (each image with its augmented version)
    cos_aug = torch.einsum("bkc,blc->bkl", slots[0], slots[1]).div_(temperature)

    # Prepare targets
    cos_np = cos_aug.detach().cpu().numpy()
    targets = np.zeros(A * B * K, dtype=int)
    for b in range(B):
        # First output is a vector of sorted row indices [0, 1, ..., K]
        _, cols = scipy.optimize.linear_sum_assignment(cos_np[b, :, :], maximize=True)
        targets[b * K : (b + 1) * K] = K + cols
        targets[B * K + b * K : B * K + (b + 1) * K] = np.argsort(cols)
    targets = torch.from_numpy(targets).to(slots.device)

    # cos: [2BK, 2K]
    cos = torch.cat(
        [
            torch.cat(
                [
                    cos_self0.reshape(B * K, K),
                    cos_aug.reshape(B * K, K),
                ],
                dim=1,
            ),
            torch.cat(
                [
                    cos_aug.transpose(1, 2).reshape(B * K, K),
                    cos_self1.reshape(B * K, K),
                ],
                dim=1,
            ),
        ],
        dim=0,
    )
    loss = F.cross_entropy(cos, targets, reduction=reduction)

    # Debug
    # with np.printoptions(linewidth=150, formatter={"bool": ".X".__getitem__}):
    #     probs = cos.detach().softmax(dim=-1).mul_(100).int().cpu().numpy()
    #     print(probs)
    #     onehot = np.zeros(cos.shape, dtype=bool)
    #     onehot[np.arange(len(targets)), targets.cpu().numpy()] = 1
    #     print(onehot)

    return loss


# TODO matching_similarity_loss
"""
class MatchingSimilarityLoss(nn.Module):
    def __init__(self, embed_dim=128, loss_dim=128):
        super().__init__()
        self.proj = timm.models.layers.Mlp(
            in_features=embed_dim,
            hidden_features=embed_dim,
            out_features=loss_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(self, x):
        x = slots
        x = self.proj(x)
        B = x.shape[0] // 2

        # B K1 C, B K2 C -> B K1 K2
        x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        x0, x1 = torch.split(x, B, dim=0)
        cos = torch.einsum("bkc,blc->bkl", x0, x1).div_(-2).add_(0.5)
        cos_np = cos.detach().numpy()

        loss = torch.tensor(0.0)
        for b in range(B):
            rows, cols = scipy.optimize.linear_sum_assignment(cos_np[b, :, :])
            loss += cos[b, rows, cols].sum()

        loss.div_(B)
        return loss


matching_similarity_loss_fn = MatchingSimilarityLoss(embed_dim, embed_dim)
matching_sim_loss = matching_similarity_loss_fn(slots)
print(matching_sim_loss)
"""
