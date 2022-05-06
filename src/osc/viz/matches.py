"""
Visualization of matching global-global and object-object.
"""
import numpy as np
import scipy.optimize
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor

from osc.utils import cos_pairwise


def viz_matches_global(
    imgs_np: np.ndarray,
    global_feats: Tensor,
) -> Figure:
    """Visualize global similarity between``B*2`` images.

    Args:
        imgs_np: numpy images, shape ``[B, 2, H, W, C]``
        global_feats: a tensor with shape [B, 2, C]

    Returns:
        A figure.
    """
    global_feats = global_feats.detach()
    B, A, _ = global_feats.shape
    if A != 2:
        raise ValueError(f"Invalid shape {global_feats.shape}")

    # cos: [BA, BA]
    cos = (
        cos_pairwise(global_feats)
        .reshape(B * A, B * A)
        .fill_diagonal_(torch.nan)
        .cpu()
        .numpy()
    )
    target = torch.arange(B * 2).reshape(B, 2).roll(1, dims=1).reshape(B * 2).numpy()
    acc = np.mean(np.nanargmax(cos, axis=-1) == target)

    fig, axs = plt.subplots(
        2,
        3,
        facecolor="white",
        figsize=1.0 * np.array([1 + B * A + 0.1, 1 + B * A]),
        gridspec_kw={
            "height_ratios": [1, B * A],
            "width_ratios": [1, B * A, 0.1],
            "hspace": 0,
            "wspace": 0.1,
        },
    )
    axs[0, 0].set_axis_off()
    axs[0, 2].set_axis_off()

    # Top row
    ax = axs[0, 1]
    ax.imshow(rearrange(imgs_np, "B A H W C -> H (B A W) C"))
    ax.set_axis_off()
    ax.set_title(
        f"Global similarity\n"
        f"Min {np.nanmin(cos):.2f} "
        f"Max {np.nanmax(cos):.2f} "
        f"Acc {acc:.2%}"
    )

    # Left column
    ax = axs[1, 0]
    ax.imshow(rearrange(imgs_np, "B A H W C -> (B A H) W C"))
    ax.set_axis_off()

    # Main plot
    ax = axs[1, 1]
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(cos, interpolation="none")
    fig.colorbar(im, cax=axs[1, 2])
    ax.scatter(
        target,
        np.arange(B * A),
        color="white",
        edgecolors="black",
        marker="s",
        s=100,
        label="Target",
    )
    ax.scatter(
        np.nanargmax(cos, axis=-1), np.arange(B * A), color="red", s=20, label="Match"
    )
    ax.legend()
    for i in range(2, B * 2, 2):
        ax.axvline(i - 0.5, c="black", lw=2)
        ax.axhline(i - 0.5, c="black", lw=2)

    return fig


def viz_matches_object(
    imgs_np: np.ndarray,
    obj_feats: Tensor,
) -> Figure:
    """Visualize similarity between ``B*2`` images with ``S`` objects each.

    Args:
        imgs_np: numpy images, shape ``[B, 2, H, W, C]``
        obj_feats: object feature tensor of shape ``[B, 2, S, D]``,
            where ``S`` is the number of objects per image.

    Returns:
        A figure with a single plot of matching probabilities.
    """
    obj_feats = obj_feats.detach()
    B, A, S, _ = obj_feats.shape
    if A != 2:
        raise ValueError(f"Invalid shape {obj_feats.shape}")

    # cos: [BAS, BAS]
    cos = (
        cos_pairwise(obj_feats)
        .reshape(B * A * S, B * A * S)
        .fill_diagonal_(torch.nan)
        .cpu()
        .numpy()
    )

    fig, axs = plt.subplots(
        2,
        3,
        facecolor="white",
        figsize=1.0 * np.array([S + B * A * S + 0.1, S + B * A * S]) / S,
        gridspec_kw={
            "height_ratios": [S, B * A * S],
            "width_ratios": [S, B * A * S, 0.1],
            "hspace": 0,
            "wspace": 0.1,
        },
    )
    axs[0, 0].set_axis_off()
    axs[0, 2].set_axis_off()

    # Top row
    ax = axs[0, 1]
    ax.imshow(rearrange(imgs_np, "B A H W C -> H (B A W) C"))
    ax.set_axis_off()
    ax.set_title(
        f"Object similarity\n" f"Min {np.nanmin(cos):.2f} " f"Max {np.nanmax(cos):.2f}"
    )

    # Left column
    ax = axs[1, 0]
    ax.imshow(rearrange(imgs_np, "B A H W C -> (B A H) W C"))
    ax.set_axis_off()

    # Main plot
    ax = axs[1, 1]
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(cos, interpolation="none")
    fig.colorbar(im, cax=axs[1, 2])
    for i in range(S, B * 2 * S, S):
        ax.axvline(i - 0.5, c="black", lw=1)
        ax.axhline(i - 0.5, c="black", lw=1)
    for i in range(2 * S, B * 2 * S, 2 * S):
        ax.axvline(i - 0.5, c="black", lw=2)
        ax.axhline(i - 0.5, c="black", lw=2)

    matches = match_objects(obj_feats)
    ax.scatter(matches, np.arange(B * 2 * S), s=1, color="red", label="Match")
    ax.legend()

    return fig


def match_objects(obj_feats: Tensor) -> np.ndarray:
    """Compute matches between ``B*2`` images with ``S`` objects each.

    Args:
        obj_feats: object feature tensor of shape ``[B, 2, S, C]``,
            where ``S`` is the number of objects per image.

    Returns:
        An array of matches (one match per row), length ``B*2*S``.
        Indices refer to the full ``[B*2*S, B*2*S]`` similarity matrix.
    """
    obj_feats = obj_feats.detach()
    B, A, S, _ = obj_feats.shape
    if A != 2:
        raise ValueError(f"Invalid shape {obj_feats.shape}")

    cos_01 = torch.einsum("bsd, btd -> bst", obj_feats[:, 0], obj_feats[:, 1])
    cos_01 = cos_01.cpu().numpy()
    targets = np.zeros((B, A, S), dtype=int)
    for b in range(B):
        # First output is a vector of sorted row idxs [0, 1, ..., S]
        _, cols = scipy.optimize.linear_sum_assignment(cos_01[b, :, :], maximize=True)
        targets[b, 0, :] = S * (A * b + 1) + cols
        targets[b, 1, :] = S * A * b + np.argsort(cols)
    return targets.flatten()
