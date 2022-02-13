"""
Visualization of object losses.
"""
import numpy as np
import scipy.optimize
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from osc.utils import cos_pairwise

from .utils import subplots_grid


def match_objects(obj_feats: torch.Tensor) -> np.ndarray:
    """Compute matches between 2B images with S objects each.

    Args:
        obj_feats: object feature tensor of shape ``[2B S C]``,
            where ``S`` is the number of objects per image.

    Returns:
        An array of matches (one match per row), length ``2BS``
    """
    B, S, C = obj_feats.shape
    B //= 2
    s0, s1 = obj_feats.detach().reshape(2, B, S, C).unbind(dim=0)
    cos = torch.einsum("ikc,ilc->ikl", s0, s1).cpu().numpy()
    targets = np.zeros(2 * B * S, dtype=int)
    for b in range(B):
        # First output is a vector of sorted row idxs [0, 1, ..., S]
        _, cols = scipy.optimize.linear_sum_assignment(cos[b, :, :], maximize=True)
        targets[b * S : (b + 1) * S] = (B + b) * S + cols
        targets[(B + b) * S : (B + b + 1) * S] = b * S + np.argsort(cols)
    return targets


def viz_contrastive_loss_objects(obj_feats: torch.Tensor, loss: float) -> Figure:
    """Visualize contrastive loss between 2B images with S objects each.

    Args:
        obj_feats: object feature tensor of shape ``[2B S C]``,
            where ``S`` is the number of objects per image.
        loss: loss value to display on the plots

    Returns:
        A figure with two plots.
    """
    B = obj_feats.shape[0] // 2
    S = obj_feats.shape[1]

    cos = cos_pairwise(obj_feats.detach().reshape(2 * B * S, -1))
    prob = (
        torch.clone(cos)
        .fill_diagonal_(-torch.inf)
        .softmax(dim=-1)
        .fill_diagonal_(np.nan)
        .cpu()
        .numpy()
    )
    cos = cos.fill_diagonal_(np.nan).cpu().numpy()
    matches = match_objects(obj_feats)

    fig, axs = subplots_grid(1, 2, ax_height_inch=2 * B, sharex=True, sharey=True)

    ax = axs[0]
    img = ax.imshow(cos)
    fig.colorbar(img, ax=ax)
    ax.set_title(
        f"Pairwise cos of slot projections\n"
        f"Min {np.nanmin(cos):.3f} Max {np.nanmax(cos):.3f} Loss {loss:.4f}"
    )
    ax.scatter(matches, np.arange(2 * B * S), color="red", s=10)

    ax = axs[1]
    img = ax.imshow(prob)
    fig.colorbar(img, ax=ax, format=PercentFormatter(1.0))
    ax.set_title(
        f"Match probs of slot projections\n"
        f"Min {np.nanmin(prob):.1%} Max {np.nanmax(prob):.1%}"
    )
    ax.scatter(matches, np.arange(2 * B * S), color="red", s=10)

    for ax in axs.flat:
        ax.axhline(B * S - 0.5, color="black", lw=4)
        ax.axvline(B * S - 0.5, color="black", lw=4)
        for line in range(S, 2 * B * S, S):
            ax.axhline(line - 0.5, color="black", lw=2)
            ax.axvline(line - 0.5, color="black", lw=2)

        ax.set_xlabel("Image idx | slot idx")
        ax.set_xticks(np.arange(2 * B * S))
        ax.set_xticklabels(
            2 * [k if k > 0 else f"img {b} | {k}" for b in range(B) for k in range(S)],
            rotation=90,
            fontdict={"fontsize": "small"},
        )

    ax = axs[0]
    ax.set_yticks(np.arange(2 * B * S))
    ax.set_yticklabels(
        2 * [s if s > 0 else f"img {b} | {s}" for b in range(B) for s in range(S)],
        fontdict={"fontsize": "small"},
    )
    ax.set_ylabel("Image idx | slot idx")

    fig.set_facecolor("white")
    fig.tight_layout()
    return fig


def viz_contrastive_loss_objects_probs(
    obj_feats: torch.Tensor, temp: float, loss: float
) -> Figure:
    """Visualize contrastive loss between 2B images with S objects each, probabilities only.

    Args:
        obj_feats: object feature tensor of shape ``[2B S C]``,
            where ``S`` is the number of objects per image.
        temp: loss temperature
        loss: loss value to display on the plots

    Returns:
        A figure with a single plot of matching probabilities.
    """
    B = obj_feats.shape[0] // 2
    S = obj_feats.shape[1]

    cos = cos_pairwise(obj_feats.detach().reshape(2 * B * S, -1))
    prob = (
        cos.div_(temp)
        .fill_diagonal_(-torch.inf)
        .softmax(dim=-1)
        .fill_diagonal_(np.nan)
        .cpu()
        .numpy()
    )

    fig, ax = plt.subplots(1, 1, figsize=(1.7 * 2 * B, 1.7 * 2 * B))

    img = ax.imshow(prob)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    fig.colorbar(img, cax=cax, format=PercentFormatter(1.0))

    matches = match_objects(obj_feats)
    ax.scatter(matches, np.arange(2 * B * S), color="red", s=10)
    ax.set_title(
        f"Match probs of slot projections (temp {temp:.2f})\n"
        f"Min {np.nanmin(prob):.1%} Max {np.nanmax(prob):.1%} Loss {loss:.4f}"
    )

    ax.axhline(B * S - 0.5, color="black", lw=4)
    ax.axvline(B * S - 0.5, color="black", lw=4)
    for line in range(S, 2 * B * S, S):
        ax.axhline(line - 0.5, color="black", lw=2)
        ax.axvline(line - 0.5, color="black", lw=2)

    ax.set_xlabel("Image idx | slot idx")
    ax.set_xticks(np.arange(2 * B * S))
    ax.set_xticklabels(
        2 * [k if k > 0 else f"img {b} | {k}" for b in range(B) for k in range(S)],
        rotation=90,
        fontdict={"fontsize": "small"},
    )

    ax.set_ylabel("Image idx | slot idx")
    ax.set_yticks(np.arange(2 * B * S))
    ax.set_yticklabels(
        2 * [s if s > 0 else f"img {b} | {s}" for b in range(B) for s in range(S)],
        fontdict={"fontsize": "small"},
    )

    fig.set_facecolor("white")
    fig.tight_layout()
    return fig
