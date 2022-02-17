"""
Visualization of global losses.
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

from osc.utils import cos_pairwise

from .utils import subplots_grid


def viz_contrastive_loss_global(global_feats: Tensor, loss: float) -> Figure:
    """Visualize global contrastive loss between 2B images.

    Args:
        global_feats: a tensor with shape [2B, C]
        loss: loss value to show on the plot

    Returns:
        A figure with two axes.
    """
    A = 2
    B = global_feats.shape[0] // A

    # cos, prob: [AB AB]
    cos = cos_pairwise(global_feats.detach().reshape(A * B, -1))
    prob = (
        torch.clone(cos)
        .fill_diagonal_(-torch.inf)
        .softmax(dim=-1)
        .fill_diagonal_(np.nan)
        .cpu()
        .numpy()
    )
    acc = np.mean(np.nanargmax(prob, axis=1) == np.roll(np.arange(A * B), B))
    cos = cos.fill_diagonal_(np.nan).cpu().numpy()

    fig, axs = subplots_grid(1, 2, ax_height_inch=6, sharex=True, sharey=True)

    ax = axs[0]
    img = ax.imshow(cos)
    fig.colorbar(img, ax=ax)
    ax.scatter(np.nanargmax(cos, axis=1), np.arange(2 * B), color="red")
    ax.set_title(
        f"Pairwise cos of global projections\n"
        f"Min {np.nanmin(cos):.3f} Max {np.nanmax(cos):.3f} Loss {loss:.4f}"
    )

    ax = axs[1]
    img = ax.imshow(prob)
    fig.colorbar(img, ax=ax, format=PercentFormatter(1.0))
    ax.scatter(np.nanargmax(prob, axis=1), np.arange(2 * B), color="red")
    ax.set_title(
        f"Match probs of global projections\n"
        f"Min {np.nanmin(prob):.1%} Max {np.nanmax(prob):.1%} Acc {acc:.2%}"
    )

    for ax in axs.flat:
        ax.axhline(B - 0.5, color="black", lw=4, ls="--")
        ax.axvline(B - 0.5, color="black", lw=4, ls="--")

        ax.set_xticks(np.arange(2 * B))
        ax.set_xticklabels(np.tile(np.arange(B), 2))
        ax.set_xlabel("Image idx")

    ax = axs[0]
    ax.set_yticks(np.arange(2 * B))
    ax.set_yticklabels(np.tile(np.arange(B), 2))
    ax.set_ylabel("Image idx")

    fig.tight_layout()
    return fig


def viz_contrastive_loss_global_probs(
    global_feats: Tensor, temp: float, loss: float
) -> Figure:
    """Visualize global contrastive loss between 2B images, probabilities only.

    Args:
        global_feats: a tensor with shape [2B, C]
        temp: loss temperature
        loss: loss value to show on the plot

    Returns:
        A figure with a single plot of matching probabilities.
    """
    A = 2
    B = global_feats.shape[0] // A

    # cos, prob: [AB AB]
    cos = cos_pairwise(global_feats.detach().reshape(A * B, -1))
    prob = (
        cos.div_(temp)
        .fill_diagonal_(-torch.inf)
        .softmax(dim=-1)
        .fill_diagonal_(np.nan)
        .cpu()
        .numpy()
    )
    acc = np.mean(np.nanargmax(prob, axis=1) == np.roll(np.arange(A * B), B))

    fig, ax = plt.subplots(1, 1, figsize=(1.7 * 2 * B, 1.7 * 2 * B))

    img = ax.imshow(prob)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    fig.colorbar(img, cax=cax, format=PercentFormatter(1.0))
    ax.scatter(np.nanargmax(prob, axis=1), np.arange(2 * B), color="red")
    ax.set_title(
        f"Match probs of global projections (temp {temp:.2f})\n"
        f"Min {np.nanmin(prob):.1%} Max {np.nanmax(prob):.1%} "
        f"Acc {acc:.2%} Loss {loss:.4f}"
    )

    ax.axhline(B - 0.5, color="black", lw=4, ls="-")
    ax.axvline(B - 0.5, color="black", lw=4, ls="-")

    ax.set_xticks(np.arange(2 * B))
    ax.set_xticklabels(np.tile(np.arange(B), 2))
    ax.set_xlabel("Image idx")

    ax.set_yticks(np.arange(2 * B))
    ax.set_yticklabels(np.tile(np.arange(B), 2))
    ax.set_ylabel("Image idx")

    fig.tight_layout()
    return fig
