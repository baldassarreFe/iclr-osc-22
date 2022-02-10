"""
Visualization utils.
"""
from collections import defaultdict
from itertools import product
from operator import itemgetter
from pathlib import Path
from typing import Sequence, Tuple, Union

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as PilImage
import scipy.optimize
import scipy.stats
import skimage.filters
import sklearn.cluster
import torch
import torch.nn.functional
from IPython.display import Image, display
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

import osc.data.clevr_with_masks
import osc.rollout
import osc.utils
from osc.rollout import slot_attn_rollout


def viz_history(history):
    duration_sec = max(map(itemgetter("time"), history), default=-1)
    num_steps = max(map(itemgetter("step"), history), default=0)
    print(f"Total training time: {duration_sec/60:.1f} minutes")
    print(f"Average speed: {num_steps/duration_sec:.2f} batches/second")

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    history = history_groupby_name(history)

    for name, ax in zip(["l_global", "l_objects"], axs):
        steps, values = zip(*map(itemgetter("step", "value"), history[f"{name}/train"]))
        values = pd.Series(values).ewm(alpha=0.1).mean().values
        ax.plot(steps, values, label="train")

        steps, values = zip(*map(itemgetter("step", "value"), history[f"{name}/val"]))
        ax.plot(steps, values, label="val")

        ax.set_title(name)
        ax.legend()

    name = "lr"
    ax = axs[-1]
    steps, values = zip(*map(itemgetter("step", "value"), history["lr"]))
    ax.plot(steps, values)
    ax.set_title(name)

    epoch_markers = [h["step"] for h in history["l_global/val"]]
    for ax in axs:
        ax.grid(True, axis="y")
        for e in epoch_markers:
            ax.axvline(e, lw=0.1, color="black")

    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig("history.png", dpi=200)
    plt.close(fig)
    display(Image(url="history.png", width=1000))


def history_groupby_name(history):
    history_dict = defaultdict(list)
    for h in history:
        history_dict[h["name"]].append(h)
    return history_dict


def viz_positional_embedding(
    embed: torch.Tensor,
    num_patches: Tuple[int, int],
    target_patches: Tuple[int, int] = None,
    side_inch=1.0,
):
    assert embed.shape[0] == np.prod(num_patches)
    assert embed.ndim == 2
    embed = embed.detach()
    title = f"Positional embedding {tuple(num_patches)}"

    # Optionally resize 2D positional embedding
    if target_patches is not None:
        embed = einops.rearrange(
            embed, "(P_h P_w) C -> 1 C P_h P_w ", P_h=num_patches[0], P_w=num_patches[0]
        )
        embed = torch.nn.functional.interpolate(
            embed, size=target_patches, mode="bilinear", align_corners=False
        )
        embed = einops.rearrange(embed, "1 C P_h P_w -> (P_h P_w) C")
        title += f" -> {tuple(target_patches)}"
        num_patches = target_patches

    # [P_h, P_w, P_h, P_w]
    cos = (
        osc.utils.cos_pairwise(embed).reshape(*num_patches, *num_patches).cpu().numpy()
    )

    fig: plt.Figure
    fig, axs = plt.subplots(
        *num_patches, figsize=side_inch * np.array(num_patches[::-1])
    )
    for (h, w), ax in np.ndenumerate(axs):
        ax.imshow(cos[h, w])
        ax.set_xticks([])
        ax.set_yticks([])
    for h, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(f"{h} ", rotation=0)
    for w, ax in enumerate(axs[-1, :]):
        ax.set_xlabel(str(w))
    fig.set_facecolor("white")
    fig.suptitle(title)
    fig.tight_layout(h_pad=0.0, w_pad=0.0)
    return fig


def viz_contrastive_loss_global(global_feats: torch.Tensor, loss: float) -> Figure:
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
    cos = osc.utils.cos_pairwise(global_feats.detach().reshape(A * B, -1))
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
    global_feats: torch.Tensor, temp: float, loss: float
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
    cos = osc.utils.cos_pairwise(global_feats.detach().reshape(A * B, -1))
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

    cos = osc.utils.cos_pairwise(obj_feats.detach().reshape(2 * B * S, -1))
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

    cos = osc.utils.cos_pairwise(obj_feats.detach().reshape(2 * B * S, -1))
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


def viz_vit_rollout_all_options(images, attns_dict):
    A, B = images.shape[:2]
    images = images.detach().cpu().numpy()
    images = einops.rearrange(images, "A B C H W -> B A H W C")

    K_h = K_w = int(np.sqrt(list(attns_dict.values())[0].shape[-1]))

    fig, axs = plt.subplots(B * A, 1 + 4, figsize=2.5 * (np.array([1 + 4, B * A])))
    axs = einops.rearrange(axs, "(B A) L -> B A L", B=B, A=A)

    for b, a in np.ndindex(B, A):
        axs[b, a, 0].imshow(images[b, a])
        axs[b, a, 0].set_ylabel(f"Img {b}\nAug {a}")

    for i, (adjust_residual, head_reduction) in enumerate(
        product([True, False], ["mean", "max"])
    ):
        rollout = osc.rollout.self_attn_rollout(
            attns_dict,
            head_reduction=head_reduction,
            adjust_residual=adjust_residual,
            global_avg_pool=True,
        )
        rollout = einops.rearrange(
            rollout.cpu().numpy(),
            "(A B) (K_h K_w) -> B A K_h K_w",
            A=A,
            B=B,
            K_h=K_h,
            K_w=K_w,
        )

        axs[0, 0, i + 1].set_title(f"{head_reduction} residual={adjust_residual}")
        for b, a in np.ndindex(B, A):
            axs[b, a, i + 1].imshow(rollout[b, a])

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.set_facecolor("white")
    return fig


def viz_slot_rollout_all_options(images, slot_attns_dict, vit_rollout):
    A, B = images.shape[:2]
    images = images.detach().cpu().numpy()
    images = einops.rearrange(images, "A B C H W -> B A H W C")

    K_h = K_w = int(np.sqrt(list(slot_attns_dict.values())[0].shape[-1]))
    S = list(slot_attns_dict.values())[0].shape[-2]

    fig, axs = plt.subplots(
        B * A * 4, 1 + S, figsize=2 * (np.array([1 + S, B * A * 4]))
    )
    axs = einops.rearrange(axs, "(B A opt) Sp -> B A opt Sp", B=B, A=A, opt=4)

    for b, a in np.ndindex(B, A):
        axs[b, a, 0, 0].imshow(images[b, a])

    for opt, (full, normalize) in enumerate(product([False, True], ["layer", "all"])):
        slot_rollout = slot_attn_rollout(slot_attns_dict, normalize=normalize)
        if full:
            slot_rollout = torch.einsum("bsj,bjk->bsk", slot_rollout, vit_rollout)
        slot_rollout = einops.rearrange(
            slot_rollout.cpu().numpy(),
            "(A B) S (K_h K_w) -> B A S K_h K_w",
            A=A,
            B=B,
            K_h=K_h,
            K_w=K_w,
        )
        for b, a in np.ndindex(B, A):
            for s in range(S):
                axs[b, a, opt, s + 1].imshow(slot_rollout[b, a, s])
            axs[b, a, opt, 0].set_ylabel(f'{normalize}\n{"full" if full else "slot"}')

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.set_facecolor("white")
    return fig


def viz_vit_attns(images, attns_dict, head_reduction="mean", adjust_residual=True):
    A, B = images.shape[:2]
    images = images.detach().cpu().numpy()
    images = einops.rearrange(images, "A B C H W -> B A H W C")

    L = len(attns_dict)
    attns = [attns_dict[k].detach() for k in sorted(attns_dict.keys())]
    attns = einops.rearrange(
        attns, "L (A B) h Q_hw K_hw -> B A L h Q_hw K_hw", A=A, B=B
    )
    num_heads = attns.shape[3]
    K_h = K_w = int(np.sqrt(attns.shape[5]))

    # For each layer:
    # - sum over queries to find most attended key, average over heads
    # - sum over queries to find most attended key, for each head
    # - rearrange keys into a square image
    attns = torch.concat(
        [
            einops.reduce(attns, "B A L h Q_hw K_hw -> B A L 1 K_hw", "sum"),
            einops.reduce(attns, "B A L h Q_hw K_hw -> B A L h K_hw", "sum"),
        ],
        dim=3,
    )
    attns = einops.rearrange(
        attns.cpu().numpy(), "B A L h (K_h K_w) -> B A L h K_h K_w", K_h=K_h, K_w=K_w
    )

    # Rollout
    rollout = osc.rollout.self_attn_rollout(
        attns_dict,
        head_reduction=head_reduction,
        adjust_residual=adjust_residual,
        global_avg_pool=True,
    )
    rollout = einops.rearrange(
        rollout.cpu().numpy(),
        "(A B) (K_h K_w) -> B A K_h K_w",
        A=A,
        B=B,
        K_h=K_h,
        K_w=K_w,
    )

    # One figure per pair of augmented images
    for b in range(B):
        fig, axs = plt.subplots(
            1 + L + 1,
            A * (num_heads + 1),
            figsize=2
            * np.array(
                [
                    0.2 + A * (num_heads + 1),
                    0.5 + 1 + L + 1,
                ]
            ),
        )

        # Top row: images (leftmost only, not repeated)
        axs_img = axs[0, :].reshape(A, num_heads + 1)
        axs_img[0, 0].set_ylabel("Input")
        for a in range(A):
            ax = axs_img[a, 0]
            ax.imshow(images[b, a])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Aug {a}")
        for ax in axs_img[:, 1:].flat:
            ax.set_axis_off()

        # Mid rows: one layer per row, avg all heads 1st column, then individual heads
        axs_heads = axs[1:-1, :].reshape(L, A, num_heads + 1)
        for lyr in range(L):
            axs_heads[lyr, 0, 0].set_ylabel(f"Layer {lyr}")

            for a in range(A):
                for h in range(num_heads + 1):
                    ax = axs_heads[lyr, a, h]
                    ax.imshow(
                        attns[b, a, lyr, h], cmap="inferno" if h == 0 else "viridis"
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if lyr == 0:
                        ax.set_title("Avg heads" if h == 0 else f"Head {h-1}")

        # Bottom row: rollout (leftmost only, not repeated)
        axs_roll = axs[-1, :].reshape(A, num_heads + 1)
        axs_roll[0, 0].set_ylabel("Rollout")
        for a in range(A):
            ax = axs_roll[a, 0]
            ax.imshow(rollout[b, a], cmap="inferno")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axs_roll[:, 1:].flat:
            ax.set_axis_off()

        fig.suptitle(f"Image {b}")
        fig.tight_layout()
        fig.set_facecolor("white")
        fig.savefig(f"img-{b}-vit-attns.png", dpi=200)
        plt.close(fig)
        display(Image(url=f"img-{b}-vit-attns.png", width=1400))


def viz_slot_attns(
    images, vit_attns_dict, slot_attns_dict, head_reduction="mean", adjust_residual=True
):
    num_augs = images.shape[0]
    batch_size = images.shape[1]
    images = images.detach().cpu().numpy()
    images = einops.rearrange(images, "A B C H W -> B A H W C")

    attns = [slot_attns_dict[i].detach() for i in sorted(slot_attns_dict.keys())]
    attns = einops.rearrange(
        attns,
        "I (A B) k K_hw -> B A I k K_hw",
        A=num_augs,
        B=batch_size,
    )
    attns = attns + 1e-8
    attns = attns / attns.sum(axis=-1, keepdims=True)
    num_iters = attns.shape[2]
    num_slots = attns.shape[3]

    K_h = K_w = int(np.sqrt(attns.shape[-1]))
    attns = einops.rearrange(
        attns,
        "B A I k (K_h K_w) -> B A I k K_h K_w",
        K_h=K_h,
        K_w=K_w,
    )
    attns = attns.cpu().numpy()

    # [A*B, Q_hw, K_hw]
    vit_rollout = osc.rollout.self_attn_rollout(
        vit_attns_dict,
        head_reduction=head_reduction,
        adjust_residual=adjust_residual,
        global_avg_pool=False,
    )
    # [A*B, K, K_hw]
    slot_rollouts = slot_attn_rollout(slot_attns_dict)
    # [A*B, K, K_hw]
    full_rollouts = torch.einsum("bki,bij->bkj", slot_rollouts, vit_rollout)

    slot_rollouts = einops.rearrange(
        slot_rollouts.cpu().numpy(),
        "(A B) k (K_h K_w) -> B A k K_h K_w",
        A=num_augs,
        B=batch_size,
        K_h=K_h,
        K_w=K_w,
    )
    full_rollouts = einops.rearrange(
        full_rollouts.cpu().numpy(),
        "(A B) k (K_h K_w) -> B A k K_h K_w",
        A=num_augs,
        B=batch_size,
        K_h=K_h,
        K_w=K_w,
    )

    for b in range(batch_size):

        fig, axs = plt.subplots(
            1 + num_iters + 1 + 1,
            num_augs * num_slots,
            figsize=2 * np.array([num_augs * num_slots, 0.5 + 1 + num_iters + 1 + 1]),
            squeeze=False,
        )

        # Top row: images
        axs_img = axs[0, :].reshape(num_augs, num_slots)
        for a in range(num_augs):
            ax = axs_img[a, 0]
            ax.imshow(images[b, a])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Aug {a}")
        for ax in axs_img[:, 1:].flat:
            ax.set_axis_off()

        # Mid rows: slot attn iterations
        axs_slt = axs[1:-2, :].reshape(num_iters, num_augs, num_slots)
        for i in range(num_iters):
            axs_slt[i, 0, 0].set_ylabel(f"Iteration {i}")

            for a in range(num_augs):
                # vmin = attns[b, a, i, :].min().item()
                # vmax = attns[b, a, i, :].max().item()
                # print(b,a,i,vmin,vmax)
                for k in range(num_slots):
                    ax = axs_slt[i, a, k]
                    # ax.imshow(attns[b, a, i, k], vmin=vmin, vmax=vmax)
                    ax.imshow(attns[b, a, i, k])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title(f"Slot {k}")

        # Bottom row 1: slot rollout for each slot
        axs_roll = axs[-2, :].reshape(num_augs, num_slots)
        axs_roll[0, 0].set_ylabel("Slot rollouts")
        for a in range(num_augs):
            for k in range(num_slots):
                ax = axs_roll[a, k]
                ax.imshow(slot_rollouts[b, a, k], cmap="inferno")
                ax.set_xticks([])
                ax.set_yticks([])

        # Bottom row 2: full rollout for each slot
        axs_roll = axs[-1, :].reshape(num_augs, num_slots)
        axs_roll[0, 0].set_ylabel("Full rollouts")
        for a in range(num_augs):
            for k in range(num_slots):
                ax = axs_roll[a, k]
                ax.imshow(full_rollouts[b, a, k], cmap="inferno")
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(f"Image {b}")
        fig.tight_layout()
        fig.set_facecolor("white")
        fig.savefig(f"img-{b}-slot-attns.png", dpi=200)
        plt.close(fig)
        display(Image(url=f"img-{b}-slot-attns.png", width=1700))


def kmeans_backbone(images, f_backbone, num_patches: Tuple[int, int]):
    assert f_backbone.shape[1] == np.prod(num_patches)
    num_augs = images.shape[0]
    batch_size = images.shape[1]

    f_backbone = einops.rearrange(
        f_backbone.detach().cpu().numpy(),
        "(A B) P_hw C -> B (A P_hw) C",
        A=num_augs,
        B=batch_size,
    )
    images = einops.rearrange(
        images.detach().cpu().numpy(),
        "A B C H W -> B A H W C",
    )

    for b in range(batch_size):
        fig, axs = plt.subplots(2, num_augs, figsize=np.array(num_patches[::-1]) / 2)

        kmeans = sklearn.cluster.KMeans(
            init="k-means++", n_clusters=11, n_init=4, random_state=0
        )
        clust = kmeans.fit_predict(f_backbone[b]).reshape(num_augs, *num_patches)
        for a in range(num_augs):
            axs[0, a].imshow(images[b, a])
            axs[0, a].set_title(f"Aug {a}")
            axs[1, a].imshow(clust[a], cmap="tab20")

        for ax in axs[:-1, :].flat:
            ax.set_xticks([])
        for ax in axs[:, 1:].flat:
            ax.set_yticks([])

        fig.suptitle(f"Image {b} - backbone features K-Means")
        fig.tight_layout()
        fig.set_facecolor("white")
        fig.savefig(f"img-{b}-vit-kmeans.png", dpi=200)
        plt.close(fig)
        display(Image(url=f"img-{b}-vit-kmeans.png", width=600))


def array_to_pil(
    img: Union[np.ndarray, torch.Tensor], cmap: str = None, scale_range=True
) -> PilImage.Image:
    """Array or tensor to PIL image. Works for both grayscale and RGB images.

    Args:
        img: [H W] or [H W C] image, dtype uint8 or float
        cmap: colormap for grayscale images
        scale_range: if True, rescale a grayscale image to cover the full color range

    Returns:
        A PIL image.
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    if img.ndim == 3:
        if any(img.dtype == t for t in [np.float, np.float32, np.float64]):
            img = np.cast[np.uint8](img * 255)
        if img.dtype == np.uint8:
            return PilImage.fromarray(img)
        raise ValueError(img.dtype)

    if img.ndim == 2:
        if any(img.dtype == t for t in [np.float, np.float32, np.float64]):
            if scale_range:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = plt.get_cmap(cmap)(img)[..., :3]  # RGB only, no RGBA
            img = np.cast[np.uint8](img * 255)
        if img.dtype == np.uint8:
            return PilImage.fromarray(img)
        raise ValueError(img.dtype)

    raise ValueError(img.shape)


def img_html(src, width=None, height=None, style="", tooltip=None):
    if width is not None:
        width = f'width="{width}"'
    if height is not None:
        height = f'height="{height}"'
    if tooltip is not None:
        tooltip = f'title="{tooltip}"'
    return f'<img src="{src}" {tooltip} {width} {height} style="{style}" alt="{src}">'


def text_html(text, rot=0, align="center"):
    return (
        f'<div style="text-align: {align}; transform: rotate({rot}deg);">{text}<div/>'
    )


def batched_otsu(x: np.ndarray):
    result = np.empty_like(x)
    for b in np.ndindex(x.shape[:-2]):
        result[b] = x[b] > skimage.filters.threshold_otsu(x[b])
    return result


def kmeans_clusters(
    x: Union[torch.Tensor, np.ndarray], n_clusters: int = 11
) -> np.ndarray:
    """Batched K-means clustering.

    Args:
        x: N samples of C-dimensional features with leading batch dimensions,
           e.g. shape [..., N, C]
        n_clusters: desired number of clusters

    Returns:
        int array of cluster IDs, shape [..., N, C]
    """

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    res = np.empty(x.shape[:-1], dtype=int)
    for b in np.ndindex(x.shape[:-2]):
        kmeans = sklearn.cluster.KMeans(
            init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0
        )
        res[b] = kmeans.fit_predict(x[b])

    return res


def make_grid_pil(grid: Sequence[Sequence[PilImage.Image]]):
    if len(set(img.size for row in grid for img in row)) != 1:
        raise ValueError("Grid images must all have the same size")
    ncols = max(len(row) for row in grid)
    grid = [[np.asarray(img) for img in row] for row in grid]
    grid = [row + (ncols - len(row)) * [np.zeros_like(row[0])] for row in grid]
    grid = np.concatenate([np.concatenate(row, axis=1) for row in grid], axis=0)
    return PilImage.fromarray(grid)


def subplots_grid(
    nrows: int = 1,
    ncols: int = 1,
    ax_aspect_hw: Tuple[int, int] = (1, 1),
    ax_height_inch: float = 4.0,
    dpi: int = 200,
    **kwargs,
) -> Tuple[Figure, NDArray[Axis]]:
    figsize_wh = (
        ax_height_inch * ncols * ax_aspect_hw[1] / ax_aspect_hw[0],
        ax_height_inch * nrows,
    )
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize_wh, dpi=dpi, **kwargs)
    fig.set_facecolor("white")
    return fig, axs


def remove_xyticks(axs: NDArray[Axis], keep_bottom_left=True):
    if keep_bottom_left:
        axs = axs[:-1, 1:]
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])


def fig_save_display(fig: Figure, path: Union[str, Path], dpi=100, width=400):
    fig.set_facecolor("white")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    display(Image(url=path, width=width))
