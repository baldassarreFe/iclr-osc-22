"""
Attention visualization utils: single attn maps and rollouts.
"""
from itertools import product

import numpy as np
import torch
from einops import rearrange, reduce
from IPython.core.display import Image
from IPython.core.display_functions import display
from matplotlib import pyplot as plt

from .rollout import self_attn_rollout, slot_attn_rollout


def viz_vit_attns(images, attns_dict, head_reduction="mean", adjust_residual=True):
    A, B = images.shape[:2]
    images = images.detach().cpu().numpy()
    images = rearrange(images, "A B C H W -> B A H W C")

    L = len(attns_dict)
    attns = [attns_dict[k].detach() for k in sorted(attns_dict.keys())]
    attns = rearrange(attns, "L (A B) h Q_hw K_hw -> B A L h Q_hw K_hw", A=A, B=B)
    num_heads = attns.shape[3]
    K_h = K_w = int(np.sqrt(attns.shape[5]))

    # For each layer:
    # - sum over queries to find most attended key, average over heads
    # - sum over queries to find most attended key, for each head
    # - rearrange keys into a square image
    attns = torch.concat(
        [
            reduce(attns, "B A L h Q_hw K_hw -> B A L 1 K_hw", "sum"),
            reduce(attns, "B A L h Q_hw K_hw -> B A L h K_hw", "sum"),
        ],
        dim=3,
    )
    attns = rearrange(
        attns.cpu().numpy(), "B A L h (K_h K_w) -> B A L h K_h K_w", K_h=K_h, K_w=K_w
    )

    # Rollout
    rollout = self_attn_rollout(
        attns_dict,
        head_reduction=head_reduction,
        adjust_residual=adjust_residual,
        global_avg_pool=True,
    )
    rollout = rearrange(
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
    images = rearrange(images, "A B C H W -> B A H W C")

    attns = [slot_attns_dict[i].detach() for i in sorted(slot_attns_dict.keys())]
    attns = rearrange(
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
    attns = rearrange(
        attns,
        "B A I k (K_h K_w) -> B A I k K_h K_w",
        K_h=K_h,
        K_w=K_w,
    )
    attns = attns.cpu().numpy()

    # [A*B, Q_hw, K_hw]
    vit_rollout = self_attn_rollout(
        vit_attns_dict,
        head_reduction=head_reduction,
        adjust_residual=adjust_residual,
        global_avg_pool=False,
    )
    # [A*B, K, K_hw]
    slot_rollouts = slot_attn_rollout(slot_attns_dict)
    # [A*B, K, K_hw]
    full_rollouts = torch.einsum("bki,bij->bkj", slot_rollouts, vit_rollout)

    slot_rollouts = rearrange(
        slot_rollouts.cpu().numpy(),
        "(A B) k (K_h K_w) -> B A k K_h K_w",
        A=num_augs,
        B=batch_size,
        K_h=K_h,
        K_w=K_w,
    )
    full_rollouts = rearrange(
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


def viz_vit_rollout_all_options(images, attns_dict):
    """Visualize backbone rollout: residual adjustment, head reduction"""
    A, B = images.shape[:2]
    images = images.detach().cpu().numpy()
    images = rearrange(images, "A B C H W -> B A H W C")

    K_h = K_w = int(np.sqrt(list(attns_dict.values())[0].shape[-1]))

    fig, axs = plt.subplots(B * A, 1 + 4, figsize=2.5 * (np.array([1 + 4, B * A])))
    axs = rearrange(axs, "(B A) L -> B A L", B=B, A=A)

    for b, a in np.ndindex(B, A):
        axs[b, a, 0].imshow(images[b, a])
        axs[b, a, 0].set_ylabel(f"Img {b}\nAug {a}")

    for i, (adjust_residual, head_reduction) in enumerate(
        product([True, False], ["mean", "max"])
    ):
        rollout = self_attn_rollout(
            attns_dict,
            head_reduction=head_reduction,
            adjust_residual=adjust_residual,
            global_avg_pool=True,
        )
        rollout = rearrange(
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
    """Visualize slot attention rollout: per-iteration or final normalization."""
    A, B = images.shape[:2]
    images = images.detach().cpu().numpy()
    images = rearrange(images, "A B C H W -> B A H W C")

    K_h = K_w = int(np.sqrt(list(slot_attns_dict.values())[0].shape[-1]))
    S = list(slot_attns_dict.values())[0].shape[-2]

    fig, axs = plt.subplots(
        B * A * 4, 1 + S, figsize=2 * (np.array([1 + S, B * A * 4]))
    )
    axs = rearrange(axs, "(B A opt) Sp -> B A opt Sp", B=B, A=A, opt=4)

    for b, a in np.ndindex(B, A):
        axs[b, a, 0, 0].imshow(images[b, a])

    for opt, (full, normalize) in enumerate(product([False, True], ["layer", "all"])):
        slot_rollout = slot_attn_rollout(slot_attns_dict, normalize=normalize)
        if full:
            slot_rollout = torch.einsum("bsj,bjk->bsk", slot_rollout, vit_rollout)
        slot_rollout = rearrange(
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
