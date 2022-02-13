"""
Visualization of positional embeddings.
"""
from typing import Tuple

import einops
import numpy as np
import torch
import torch.nn.functional
from matplotlib import pyplot as plt

from osc.utils import cos_pairwise


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
    cos = cos_pairwise(embed).reshape(*num_patches, *num_patches).cpu().numpy()

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
