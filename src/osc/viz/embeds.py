"""
Visualization of positional embeddings.
"""
from typing import Tuple

import numpy as np
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor

from osc.utils import cos_pairwise


def viz_positional_embedding(
    embed: Tensor,
    res_HW: Tuple[int, int] = None,
    side_inch=1.0,
) -> Figure:
    """Visualize 2D positional embedding, possibly resized to a different grid size.

    Args:
        embed: embedding of shape ``[H W D]``
        res_HW: optionally, a tuple ``(H, W)`` of dimensions to resize the embedding
        side_inch: height of one embedding in the resulting figure

    Returns:
        A figure containing ``[H W]`` heatmaps of cosine similarity.
    """
    H, W, D = embed.shape
    embed = embed.detach()
    title = f"Positional embedding ({H}, {W})"

    # Optionally resize 2D positional embedding
    if res_HW is not None:
        embed = rearrange(embed, "H W D -> 1 D H W")
        embed = F.interpolate(embed, size=res_HW, mode="bicubic", align_corners=False)
        embed = rearrange(embed, "1 D H W -> H W D")
        H, W = res_HW
        title += f" ->  ({H}, {W})"

    # [H W H W]
    cos = cos_pairwise(embed).cpu().numpy()

    fig: plt.Figure
    fig, axs = plt.subplots(H, W, figsize=side_inch * np.array([W, H]))
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
