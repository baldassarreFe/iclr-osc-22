"""
Visualization utils.
"""
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import skimage.filters
import torch
from IPython.core.display import Image
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image as PilImage


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
    """Create a grid of subplots with identical aspect ratio.

    Args:
        nrows:
        ncols:
        ax_aspect_hw:
        ax_height_inch:
        dpi:
        **kwargs:

    Returns:
        A figure and an array of axes (possibly squeezed).
    """
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


def batched_otsu(x: np.ndarray):
    result = np.empty_like(x)
    for b in np.ndindex(x.shape[:-2]):
        result[b] = x[b] > skimage.filters.threshold_otsu(x[b])
    return result
