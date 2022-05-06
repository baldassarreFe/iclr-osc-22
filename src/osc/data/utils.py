"""
Helper functions to bridge tensorflow image data to pytorch.

Functions:

- Image normalization back and forth
- Axis order
- Largest center crop
"""


import tensorflow as tf
import torch
import torch.nn.functional
import torch.utils.data

from osc.utils import ImgMean, ImgSizeHW, ImgStd


@tf.function
def normalize_tf(img: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
    """Normalize tf.float32 image with [..., H, W, C] channel order.

    Args:
        img: tf.float32 of shape [..., H, W, C]
        mean: tf.float32 of shape [C]
        std: tf.float32 of shape [C]

    Returns:
        Normalized image.
    """
    return (img - mean) / std


def unnormalize_pt(img: torch.Tensor, mean: ImgMean, std: ImgStd) -> torch.Tensor:
    """Un-normalize torch.float32 image with [..., C, H, W] channel order.

    Args:
        img: torch.float32 of shape [..., C, H, W]
        mean: tuple of floats shape [C]
        std: tuple of floats shape [C]

    Returns:
        Un-normalized image with values in range [0, 1].
    """
    mean = torch.tensor(mean).to(img.device)
    std = torch.tensor(std).to(img.device)
    img = img * std[:, None, None] + mean[:, None, None]
    return torch.clip(img, 0, 1)


@tf.function
def img_hwc_to_chw(img: tf.Tensor) -> tf.Tensor:
    """Image channels [..., H, W, C] -> [..., C, H, W]"""
    return tf.experimental.numpy.moveaxis(img, -1, -3)


@tf.function
def img_chw_to_hwc(img: tf.Tensor) -> tf.Tensor:
    """Image channels [..., C, H, W] -> [..., H, W, C]"""
    return tf.experimental.numpy.moveaxis(img, -3, -1)


@tf.function
def largest_center_crop(img: tf.Tensor, *, crop_size: ImgSizeHW) -> tf.Tensor:
    """Crop the largest possible square from image center and resize to desired size.

    It's probably equivalent to :func:`tf.keras.preprocessing.image.smart_resize`.

    Args:
        img: tf.float32 tensor of shape ``[H W C]``
        crop_size: expected output size, for each edge.

    Returns:
        A tf.float32 tensor of shape ``[*crop_size, C]``.
    """

    # Same as `H, W = img.shape[:2]` but works in graph mode
    shape = tf.shape(img)[:2]
    H = shape[0]
    W = shape[1]
    S = tf.reduce_min(shape)
    y0 = (H - S) // 2
    x0 = (W - S) // 2
    y1 = (H + S) // 2
    x1 = (W + S) // 2

    img = img[y0:y1, x0:x1, :]
    img = tf.image.resize(img, crop_size, method="bilinear", antialias=True)
    return img
