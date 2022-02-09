from typing import Callable, Protocol, Tuple

import tensorflow as tf
import torch
import torch.nn.functional
import torch.utils.data

from osc.utils import ImgMean, ImgSizeHW, ImgStd

from .random_resized_crop import random_resized_crop


@tf.function
def normalize_tf(img: tf.Tensor, mean: ImgMean, std: ImgStd) -> tf.Tensor:
    """Normalize tf.float32 image with [..., H, W, C] channel order.

    Args:
        img: tf.float32 of shape [..., H, W, C]
        mean: tuple of floats shape [C]
        std: tuple of floats shape [C]

    Returns:
        Normalized image.
    """
    mean = tf.convert_to_tensor(mean)
    std = tf.convert_to_tensor(std)
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
def augment_train(
    img: tf.Tensor, *, seed: tf.Tensor, crop_size: ImgSizeHW, mean: ImgMean, std: ImgStd
) -> tf.Tensor:
    """Augment image for train/val. Pass different seeds to get different augmentations.

    Augmentations applied:

    - uint8 [0-255] -> float32 [0-1]
    - random flip
    - random resized crop (random aspect ratio, zoom factor, resize to ``crop_size``
    - color jitter (brightness, saturation, hue)
    - normalization to given ``mean`` and ``std``
    - pytorch channel order [H W C] -> [C H W]
    """
    img = tf.image.convert_image_dtype(img, tf.float32)

    seeds = tf.random.experimental.stateless_split(seed, num=5)
    img = tf.image.stateless_random_flip_left_right(img, seeds[0])
    img = random_resized_crop(img, size=crop_size, scale=(0.3, 1.0), seed=seeds[1])
    img = tf.image.stateless_random_brightness(img, 0.2, seeds[2])
    img = tf.image.stateless_random_saturation(img, 0.9, 1.0, seeds[3])
    img = tf.image.stateless_random_hue(img, 0.05, seeds[4])

    img = normalize_tf(img, mean, std)
    img = img_hwc_to_chw(img)

    return img


@tf.function
def augment_center_crop(
    img: tf.Tensor, *, crop: float, img_size: ImgSizeHW, mean: ImgMean, std: ImgStd
) -> tf.Tensor:
    """Deterministic center crop and processing as in :func:`augment_train`

    Args:
        img: TF uint8 tensor of shape [H W C]
        crop:
        img_size:
        mean:
        std:

    Returns:

    """
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.central_crop(img, crop)
    img = tf.image.resize(img, img_size, method="bilinear", antialias=True)

    img = normalize_tf(img, mean, std)
    img = img_hwc_to_chw(img)

    return img


AugmentFn = Callable[
    [
        tf.Tensor,
    ],
    tf.Tensor,
]


def augment_twice(augment_fn0: AugmentFn, augment_fn1: AugmentFn):
    """Augment image twice with two different functions.

    Args:
        augment_fn0: callable, augment_fn0(img)
        augment_fn1: callable, augment_fn1(img)

    Returns:
        Callable, augment_twice(fn0, fn1)(img) -> (fn0(img), fn1(img))
    """

    @tf.function
    def twice(img: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return augment_fn0(img), augment_fn1(img)

    return twice


class AugmentFnStateless(Protocol):
    """Function type signature for TF stateless random."""

    def __call__(self, img: tf.Tensor, *, seed: tf.Tensor) -> tf.Tensor:
        ...


def wrap_with_seed(
    augment_fn: AugmentFnStateless, *, initial_seed: int = 0
) -> AugmentFn:
    """Wrap stateless random functions by providing a new seed to each invocation.

    Args:
        augment_fn: callable, augment_fn(img, seed)
        initial_seed: initial seed for the generator.

    Returns:
        Callable that wraps ``augment_fn``.
    """
    rng = tf.random.Generator.from_seed(initial_seed)

    @tf.function
    def wrapped(img: tf.Tensor) -> tf.Tensor:
        seed = rng.make_seeds(count=1)[:, 0]
        return augment_fn(img, seed=seed)

    return wrapped
