"""
Data augmentation functions.

All augmentations use stateless random seeds, a helper function can be used to
add a sequence of such seeds to any :class:`tf.data.Dataset` instance.
Augmentations include geometric and color jitter.
"""

from functools import partial

import tensorflow as tf
from tensorflow.keras.preprocessing.image import apply_affine_transform, smart_resize

from osc.data.utils import img_hwc_to_chw, normalize_tf


@tf.function
def augment_train(
    img: tf.Tensor,
    seed: tf.Tensor,
    large_boxes: tf.Tensor,
    large_num_crops: int,
    large_strength: float,
    large_crop_hw: (int, int),
    small_boxes: tf.Tensor,
    small_num_crops: int,
    small_strength: float,
    small_crop_hw: (int, int),
    normalize_mean: tf.Tensor,
    normalize_std: tf.Tensor,
) -> (tf.Tensor, tf.Tensor):
    """Augment image for train/val. Pass different seeds to get different augmentations.

    Augmentations applied:

    - uint8 [0-255] -> float32 [0-1]
    - crop using one of the provided boxes selected at random
    - resize to ``crop_size``
    - geometric augmentations:
        - DISABLED random flip left-to-right
        - shear and rotate
    - color augmentation:
        - brightness
        - contrast
        - saturation
        - hue (very little)
    - normalization to given ``mean`` and ``std``
    - order axes for pytorch ``[H W RGB] -> [RGB H W]``

    Args:
        img: :class:``tf.uint8`` image ``[H W RGB]``
        seed:
        large_boxes:
        large_num_crops:
        large_strength:
        large_crop_hw:
        small_boxes:
        small_num_crops:
        small_strength:
        small_crop_hw:
        normalize_mean:
        normalize_std:

    Returns:
        Large and small crop tensors, with shapes ``[num_crops, RGB, H, W]``,
        according to the respective parameters.
    """
    img = tf.image.convert_image_dtype(img, tf.float32)
    seeds = tf.random.experimental.stateless_split(seed, 2)

    large = _augment_train_helper(
        img=img,
        seed=seeds[0],
        boxes=large_boxes,
        num_crops=large_num_crops,
        strength=large_strength,
        crop_hw=large_crop_hw,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    if small_num_crops > 0:
        small = _augment_train_helper(
            img=img,
            seed=seeds[1],
            boxes=small_boxes,
            num_crops=small_num_crops,
            strength=small_strength,
            crop_hw=small_crop_hw,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
    else:
        small = tf.zeros([0, 3, *small_crop_hw])

    return large, small


@tf.function
def _augment_train_helper(
    img: tf.Tensor,
    seed: tf.Tensor,
    boxes: tf.Tensor,
    num_crops: int,
    strength: float,
    crop_hw: (int, int),
    normalize_mean: tf.Tensor,
    normalize_std: tf.Tensor,
) -> tf.Tensor:
    seeds = tf.random.experimental.stateless_split(seed, 1 + 2 * num_crops)
    imgs = crop_and_resize(img, seeds[0], boxes, num_crops, crop_hw)
    imgs = tf.map_fn(
        lambda x: geometric_augment(x[0], x[1], strength),
        (imgs, seeds[1 : 1 + num_crops]),
        fn_output_signature=tf.float32,
    )
    imgs = tf.map_fn(
        lambda x: color_augment(x[0], x[1], strength),
        (imgs, seeds[1 + num_crops :]),
        fn_output_signature=tf.float32,
    )
    imgs = normalize_tf(imgs, normalize_mean, normalize_std)
    imgs = img_hwc_to_chw(imgs)
    return imgs


@tf.function
def geometric_augment(img: tf.Tensor, seed: tf.Tensor, strength: float) -> tf.Tensor:
    """Random flip, shear, rotate.

    Args:
        img: single tf.float32 image [H W RGB]
        seed:
        strength: augmentation strength between 0 and 1

    Returns:

    """
    seeds = tf.random.experimental.stateless_split(seed, 2)
    # img = tf.image.stateless_random_flip_left_right(img, seeds[0])
    u = tf.random.stateless_uniform([2], seeds[1], minval=-1.0, maxval=1.0)
    return tf.numpy_function(
        lambda x, s, r: apply_affine_transform(x, shear=s, theta=r, channel_axis=2),
        [img, 15.0 * strength * u[0], 30.0 * strength * u[1]],
        tf.float32,
    )


@tf.function
def color_augment(img: tf.Tensor, seed: tf.Tensor, strength: float):
    """Saturation, brightness, contrast, and hue.

    Args:
        img: single tf.float32 image [H W RGB]
        seed:
        strength:

    Returns:

    """
    if strength == 0:
        return img
    brightness = 0.3 * strength
    ctr = 0.5 * strength
    sat = 0.5 * strength
    hue = 0.03 * strength
    seeds = tf.random.experimental.stateless_split(seed, 5)
    doit = tf.random.stateless_uniform([4], seeds[0], 0.0, 1.0) > 0.5
    if doit[0]:
        img = tf.image.stateless_random_saturation(img, 1 - sat, 1 + sat, seeds[1])
    if doit[1]:
        img = tf.image.stateless_random_brightness(img, brightness, seeds[2])
    if doit[2]:
        img = tf.image.stateless_random_contrast(img, 1 - ctr, 1 + ctr, seeds[3])
    if doit[3]:
        img = tf.image.stateless_random_hue(img, hue, seeds[4])
    img = tf.clip_by_value(img, 0, 1)
    return img


@tf.function
def crop_and_resize(
    img: tf.Tensor,
    seed: tf.Tensor,
    boxes_y0x0y1x1: tf.Tensor,
    num_crops: int,
    crop_hw: (int, int),
):
    """Crop and resize using randomly picked boxes.

    Args:
        img: tf.float32 image [H W C]
        seed:
        boxes_y0x0y1x1: a list of boxes to choose from
        num_crops:
        crop_hw: resize crops to this size

    Returns:

    """
    idxs = tf.random.stateless_uniform(
        [num_crops], seed, minval=0, maxval=len(boxes_y0x0y1x1), dtype=tf.int32
    )
    boxes_y0x0y1x1 = tf.gather(boxes_y0x0y1x1, idxs, axis=0)
    return tf.image.crop_and_resize(
        img[None, :, :, :],
        boxes_y0x0y1x1,
        tf.zeros(num_crops, dtype=tf.int32),
        crop_hw,
        method="bilinear",
    )


@tf.function
def augment_val(
    img: tf.Tensor,
    crop_hw: (int, int),
    normalize_mean: tf.Tensor,
    normalize_std: tf.Tensor,
) -> tf.Tensor:
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = smart_resize(img, crop_hw, interpolation="bilinear")
    img = (img - normalize_mean) / normalize_std
    img = tf.experimental.numpy.moveaxis(img, -1, -3)
    return img


def zip_with_seeds(
    ds: tf.data.Dataset, *seeds: int, num_parallel_calls=tf.data.AUTOTUNE
) -> tf.data.Dataset:
    """Zip a dataset with one or more random seed generators."""
    rng = [
        tf.data.Dataset.random(seed=s).map(
            partial(tf.random.experimental.create_rng_state, alg="threefry"),
            num_parallel_calls=num_parallel_calls,
            deterministic=True,
        )
        for s in seeds
    ]
    return tf.data.Dataset.zip((ds, *rng))
