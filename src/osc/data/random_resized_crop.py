from typing import Tuple

import tensorflow as tf

from osc.utils import ImgSizeHW

ATTEMPTS = (10,)


@tf.function
def get_params(
    H: float,
    W: float,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    seed: Tuple[int, int],
):
    seeds = tf.random.experimental.stateless_split(seed, num=4)

    A = W * H
    target_area = A * tf.random.stateless_uniform(
        ATTEMPTS, seeds[0], minval=scale[0], maxval=scale[1]
    )

    log_ratio = tf.math.log(ratio)
    target_ratio = tf.exp(
        tf.random.stateless_uniform(
            ATTEMPTS, seeds[1], minval=log_ratio[0], maxval=log_ratio[1]
        )
    )

    h = tf.sqrt(target_area / target_ratio)
    w = tf.sqrt(target_area * target_ratio)
    ijhw = tf.stack(
        [
            tf.random.stateless_uniform(ATTEMPTS, seeds[2], minval=0, maxval=H - h + 1),
            tf.random.stateless_uniform(ATTEMPTS, seeds[3], minval=0, maxval=W - w + 1),
            h,
            w,
        ],
        axis=1,
    )
    fallback_ijhw = [
        # Center crop, valid for portrait images with ratio < min(ratio)
        [(H - W / ratio[0]) / 2, 0, W / ratio[0], W],
        # Center crop, valid for landscape images with ratio > max(ratio)
        [0, W - H * ratio[1], H, H * ratio[1]],
        # No crop, always valid
        [0, 0, H, W],
    ]
    ijhw = tf.concat([ijhw, fallback_ijhw], axis=0)
    valid = (0 < ijhw[:, 2]) & (ijhw[:, 2] <= H) & (0 < ijhw[:, 3]) & (ijhw[:, 3] <= W)

    idx = tf.argmax(valid)
    return ijhw[idx, :]


@tf.function
def random_resized_crop(
    img: tf.Tensor,
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    size: ImgSizeHW = (64, 64),
    seed: Tuple[int, int] = (0, 0),
):
    """Random resized crop as in torchvision.

    Args:
        img: a tf.float32 image with shape ``[H, W, C]``
        scale: lower and upper bounds for the random area of the crop, before resizing,
            relative to the original area of the image
        ratio: lower and upper bounds for the random aspect ratio of the crop,
            before resizing. Ratio>1 means horizontal landscape crop.
        size: expected output size of the crop, for each edge.
        seed: the seed for the random augmentation.

    Returns:
        A tf.float32 image with shape ``[*size, C]``
    """
    # Same as `H, W = img.shape[:2]` but works in graph mode
    shape = tf.cast(tf.shape(img), tf.float32)
    H = shape[0]
    W = shape[1]

    # Get crop coordinates: (y0, x0, height, width) in image coordinates
    ijhw = get_params(H, W, scale, ratio, seed)

    # Prepare box: (y0, x0, y1, x1) in normalized coordinates
    y0x0y1x1 = tf.concat([ijhw[:2], ijhw[:2] + ijhw[2:]], axis=0)
    y0x0y1x1 /= [H, W, H, W]

    # Perform the crop (crop_and_resize assumes a batch)
    img = tf.image.crop_and_resize(
        img[None, :, :, :], boxes=y0x0y1x1[None, :], box_indices=(0,), crop_size=size
    )[0]
    return img
