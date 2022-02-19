import tensorflow as tf

ATTEMPTS = (10,)


@tf.function
def get_params(img, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), seed=(0, 0)):
    # Ratio < 1: vertical portrait image
    # Ratio > 1: horizontal landscape image

    H, W = img.shape[:2]
    A = W * H

    seeds = tf.random.experimental.stateless_split(seed, num=4)

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
    img, size=(64, 64), scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), seed=(0, 0)
):
    ijhw = get_params(img, scale, ratio, seed)
    height, width = img.shape[:2]
    y0x0y1x1 = tf.concat([ijhw[:2], ijhw[:2] + ijhw[2:]], axis=0)
    y0x0y1x1 /= [height, width, height, width]
    img = tf.image.crop_and_resize(
        img[None, :, :, :], boxes=y0x0y1x1[None, :], box_indices=(0,), crop_size=size
    )[0]
    return img
