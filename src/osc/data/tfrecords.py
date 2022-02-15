"""
Helper functions to serialize/deserialize TFRecords.
"""
from typing import Tuple

import tensorflow as tf

from osc.utils import ImgSizeHW


def serialize_image(image: tf.Tensor) -> bytes:
    """Serialize an image to be written to a TFRecord.

    Args:
        image: uint8 tensor of shape ``[H W 3]``

    Example:
        When calling this function from a :mod:``tf.data`` pipeline,
        wrap it in a :func:``tf.py_function``::

        >>> tf.py_function(serialize_image, (image,), tf.string)

    Returns:
        A byte string.
    """

    feature = {
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(image).numpy()])
        )
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


@tf.function
def deserialize_image(example: bytes, *, img_size: ImgSizeHW) -> tf.Tensor:
    """Deserialize an image as read from a TFRecord.

    Args:
        example: byte serialization of the image
        img_size: expected shape of the image ``(H, W)``

    Returns:
        A uint8 tensor of shape ``[H W 3]``
    """
    example = tf.io.parse_single_example(
        example, {"image": tf.io.FixedLenFeature([], tf.string)}
    )
    image = tf.io.parse_tensor(example["image"], tf.uint8)
    image = tf.ensure_shape(image, (*img_size, 3))
    return image


def serialize_image_and_mask(image: tf.Tensor, mask: tf.Tensor):
    """Serialize an image and a mask to be written to a TFRecord.

    Args:
        image: uint8 tensor of shape ``[H W 3]``
        mask: bool tensor of shape ``[H W C]``, where ``C`` is the number of classes.

    Example:
        When calling this function from a :mod:``tf.data`` pipeline,
        wrap it in a :func:``tf.py_function``::

        >>> tf.py_function(serialize_image_and_mask, (image, mask), tf.string)

    Returns:
        A byte string.
    """
    feature = {
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(image).numpy()])
        ),
        "mask": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(mask).numpy()])
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


@tf.function
def deserialize_image_and_mask(
    example: bytes,
    *,
    img_size: ImgSizeHW,
    num_classes: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Deserialize an image and a mask as read from a TFRecord.

    Args:
        example: byte serialization of the image and mask.
        img_size: expected shape of the image ``(H, W)``.
        num_classes: expected number of classes ``C`` for the mask.

    Returns:
        One uint8 tensor of shape ``[H W 3]`` and
        one bool tensor of shape ``[H W, C]``.
    """

    example = tf.io.parse_single_example(
        example,
        features={
            "image": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
        },
    )
    image = tf.io.parse_tensor(example["image"], tf.uint8)
    image = tf.ensure_shape(image, (*img_size, 3))
    mask = tf.io.parse_tensor(example["mask"], tf.uint8)
    mask = tf.ensure_shape(mask, (*img_size, num_classes))
    return image, mask
