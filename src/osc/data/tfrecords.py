import tensorflow as tf

from osc.utils import ImgSizeHW


def serialize_image(image):
    feature = {
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(image).numpy()])
        )
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


@tf.function
def deserialize_image(example, *, img_size: ImgSizeHW):
    example = tf.io.parse_single_example(
        example, {"image": tf.io.FixedLenFeature([], tf.string)}
    )
    image = tf.io.parse_tensor(example["image"], tf.uint8)
    image = tf.ensure_shape(image, (*img_size, 3))
    return image
