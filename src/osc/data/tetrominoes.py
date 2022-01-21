from pathlib import Path
from typing import Union

import IPython.display
import matplotlib.pyplot as plt
import multi_object_datasets.tetrominoes
import numpy as np
import pandas as pd
import tensorflow as tf

IMAGE_SIZE = multi_object_datasets.tetrominoes.IMAGE_SIZE


def show_sample(sample):
    fig, axs = plt.subplots(
        1,
        1 + sample["mask"].shape[0],
        figsize=3 * np.array([1 + sample["mask"].shape[0], 1]),
        sharex=True,
        sharey=True,
    )

    axs[0].imshow(sample["image"], interpolation="none")
    axs[0].set_title("image")

    for m in range(sample["mask"].shape[0]):
        axs[m + 1].imshow(sample["mask"][m], cmap="gray", interpolation="none")
        axs[m + 1].set_title(f"mask {m}")

    fig.set_facecolor("white")
    IPython.display.display(fig)
    plt.close(fig)

    IPython.display.display(
        pd.DataFrame(
            {
                "visibility": sample["visibility"],
                "x": sample["x"],
                "y": sample["y"],
                "shape": sample["shape"],
                "color_RGB": list(sample["color"]),
            }
        )
    )


def fix_tf_dtypes(sample):
    sample["mask"] = tf.cast(tf.squeeze(sample["mask"], -1), tf.bool)
    sample["visibility"] = tf.cast(sample["visibility"], tf.bool)
    sample["x"] = tf.cast(sample["x"], tf.uint8)
    sample["y"] = tf.cast(sample["y"], tf.uint8)
    sample["shape"] = tf.cast(sample["shape"], tf.uint8)
    return sample


def get_iterator(
    data_dir: Union[str, Path],
    map_parallel_calls: int = None,
    take: int = None,
    batch_size: int = None,
    drop_remainder=False,
    shuffle: int = None,
    numpy=True,
):
    tfr_path = Path(data_dir) / "tetrominoes" / "tetrominoes_train.tfrecords"
    ds = multi_object_datasets.tetrominoes.dataset(
        tfr_path.expanduser().resolve().as_posix(),
        map_parallel_calls=map_parallel_calls,
    )
    ds = ds.map(fix_tf_dtypes)
    if take is not None:
        ds = ds.take(take)
    if shuffle is not None:
        ds = ds.shuffle(shuffle, seed=0)
    if batch_size is not None:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    if numpy:
        ds = ds.as_numpy_iterator()
    return ds
