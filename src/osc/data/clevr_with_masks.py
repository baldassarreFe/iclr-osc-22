from pathlib import Path
from typing import Union

import IPython.display
import matplotlib.pyplot as plt
import multi_object_datasets.clevr_with_masks
import numpy as np
import pandas as pd
import tensorflow as tf

IMAGE_SIZE = multi_object_datasets.clevr_with_masks.IMAGE_SIZE
NUM_SAMPLES_TOTAL = 100_000
NUM_SAMPLES_TRAIN = 70_000
NUM_SAMPLES_VAL = 15_000


def show_sample(sample):
    fig, axs = plt.subplots(
        1,
        1 + sample["mask"].shape[0],
        figsize=np.array([3, 2]) * np.array([1 + sample["mask"].shape[0], 1]),
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
                "z": sample["z"],
                "pixel_coords": list(sample["pixel_coords"]),
                "rotation": sample["rotation"],
                "size": sample["size"],
                "material": sample["material"],
                "shape": sample["shape"],
                "color": sample["color"],
            }
        )
    )


def fix_tf_dtypes(sample):
    sample["mask"] = tf.cast(tf.squeeze(sample["mask"], -1), tf.bool)
    sample["visibility"] = tf.cast(sample["visibility"], tf.bool)
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
    tfr_path = Path(data_dir) / "clevr_with_masks" / "clevr_with_masks_train.tfrecords"
    ds = multi_object_datasets.clevr_with_masks.dataset(
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
