from pathlib import Path
from typing import Union

import IPython.display
import matplotlib.pyplot as plt
import multi_object_datasets.objects_room
import numpy as np
import tensorflow as tf

MODE_TO_FILE = {
    "six_objects": "objects_room_test_six_objects.tfrecords",
    "empty_room": "objects_room_test_empty_room.tfrecords",
    "identical_color": "objects_room_test_identical_color.tfrecords",
    "train": "objects_room_train.tfrecords",
}
MODES = list(MODE_TO_FILE.keys())
IMAGE_SIZE = multi_object_datasets.objects_room.IMAGE_SIZE


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


def fix_tf_dtypes(sample):
    sample["mask"] = tf.cast(tf.squeeze(sample["mask"], -1), tf.bool)
    return sample


def get_iterator(
    data_dir: Union[str, Path],
    mode: str,
    map_parallel_calls: int = None,
    take: int = None,
    batch_size: int = None,
    drop_remainder=False,
    shuffle: int = None,
    numpy=True,
):
    tfr_path = Path(data_dir) / "objects_room" / MODE_TO_FILE[mode]
    ds = multi_object_datasets.objects_room.dataset(
        tfr_path.expanduser().resolve().as_posix(),
        mode,
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
