from pathlib import Path
from typing import Union

import IPython.display
import matplotlib.pyplot as plt
import multi_object_datasets.multi_dsprites
import numpy as np
import pandas as pd
import tensorflow as tf

MODES = ["colored_on_grayscale", "colored_on_colored", "binarized"]
MODE_TO_FILE = {mode: f"multi_dsprites_{mode}.tfrecords" for mode in MODES}
IMAGE_SIZE = multi_object_datasets.multi_dsprites.IMAGE_SIZE


def show_sample(sample):
    fig, axs = plt.subplots(
        1,
        1 + sample["mask"].shape[0],
        figsize=3 * np.array([1 + sample["mask"].shape[0], 1]),
        sharex=True,
        sharey=True,
    )

    if sample["image"].shape[2] == 3:
        axs[0].imshow(sample["image"], interpolation="none")
    else:
        axs[0].imshow(sample["image"][:, :, 0], cmap="gray", interpolation="none")
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
                "scale": sample["scale"],
                "orientation": sample["orientation"],
                "color_RGB": list(sample["color"]),
            }
        )
    )


def fix_tf_dtypes(sample):
    sample["mask"] = tf.cast(tf.squeeze(sample["mask"], -1), tf.bool)
    sample["visibility"] = tf.cast(sample["visibility"], tf.bool)
    sample["shape"] = tf.cast(sample["shape"], tf.uint8)
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
    tfr_path = Path(data_dir) / "multi_dsprites" / MODE_TO_FILE[mode]
    ds = multi_object_datasets.multi_dsprites.dataset(
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
