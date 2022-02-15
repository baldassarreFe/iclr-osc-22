"""
CLEVR with masks dataset: preprocessing and loading.
"""

import argparse
import sys
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Union

import IPython.display
import matplotlib.pyplot as plt
import multi_object_datasets.clevr_with_masks
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from osc.data.tfrecords import deserialize_image, serialize_image

IMAGE_SIZE = multi_object_datasets.clevr_with_masks.IMAGE_SIZE
MAX_NUM_ENTITIES = multi_object_datasets.clevr_with_masks.MAX_NUM_ENTITIES
NUM_SAMPLES_TOTAL = 100_000
NUM_SAMPLES_TRAIN = 70_000
NUM_SAMPLES_VAL = 15_000
NUM_SAMPLES_TEST = 15_000


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and split CLEVR with masks dataset into 3 splits: "
        "train+val only contain RGB images, "
        "test contains the full sample dictionary."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Filesystem path 'path/to/multi-object-datasets'",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files 'imgs_{train,val,test}.tfrecords'",
    )
    args = parser.parse_args()
    data_root = args.data_root / "clevr_with_masks"
    dst_paths = {
        split: Path.as_posix(data_root / f"imgs_{split}.tfrecords")
        for split in ["train", "val", "test"]
    }

    if not args.overwrite:
        for p in dst_paths.values():
            if p.is_file():
                print(
                    f"Error, output file already exists, use --overwrite: {p}",
                    file=sys.stderr,
                )
                exit(-1)

    def process(idx, example):
        if idx >= NUM_SAMPLES_TRAIN + NUM_SAMPLES_VAL:
            return example
        example = multi_object_datasets.clevr_with_masks._decode(example)
        example = tf.py_function(serialize_image, (example["image"],), tf.string)
        return example

    ds = (
        tf.data.TFRecordDataset(
            Path.as_posix(data_root / "clevr_with_masks_train.tfrecords"),
            compression_type="GZIP",
        )
        .enumerate()
        .map(
            process,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        .as_numpy_iterator()
    )

    for split, num_samples in [
        ("train", NUM_SAMPLES_TRAIN),
        ("val", NUM_SAMPLES_VAL),
        ("test", NUM_SAMPLES_TEST),
    ]:
        # Write all samples
        with tf.io.TFRecordWriter(dst_paths[split], options="GZIP") as writer:
            for example in tqdm.tqdm(
                islice(ds, num_samples),
                desc=f"Writing {split}",
                unit=" imgs",
                total=num_samples,
            ):
                writer.write(example)

        # Check reading
        ds_check = tf.data.TFRecordDataset(
            dst_paths[split], compression_type="GZIP"
        ).take(100)
        if split in {"train", "val"}:
            ds_check = ds_check.map(partial(deserialize_image, img_size=IMAGE_SIZE))
        else:
            ds_check = ds_check.map(multi_object_datasets.clevr_with_masks._decode)
            ds_check = ds_check.map(fix_tf_dtypes)
        for _ in tqdm.tqdm(ds_check, desc=f"Reading {split}", unit=" imgs", total=100):
            pass


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


if __name__ == "__main__":
    main()
