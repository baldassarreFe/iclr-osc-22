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
from keras.preprocessing.image import smart_resize

from osc.data.tfrecords import deserialize_image, serialize_image
from osc.data.utils import img_hwc_to_chw, normalize_tf
from osc.utils import ImgSizeHW

IMAGE_SIZE = multi_object_datasets.clevr_with_masks.IMAGE_SIZE
MAX_NUM_ENTITIES = multi_object_datasets.clevr_with_masks.MAX_NUM_ENTITIES
NUM_SAMPLES_TOTAL = 100_000
NUM_SAMPLES_TRAIN = 70_000
NUM_SAMPLES_VAL = 15_000
NUM_SAMPLES_TEST = 15_000

decode = multi_object_datasets.clevr_with_masks._decode


def main():
    args = parse_args()
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

    def process_train_val_test(idx, example_):
        if idx >= NUM_SAMPLES_TRAIN + NUM_SAMPLES_VAL:
            return example_
        example_ = decode(example_)
        example_ = tf.py_function(serialize_image, (example_["image"],), tf.string)
        return example_

    # The dataset comes as a single file. We split it in
    # - 70K train: images only
    # - 15K val:   images, masks, attributes, etc
    # - 15K test:  images, masks, attributes, etc
    ds = (
        tf.data.TFRecordDataset(
            Path.as_posix(data_root / "clevr_with_masks_train.tfrecords"),
            compression_type="GZIP",
        )
        .enumerate()
        .map(
            process_train_val_test,
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
            ds_check = ds_check.map(decode)
            ds_check = ds_check.map(fix_tf_dtypes)
        for _ in tqdm.tqdm(ds_check, desc=f"Reading {split}", unit=" imgs", total=100):
            pass


def parse_args():
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
    return args


def show_sample(sample):
    fig, axs = plt.subplots(
        1,
        1 + sample["mask"].shape[0],
        figsize=2
        * np.array([IMAGE_SIZE[1] / IMAGE_SIZE[0], 1])
        * np.array([1 + sample["mask"].shape[0], 1]),
        sharex=True,
        sharey=True,
    )

    axs[0].imshow(sample["image"], interpolation="none")
    axs[0].set_title("image")

    for m in range(sample["mask"].shape[0]):
        axs[m + 1].imshow(sample["mask"][m], cmap="gray", interpolation="none")
        axs[m + 1].set_title(f"mask {m}")

    fig.set_facecolor("white")
    fig.tight_layout()
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


@tf.function
def prepare_test_segmentation(
    example,
    img_size: ImgSizeHW,
    crop_size: ImgSizeHW,
    mean: tf.Tensor,
    std: tf.Tensor,
):
    """Prepare a test example for segmentation (center crop+normalization)

    Args:
        example:
        img_size: image size ``(H, W)``
        crop_size: crop size ``(H, W)``
        mean: image mean for normalization
        std: image standard deviation for normalization

    Returns:
        A dict containing the image ``[3 H W]``, the mask ``[C H W]`` and
        a bool vector of object visibility ``[C]``
    """
    # image: [H W 3]
    # mask: [C H W]
    image = example["image"]
    mask = example["mask"]

    H, W = img_size
    S = min(H, W)
    y0 = (H - S) // 2
    x0 = (W - S) // 2
    y1 = (H + S) // 2
    x1 = (W + S) // 2

    image = image[y0:y1, x0:x1, :]
    mask = mask[:, y0:y1, x0:x1]

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = normalize_tf(image, mean, std)
    image = tf.image.resize(image, crop_size)
    image = tf.transpose(image, [2, 0, 1])

    mask = tf.cast(mask, tf.uint8)
    mask = tf.transpose(mask, [1, 2, 0])
    mask = tf.image.resize(mask, crop_size)
    mask = tf.transpose(mask, [2, 0, 1])
    mask = tf.cast(mask, tf.bool)

    # image: [3 H W]
    # mask: [C H W]
    return {"image": image, "mask": mask, "visibility": example["visibility"]}


@tf.function
def prepare_test_vqa(
    example,
    crop_size: ImgSizeHW,
    mean: tf.Tensor,
    std: tf.Tensor,
):
    """Prepare a test example for VQA (center crop+normalization)

    The VQA target is a one-hot encoding of all possible questions like
    "is there at least one (size, color, material, shape) object in the scene?".
    There are 2 sizes, 8 colors, 2 materials and 3 shapes, so 96 binary values.

    Args:
        example:
        crop_size: crop size ``(H, W)``
        mean: image mean for normalization
        std: image standard deviation for normalization

    Returns:
        A dict containing the image ``[3 H W]`` and the VQA target ``[V]``.

    """
    # image: [H W RGB] -> [RGB H W]
    image = example["image"]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = smart_resize(image, crop_size)
    image = normalize_tf(image, mean, std)
    image = img_hwc_to_chw(image)

    # First object is always background, so slice [1:]
    # Background and non-existing objects have a value of 0 for all attributes,
    # so subtract 1 to shift all attributes in a [0, x] range
    size = example["size"][example["visibility"]][1:] - 1
    color = example["color"][example["visibility"]][1:] - 1
    material = example["material"][example["visibility"]][1:] - 1
    shape = example["shape"][example["visibility"]][1:] - 1

    # Count how many objects of each type
    # counts_nd: [size, color, material, shape]
    counts_nd = tf.scatter_nd(
        tf.cast(tf.stack([size, color, material, shape], axis=1), tf.int32),
        tf.ones_like(size),
        (2, 8, 2, 3),
    )
    vqa_target = tf.reshape(counts_nd > 0, (2 * 8 * 2 * 3,))

    return {"image": image, "vqa_target": vqa_target}


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
