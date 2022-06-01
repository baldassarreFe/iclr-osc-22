import argparse
import textwrap
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, BertTokenizer, TFAutoModel

IMAGE_SIZE = (320, 240)
NUM_SAMPLES_TOTAL = 100_000
NUM_SAMPLES_TRAIN = 70_000
NUM_SAMPLES_VAL = 15_000
NUM_SAMPLES_TEST = 15_000
ANSWERS = [
    # Count
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    # Binary
    "no",
    "yes",
    # Color
    "blue",
    "brown",
    "cyan",
    "gray",
    "green",
    "purple",
    "red",
    "yellow",
    # Shape
    "cylinder",
    "cube",
    "sphere",
    # Size
    "large",
    "small",
    # material
    "metal",
    "rubber",
]


def main():
    args = parse_args()
    p = args.data_root
    Path.mkdir(p / "raw", exist_ok=True, parents=True)
    Path.mkdir(p / "processed", exist_ok=True, parents=True)

    # Set tf device and let it print its warnings
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[0], "GPU")

    # BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    bert = TFAutoModel.from_pretrained("bert-base-uncased")

    # The test set does not have answers, so we fill with 255
    answer_lut = {a.encode("utf8"): i for i, a in enumerate(ANSWERS)}
    answer_lut[b""] = 255

    batch_size = 64
    samples_per_shard = 5000

    # Process full dataset:
    # - input dict with image, question and answer strings, and object annotations
    # - tokenize and encode questions
    # - turn answers into int indexes
    # - encode as tfrecords
    for split in ["train", "validation", "test"]:
        ds = (
            tfds.load("clevr", split=split, data_dir=str(p / "raw"))
            .map(ragged_question_answer, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .map(squeeze_ragged, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        count = 0
        writer = None
        bar = tqdm(desc=split, ncols=0)
        for batch in ds:
            for sample in prepare_samples(batch, tokenizer, bert, answer_lut):
                if count % samples_per_shard == 0:
                    writer = tf.io.TFRecordWriter(
                        str(
                            p
                            / "processed"
                            / f"{split}.{count//samples_per_shard}.tfrecords"
                        ),
                        options="GZIP",
                    )

                writer.write(serialize_sample(sample))
                count += 1
                bar.update()

                if count % samples_per_shard == 0:
                    writer.close()

        bar.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
        Preprocess CLEVR dataset for VQA:
        - questions are tokenized and encoded using BERT
        - answers are represented as an integer index
        """
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Filesystem path 'path/to/clevr-vqa'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for BERT",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Number of samples per shard (5000 samples give 3.7GB/shard)",
    )
    args = parser.parse_args()
    return args


@tf.function
def ragged_question_answer(sample):
    question = sample["question_answer"]["question"]
    answer = sample["question_answer"]["answer"]
    # The N question and answers become ragged tensor with shape [N, 1],
    # The extra dim is needed because ragged tensors mush have rank >=2
    return {
        "image": sample["image"],
        "question": tf.RaggedTensor.from_tensor(question[:, None]),
        "answer": tf.RaggedTensor.from_tensor(answer[:, None]),
    }


@tf.function
def squeeze_ragged(batch):
    # After batching, question and answer have shape [B, (N), 1]
    # where N is ragged. The last dimension can be squeezed to [B, (N)].
    return {
        "image": batch["image"],
        "question": tf.squeeze(batch["question"], -1),
        "answer": tf.squeeze(batch["answer"], -1),
    }


def prepare_samples(
    batch: Dict[str, Tensor],
    tokenizer: BertTokenizer,
    bert: BertModel,
    answer_lut: Dict[bytes, int],
) -> List[Dict[str, Tensor]]:
    tkns = tokenizer(
        batch["question"].flat_values.numpy().astype(str).tolist(),
        padding="longest",
        add_special_tokens=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="tf",
        return_token_type_ids=True,
    )
    num_questions = batch["question"].row_lengths()
    num_tkns_per_question = tf.math.count_nonzero(tkns["attention_mask"], axis=-1)

    hidden_states = bert.call(
        input_ids=tkns["input_ids"],
        token_type_ids=tkns["token_type_ids"],
        attention_mask=tkns["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )["hidden_states"]

    question_enc = tf.reduce_sum(hidden_states[-4:], axis=0)
    question_enc = tf.RaggedTensor.from_tensor(
        question_enc, lengths=[num_tkns_per_question]
    )
    question_enc = tf.RaggedTensor.from_row_lengths(
        question_enc, row_lengths=num_questions
    )

    question_tkn = tf.RaggedTensor.from_tensor(
        tkns["input_ids"], lengths=[num_tkns_per_question]
    )
    question_tkn = tf.RaggedTensor.from_row_lengths(
        question_tkn, row_lengths=num_questions
    )

    answer = tf.RaggedTensor.from_row_splits(
        pd.Series(batch["answer"].flat_values.numpy())
        .map(answer_lut)
        .values.astype(np.uint8),
        batch["answer"].row_splits,
    )

    return [
        {
            "image": batch["image"][b],
            "question_tkn": question_tkn[b],
            "question_enc": question_enc[b],
            "answer": answer[b],
        }
        for b in range(batch["image"].shape[0])
    ]


def serialize_sample(sample: Dict[str, Tensor]) -> bytes:
    feature = {
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(sample["image"]).numpy()]
            )
        ),
        "question_splits": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[
                    tf.io.serialize_tensor(sample["question_enc"].row_splits).numpy()
                ]
            )
        ),
        "question_tkn_flat": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[
                    tf.io.serialize_tensor(sample["question_tkn"].flat_values).numpy()
                ]
            )
        ),
        "question_enc_flat": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[
                    tf.io.serialize_tensor(sample["question_enc"].flat_values).numpy()
                ]
            )
        ),
        "answer": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(sample["answer"]).numpy()]
            )
        ),
    }
    sample = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample.SerializeToString()


def deserialize_sample(sample: bytes) -> Dict[str, Tensor]:
    sample = tf.io.parse_single_example(
        sample,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "question_splits": tf.io.FixedLenFeature([], tf.string),
            "question_tkn_flat": tf.io.FixedLenFeature([], tf.string),
            "question_enc_flat": tf.io.FixedLenFeature([], tf.string),
            "answer": tf.io.FixedLenFeature([], tf.string),
        },
    )
    image = tf.io.parse_tensor(sample["image"], tf.uint8)
    question_splits = tf.io.parse_tensor(sample["question_splits"], tf.int64)
    question_tkn = tf.io.parse_tensor(sample["question_tkn_flat"], tf.int32)
    question_tkn = tf.RaggedTensor.from_row_splits(question_tkn, question_splits)
    question_enc = tf.io.parse_tensor(sample["question_enc_flat"], tf.float32)
    question_enc = tf.RaggedTensor.from_row_splits(question_enc, question_splits)
    answer = tf.io.parse_tensor(sample["answer"], tf.uint8)
    return {
        "image": image,
        "question_tkn": question_tkn,
        "question_enc": question_enc,
        "answer": answer,
    }
