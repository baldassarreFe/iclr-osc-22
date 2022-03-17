"""
Main training methods.
"""

import logging
import os
import random
import time
from collections import defaultdict
from contextlib import suppress
from functools import partial
from itertools import islice
from operator import attrgetter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Protocol, Tuple

import hydra
import namesgenerator
import numpy as np
import omegaconf
import pandas as pd
import PIL.Image
import submitit
import tabulate
import tensorflow as tf
import torch
import torch.nn.functional
import tqdm
import wandb
import wandb.util
from einops import rearrange, reduce
from einops.layers.torch import Reduce
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)
from torch import Tensor
from torchvision.transforms.functional import to_pil_image

import osc.data.clevr_with_masks
from osc.data.tfrecords import deserialize_image
from osc.data.utils import augment_train, augment_twice, unnormalize_pt, wrap_with_seed
from osc.loss_global import contrastive_loss, cosine_sim_loss
from osc.loss_objects import (
    matching_contrastive_loss,
    matching_contrastive_loss_per_img,
    matching_similarity_loss_per_img,
)
from osc.lr_scheduler import LinearWarmupCosineAnneal
from osc.models.attentions import CrossAttentionDecoder, SlotAttention
from osc.models.embeds import (
    KmeansEuclideanObjectTokens,
    LearnedObjectTokens,
    PositionalEmbedding,
    SampledObjectTokens,
)
from osc.models.models import Model, ModelOutput, forward_with_attns
from osc.models.utils import MLP
from osc.models.vit import ViTBackbone
from osc.utils import (
    AverageMetric,
    SigIntCatcher,
    StepCounter,
    batches_per_epoch,
    normalize_sum_to_one,
    seed_everything,
)
from osc.viz.backbone import kmeans_clusters
from osc.viz.embeds import viz_positional_embedding
from osc.viz.loss_global import viz_contrastive_loss_global_probs
from osc.viz.loss_objects import viz_contrastive_loss_objects_probs
from osc.viz.rollout import cross_attn_rollout, self_attn_rollout, slot_attn_rollout
from osc.viz.segmentation import match_segmentation_masks
from osc.viz.utils import array_to_pil, batched_otsu_pt, make_grid_pil, subplots_grid
from osc.wandb_utils import setup_wandb

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    log.info("Config original:\n%s", OmegaConf.to_yaml(cfg))
    cfg = update_cfg(cfg)
    with open("train.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    setup_wandb(cfg)
    seed_everything(cfg.other.seed)
    log_env_info()

    ds_train = build_dataset_train(cfg)
    ds_val = build_dataset_val(cfg)
    ds_test = build_dataset_test(cfg)
    ds_vqa = build_dataset_vqa(cfg)
    viz_batch = get_viz_batch(cfg)

    model = build_model(cfg).to(cfg.other.device)
    log_model_parameters(model)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn_global = build_loss_fn_global(cfg)
    loss_fn_objects = build_loss_fn_objects(cfg)
    run_train_val_viz_epochs(
        cfg,
        model,
        optimizer,
        scheduler,
        ds_train,
        ds_val,
        ds_test,
        ds_vqa,
        viz_batch,
        loss_fn_global,
        loss_fn_objects,
    )

    log.info("Run dir: %s", Path.cwd().relative_to(hydra.utils.get_original_cwd()))


def update_cfg(cfg: DictConfig, readonly=True):
    up = DictConfig({})

    if cfg.other.debug:
        OmegaConf.update(up, "training.num_epochs", 3)
        OmegaConf.update(up, "training.checkpoint_interval", 1)
        OmegaConf.update(up, "data.train.max_samples", 10 * cfg.training.batch_size)
        OmegaConf.update(up, "data.val.max_samples", 10 * cfg.training.batch_size)
        OmegaConf.update(up, "data.test.max_samples", 5 * cfg.training.batch_size)
        OmegaConf.update(up, "data.viz.epoch_interval", 1)
        OmegaConf.update(up, "data.test.epoch_interval", 1)
        OmegaConf.update(up, "other.seed", 42)
        OmegaConf.update(up, "data.train.seed", 42)

    else:
        if cfg.training.num_epochs is None:
            OmegaConf.update(
                up,
                "training.num_epochs",
                cfg.lr_scheduler.warmup.epochs + cfg.lr_scheduler.decay.epochs,
            )

        if cfg.data.train.max_samples is None:
            if cfg.data.name == "CLEVR10":
                OmegaConf.update(
                    up,
                    "data.train.max_samples",
                    osc.data.clevr_with_masks.NUM_SAMPLES_TRAIN,
                )
            else:
                raise ValueError(cfg.data.name)

        if cfg.data.val.max_samples is None:
            if cfg.data.name == "CLEVR10":
                OmegaConf.update(
                    up,
                    "data.val.max_samples",
                    osc.data.clevr_with_masks.NUM_SAMPLES_VAL,
                )
            else:
                raise ValueError(cfg.data.name)

        if cfg.data.test.max_samples is None:
            if cfg.data.name == "CLEVR10":
                OmegaConf.update(
                    up,
                    "data.test.max_samples",
                    osc.data.clevr_with_masks.NUM_SAMPLES_TEST,
                )
            else:
                raise ValueError(cfg.data.name)

        if cfg.other.seed is None:
            OmegaConf.update(up, "other.seed", random.randint(0, np.power(2, 32)))

        if cfg.data.train.seed is None:
            OmegaConf.update(up, "data.train.seed", random.randint(0, np.power(2, 32)))

        if cfg.logging.id is None:
            OmegaConf.update(up, "logging.id", wandb.util.generate_id())

        if cfg.logging.name is None:
            OmegaConf.update(
                up,
                "logging.name",
                namesgenerator.get_random_name("-") + f"-{random.randint(0, 99):<02d}",
            )

    log.info("Config updates:\n%s", OmegaConf.to_yaml(up))
    with omegaconf.open_dict(cfg):
        result = OmegaConf.merge(cfg, up)
    OmegaConf.set_struct(result, True)
    OmegaConf.set_readonly(result, readonly)
    return result


def log_env_info():
    log.info("PID: %d", os.getpid())
    log.info("Run dir: %s", Path.cwd().relative_to(hydra.utils.get_original_cwd()))

    conda = [f"{k}={v}" for k, v in os.environ.items() if k.startswith("CONDA_")]
    if len(conda) > 0:
        log.info("Conda:\n%s", "\n".join(conda))

    slurm = [f"{k}={v}" for k, v in os.environ.items() if k.startswith("SLURM_")]
    if len(slurm) > 0:
        log.info("Slurm:\n%s", "\n".join(slurm))

    # Ignore errors when running a single run locally
    with suppress(RuntimeError):
        env = submitit.JobEnvironment()
        log.info("Submitit env:\n%s", env)


def build_model(cfg: DictConfig) -> Model:
    """Build model"""
    num_patches = np.array(cfg.data.crop_size) // np.array(
        cfg.model.backbone.patch_size
    )
    activations = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
    }

    if cfg.model.backbone.pos_embed == "learned":
        backbone_pos_embed = PositionalEmbedding(
            np.prod(num_patches),
            cfg.model.backbone.embed_dim,
            dropout=cfg.model.backbone.pos_embed_dropout,
        )
    elif cfg.model.backbone.pos_embed is None:
        backbone_pos_embed = None
    else:
        raise ValueError(cfg.model.backbone.pos_embed)

    if cfg.model.obj_fn is None or cfg.model.obj_fn.pos_embed is None:
        obj_fn_pos_embed = None
    elif cfg.model.obj_fn.pos_embed == "backbone":
        obj_fn_pos_embed = backbone_pos_embed
    elif cfg.model.obj_fn.pos_embed == "learned":
        obj_fn_pos_embed = PositionalEmbedding(
            np.prod(num_patches),
            cfg.model.backbone.embed_dim,
            dropout=cfg.model.backbone.pos_embed_dropout,
        )
    else:
        raise ValueError(cfg.model.obj_fn.pos_embed)

    backbone = ViTBackbone(
        img_size=cfg.data.crop_size,
        pos_embed=backbone_pos_embed,
        pos_embed_every_layer=cfg.model.backbone.pos_embed_every_layer,
        embed_dim=cfg.model.backbone.embed_dim,
        patch_size=cfg.model.backbone.patch_size,
        num_heads=cfg.model.backbone.num_heads,
        num_layers=cfg.model.backbone.num_layers,
        block_drop=cfg.model.backbone.block_drop,
        block_attn_drop=cfg.model.backbone.block_attn_drop,
        drop_path=cfg.model.backbone.drop_path,
        mlp_ratio=cfg.model.backbone.mlp_ratio,
        global_pool=cfg.model.backbone.global_pool,
    )

    global_fn = MLP(
        in_features=cfg.model.backbone.embed_dim,
        hidden_features=int(
            np.round(cfg.model.global_fn.hidden_mult * cfg.model.backbone.embed_dim)
        ),
        out_features=cfg.model.backbone.embed_dim,
        activation=activations[cfg.model.global_fn.activation],
        dropout=0.0,
    )

    global_proj = MLP(
        in_features=cfg.model.backbone.embed_dim,
        hidden_features=int(
            np.round(cfg.model.global_proj.hidden_mult * cfg.model.backbone.embed_dim)
        ),
        out_features=cfg.model.backbone.embed_dim,
        activation=activations[cfg.model.global_proj.activation],
        out_bias=False,
        dropout=0.0,
    )

    if cfg.model.obj_queries is None:
        obj_queries = None
    elif cfg.model.obj_queries.name == "sample":
        obj_queries = SampledObjectTokens(
            embed_dim=cfg.model.backbone.embed_dim,
            num_objects=cfg.model.obj_queries.num_objects,
            num_components=cfg.model.obj_queries.num_components,
        )
    elif cfg.model.obj_queries.name == "learned":
        obj_queries = LearnedObjectTokens(
            embed_dim=cfg.model.backbone.embed_dim,
            num_objects=cfg.model.obj_queries.num_objects,
        )
    elif cfg.model.obj_queries.name == "kmeans_euclidean":
        obj_queries = KmeansEuclideanObjectTokens(
            num_objects=cfg.model.obj_queries.num_objects
        )
    else:
        raise ValueError(cfg.model.obj_queries.name)

    if cfg.model.obj_fn is None:
        obj_fn = None
    elif cfg.model.obj_fn.name == "slot-attention":
        obj_fn = SlotAttention(
            dim=cfg.model.backbone.embed_dim,
            pos_embed=obj_fn_pos_embed,
            iters=cfg.model.obj_fn.num_iters,
            hidden_dim=int(
                np.round(cfg.model.obj_fn.hidden_mult * cfg.model.backbone.embed_dim)
            ),
        )
    elif cfg.model.obj_fn.name == "cross-attention":
        obj_fn = CrossAttentionDecoder(
            dim=cfg.model.backbone.embed_dim,
            num_layers=cfg.model.obj_fn.num_layers,
            num_heads=cfg.model.obj_fn.num_heads,
            pos_embed=obj_fn_pos_embed,
            block_drop=cfg.model.obj_fn.block_drop,
            block_attn_drop=cfg.model.obj_fn.block_attn_drop,
            drop_path=cfg.model.obj_fn.drop_path,
            mlp_ratio=cfg.model.obj_fn.mlp_ratio,
        )
    else:
        raise ValueError(cfg.model.obj_fn.name)

    if cfg.model.obj_proj is None:
        obj_proj = None
    else:
        obj_proj = MLP(
            in_features=cfg.model.backbone.embed_dim,
            hidden_features=int(
                np.round(cfg.model.obj_proj.hidden_mult * cfg.model.backbone.embed_dim)
            ),
            out_features=cfg.model.backbone.embed_dim,
            activation=activations[cfg.model.obj_proj.activation],
            out_bias=False,
            dropout=0.0,
        )

    return Model(
        architecture=cfg.model.architecture,
        backbone=backbone,
        global_fn=global_fn,
        global_proj=global_proj,
        obj_queries=obj_queries,
        obj_fn=obj_fn,
        obj_proj=obj_proj,
    )


def log_model_parameters(model: torch.nn.Module):
    """Log model parameters as a table. First level only."""
    table = [
        [
            name,
            child.__class__.__name__,
            sum(p.numel() for p in child.parameters() if p.requires_grad),
        ]
        for name, child in model.named_children()
    ]
    table.append(["TOTAL", "", sum(r[-1] for r in table)])
    table = tabulate.tabulate(table, headers=["Module", "Class", "Parameters"])
    log.info("Model parameters:\n%s", table)


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build optimizer for training.

    Args:
        cfg: configuration
        model: model

    Returns:
        An optimizer.
    """
    if cfg.optimizer.name == "adam":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.start_lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
    raise ValueError(cfg.optimizer.name)


def build_scheduler(cfg, optimizer):
    """Build learning rate scheduler for training.

    Args:
        cfg: configuration
        optimizer: optimizer

    Returns:
        A learning rate scheduler.
    """
    if cfg.lr_scheduler.decay.name == "cosine":
        bpe = batches_per_epoch_train(cfg)
        return LinearWarmupCosineAnneal(
            optimizer,
            warmup_steps=cfg.lr_scheduler.warmup.epochs * bpe,
            decay_steps=cfg.lr_scheduler.decay.epochs * bpe,
            end_lr=cfg.lr_scheduler.decay.end_lr,
        )
    else:
        raise ValueError(cfg.lr_scheduler.decay.name)


class ModelLoss(Protocol):
    """Function type signature for model loss functions."""

    def __call__(
        self, output: ModelOutput, *, reduction: str = "mean"
    ) -> Optional[Tensor]:
        ...


def build_loss_fn_global(cfg: DictConfig) -> ModelLoss:
    """Build global loss function."""
    if cfg.losses.l_global is None:

        def loss_fn_global(output: ModelOutput, *, reduction="mean") -> None:
            return None

    elif cfg.losses.l_global.name == "sim":

        def loss_fn_global(output: ModelOutput, *, reduction="mean") -> Tensor:
            return cosine_sim_loss(
                output.f_global, output.p_global, reduction=reduction
            )

    elif cfg.losses.l_global.name == "ctr":

        def loss_fn_global(output: ModelOutput, *, reduction="mean") -> Tensor:
            return contrastive_loss(
                output.p_global,
                temperature=cfg.losses.l_global.temp,
                reduction=reduction,
            )

    else:
        raise ValueError(cfg.losses.l_global.name)

    return loss_fn_global


def build_loss_fn_objects(cfg: DictConfig) -> ModelLoss:
    """Build object loss function."""
    if cfg.model.obj_proj is None and cfg.losses.l_objects.name is not None:
        raise ValueError(
            "Can not build object loss if model does not output object projections."
        )

    if cfg.losses.l_objects.name is None:

        def loss_fn_objects(output: ModelOutput, *, reduction="mean") -> None:
            return None

    elif cfg.losses.l_objects.name == "sim_img":

        def loss_fn_objects(output: ModelOutput, *, reduction="mean") -> Tensor:
            return matching_similarity_loss_per_img(
                output.f_slots,
                output.p_slots,
                reduction=reduction,
            )

    elif cfg.losses.l_objects.name == "ctr_img":

        def loss_fn_objects(output: ModelOutput, *, reduction="mean") -> Tensor:
            return matching_contrastive_loss_per_img(
                output.p_slots,
                temperature=cfg.losses.l_objects.temp,
                reduction=reduction,
            )

    elif cfg.losses.l_objects.name == "ctr_all":

        def loss_fn_objects(output: ModelOutput, *, reduction="mean") -> Tensor:
            return matching_contrastive_loss(
                output.p_slots,
                temperature=cfg.losses.l_objects.temp,
                reduction=reduction,
            )

    else:
        raise ValueError(cfg.losses.l_global.name)

    return loss_fn_objects


def build_dataset_train(cfg):
    """Build training dataset.

    Args:
        cfg: configuration

    Returns:
        Training dataset.
    """
    if cfg.data.name == "CLEVR10":
        tfr_path = Path(cfg.data.root) / "clevr_with_masks" / "imgs_train.tfrecords"
        img_size = osc.data.clevr_with_masks.IMAGE_SIZE
        if cfg.data.train.max_samples > osc.data.clevr_with_masks.NUM_SAMPLES_TRAIN:
            raise ValueError(cfg.data.train.max_samples)
    else:
        raise ValueError(cfg.data.name)

    ds = tf.data.TFRecordDataset(
        tfr_path.expanduser().resolve().as_posix(),
        compression_type="GZIP",
    )

    # Limit number of samples per epoch
    ds = ds.take(cfg.data.train.max_samples)

    # Decode image
    ds = ds.map(
        partial(deserialize_image, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # Shuffle using a buffer
    bpe = batches_per_epoch_train(cfg)
    ds = ds.shuffle(min(10, bpe) * cfg.training.batch_size, seed=cfg.data.train.seed)

    # Augment twice [H W C] -> ([C H W], [C H W])
    augment_fn = partial(
        augment_train,
        crop_size=tuple(cfg.data.crop_size),
        crop_scale=tuple(cfg.data.crop_scale),
        mean=tuple(cfg.data.normalize.mean),
        std=tuple(cfg.data.normalize.std),
    )
    ds = ds.map(
        augment_twice(
            wrap_with_seed(augment_fn, initial_seed=cfg.data.train.seed + 1),
            wrap_with_seed(augment_fn, initial_seed=cfg.data.train.seed + 2),
        ),
    )

    # Batch: B x ([C H W], [C H W]) -> ([B C H W], [B C H W])
    ds = ds.batch(cfg.training.batch_size, drop_remainder=True)

    # Stack augmented batches ([B C H W], [B C H W]) -> [A B C H W]
    ds = ds.map(
        lambda b0, b1: tf.stack([b0, b1], axis=0),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # Add batch idx
    ds = ds.enumerate()
    # Repeat per epoch
    ds = ds.repeat(cfg.training.num_epochs)
    # Other stuff
    ds = ds.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    log.info(
        "Dataset train, %d samples, %d batch size, %d batches, %d epochs",
        cfg.data.train.max_samples,
        cfg.training.batch_size,
        bpe,
        cfg.training.num_epochs,
    )

    return ds


def build_dataset_val(cfg: DictConfig) -> Iterable:
    """Build validation dataset.

    Args:
        cfg: configuration

    Returns:
        Validation dataset.
    """
    if cfg.data.name == "CLEVR10":
        tfr_path = Path(cfg.data.root) / "clevr_with_masks" / "imgs_val.tfrecords"
        img_size = osc.data.clevr_with_masks.IMAGE_SIZE
        if cfg.data.val.max_samples > osc.data.clevr_with_masks.NUM_SAMPLES_VAL:
            raise ValueError(cfg.data.val.max_samples)
    else:
        raise ValueError(cfg.data.name)

    ds = tf.data.TFRecordDataset(
        tfr_path.expanduser().resolve().as_posix(),
        compression_type="GZIP",
    )

    # Limit number of samples per epoch
    ds = ds.take(cfg.data.val.max_samples)

    # Decode image
    ds = ds.map(
        partial(deserialize_image, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # Zip with two random sequences to get per-image random seeds
    # that repeat identically at every epoch
    ds = tf.data.Dataset.zip(
        (
            ds,
            tf.data.Dataset.random(cfg.data.val.seed),
            tf.data.Dataset.random(cfg.data.val.seed + 1),
        )
    )

    # Augment twice [H W C] -> ([C H W], [C H W])
    augment_fn = partial(
        augment_train,
        crop_size=tuple(cfg.data.crop_size),
        crop_scale=tuple(cfg.data.crop_scale),
        mean=tuple(cfg.data.normalize.mean),
        std=tuple(cfg.data.normalize.std),
    )
    ds = ds.map(
        lambda img, s0, s1: (
            augment_fn(img, seed=[s0, 0]),
            augment_fn(img, seed=[s1, 0]),
        )
    )

    # Batch: B x ([C H W], [C H W]) -> ([B C H W], [B C H W])
    ds = ds.batch(cfg.training.batch_size, drop_remainder=False)

    # Stack augmented batches ([B C H W], [B C H W]) -> [A B C H W]
    ds = ds.map(
        lambda b0, b1: tf.stack([b0, b1], axis=0),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # Add batch idx
    ds = ds.enumerate()
    # Repeat per epoch
    ds = ds.repeat(cfg.training.num_epochs)
    # Other stuff
    ds = ds.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    log.info(
        "Dataset val, %d samples, %d batch size, %d batches, %d epochs",
        cfg.data.val.max_samples,
        cfg.training.batch_size,
        batches_per_epoch_val(cfg),
        cfg.training.num_epochs,
    )
    return ds


def build_dataset_test(cfg: DictConfig) -> tf.data.Dataset:
    """Build test dataset.

    Args:
        cfg: configuration

    Returns:
        Test dataset.
    """
    if cfg.data.name == "CLEVR10":
        tfr_path = Path(cfg.data.root) / "clevr_with_masks" / "imgs_test.tfrecords"
        img_size = osc.data.clevr_with_masks.IMAGE_SIZE
        if cfg.data.test.max_samples > osc.data.clevr_with_masks.NUM_SAMPLES_TEST:
            raise ValueError(cfg.data.test.max_samples)

        ds = (
            tf.data.TFRecordDataset(
                tfr_path.expanduser().resolve().as_posix(), compression_type="GZIP"
            )
            .take(cfg.data.test.max_samples)
            .map(
                osc.data.clevr_with_masks.decode,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            .map(
                osc.data.clevr_with_masks.fix_tf_dtypes,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            .map(
                partial(
                    osc.data.clevr_with_masks.prepare_test_segmentation,
                    img_size=tuple(img_size),
                    crop_size=tuple(cfg.data.crop_size),
                    mean=tuple(cfg.data.normalize.mean),
                    std=tuple(cfg.data.normalize.std),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
        )
    else:
        raise ValueError(cfg.data.name)

    # Other stuff
    ds = ds.batch(cfg.training.batch_size, drop_remainder=False)
    ds = ds.enumerate()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    log.info(
        "Dataset test segmentation, %d samples, %d batch size, %d batches",
        cfg.data.test.max_samples,
        cfg.training.batch_size,
        batches_per_epoch_test(cfg),
    )
    return ds


def build_dataset_vqa(cfg: DictConfig) -> tf.data.Dataset:
    """Build VQA dataset for linear probing.

    Args:
        cfg: configuration

    Returns:
        VQA dataset.
    """
    if cfg.data.name == "CLEVR10":
        tfr_path = Path(cfg.data.root) / "clevr_with_masks" / "imgs_test.tfrecords"
        img_size = osc.data.clevr_with_masks.IMAGE_SIZE
        if cfg.data.test.max_samples > osc.data.clevr_with_masks.NUM_SAMPLES_TEST:
            raise ValueError(cfg.data.test.max_samples)

        ds = (
            tf.data.TFRecordDataset(
                tfr_path.expanduser().resolve().as_posix(), compression_type="GZIP"
            )
            .take(cfg.data.test.max_samples)
            .map(
                osc.data.clevr_with_masks.decode,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            .map(
                osc.data.clevr_with_masks.fix_tf_dtypes,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            .map(
                partial(
                    osc.data.clevr_with_masks.prepare_test_vqa,
                    img_size=tuple(img_size),
                    crop_size=tuple(cfg.data.crop_size),
                    mean=tuple(cfg.data.normalize.mean),
                    std=tuple(cfg.data.normalize.std),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
        )
    else:
        raise ValueError(cfg.data.name)

    # Other stuff
    ds = ds.batch(cfg.training.batch_size, drop_remainder=False)
    ds = ds.enumerate()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    log.info(
        "Dataset test VQA, %d samples, %d batch size, %d batches",
        cfg.data.test.max_samples,
        cfg.training.batch_size,
        batches_per_epoch_test(cfg),
    )
    return ds


def get_viz_batch(cfg: DictConfig) -> np.ndarray:
    """Prepare a batch of images for visualization.

    Args:
        cfg: configuration

    Returns:
        A batch of images, shape ``[A B C H W]``.
    """
    if cfg.data.name == "CLEVR10":
        tfr_path = Path(cfg.data.root) / "clevr_with_masks" / "imgs_val.tfrecords"
        img_size = osc.data.clevr_with_masks.IMAGE_SIZE
        if cfg.data.viz.max_samples > osc.data.clevr_with_masks.NUM_SAMPLES_VAL:
            raise ValueError(cfg.data.viz.max_samples)
    else:
        raise ValueError(cfg.data.name)

    ds = tf.data.TFRecordDataset(
        tfr_path.expanduser().resolve().as_posix(),
        compression_type="GZIP",
    )

    # Limit number of samples
    ds = ds.take(cfg.data.viz.max_samples)

    # Decode image
    ds = ds.map(partial(deserialize_image, img_size=img_size))

    # Zip with two random sequences to get per-image random seeds
    ds = tf.data.Dataset.zip(
        (
            ds,
            tf.data.Dataset.random(cfg.data.viz.seed),
            tf.data.Dataset.random(cfg.data.viz.seed + 1),
        )
    )

    # Augment twice [H W C] -> ([C H W], [C H W])
    augment_fn = partial(
        augment_train,
        crop_size=tuple(cfg.data.crop_size),
        crop_scale=tuple(cfg.data.crop_scale),
        mean=tuple(cfg.data.normalize.mean),
        std=tuple(cfg.data.normalize.std),
    )
    ds = ds.map(
        lambda img, s0, s1: (
            augment_fn(img, seed=[s0, 0]),
            augment_fn(img, seed=[s1, 0]),
        )
    )

    # Batch: B x ([C H W], [C H W]) -> ([B C H W], [B C H W])
    ds = ds.batch(cfg.data.viz.max_samples, drop_remainder=False)

    # Stack augmented batches ([B C H W], [B C H W]) -> [A B C H W]
    ds = ds.map(
        lambda b0, b1: tf.stack([b0, b1], axis=0),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    images = ds.get_single_element().numpy()
    log.info("Viz batch, %d samples, shape %s", cfg.data.viz.max_samples, images.shape)
    return images


def run_train_val_viz_epochs(
    cfg: DictConfig,
    model: Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ds_train: Iterable,
    ds_val: Iterable,
    ds_test: tf.data.Dataset,
    ds_vqa: tf.data.Dataset,
    viz_batch: np.ndarray,
    loss_fn_global: ModelLoss,
    loss_fn_objects: ModelLoss,
):
    """Run train, val, viz for a certain number of epochs. Handle CTRL+C gracefully.

    Args:
        cfg:
        model:
        optimizer:
        scheduler:
        ds_train:
        ds_val:
        ds_test:
        ds_vqa:
        viz_batch:
        loss_fn_global:
        loss_fn_objects:
    """
    log.info("Start training for %d epochs", cfg.training.num_epochs)
    step_counter = StepCounter()
    with SigIntCatcher() as should_stop:
        for epoch in range(cfg.training.num_epochs):
            start = time.time()
            is_last = epoch == cfg.training.num_epochs - 1

            run_train_epoch(
                cfg,
                ds_train,
                epoch,
                model,
                optimizer,
                scheduler,
                step_counter,
                loss_fn_global,
                loss_fn_objects,
            )
            run_val_epoch(
                cfg, ds_val, epoch, model, step_counter, loss_fn_global, loss_fn_objects
            )

            if epoch % cfg.data.test.epoch_interval == 0 or is_last or should_stop:
                run_test_segmentation(cfg, ds_test, epoch, model, step_counter)
                run_test_linear_probes(cfg, ds_vqa, epoch, model, step_counter)

            if epoch % cfg.data.viz.epoch_interval == 0 or is_last or should_stop:
                run_viz(
                    cfg,
                    viz_batch,
                    epoch,
                    model,
                    step_counter,
                    loss_fn_global,
                    loss_fn_objects,
                )

            if epoch % cfg.training.checkpoint_interval == 0 or is_last or should_stop:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "steps": int(step_counter),
                    },
                    f"./checkpoint.{epoch}.pth",
                )

            log.info("Done epoch %d in %.1f minutes", epoch, (time.time() - start) / 60)
            if should_stop:
                break
    log.info("Done training for %d/%d epochs", epoch + 1, cfg.training.num_epochs)


# noinspection PyProtectedMember
def run_train_epoch(
    cfg: DictConfig,
    ds_train: Iterable,
    epoch: int,
    model: Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step_counter: StepCounter,
    loss_fn_global: ModelLoss,
    loss_fn_objects: ModelLoss,
):
    """Run one epoch of training.

    Args:
        cfg:
        ds_train:
        epoch:
        model:
        optimizer:
        scheduler:
        step_counter:
        loss_fn_global:
        loss_fn_objects:
    """
    model.train()

    wandb.log({"epoch": epoch}, commit=False, step=int(step_counter))

    last = -1
    bpe = batches_per_epoch_train(cfg)
    epoch_bar = tqdm.tqdm(
        total=cfg.training.batch_size * bpe,
        desc=f"Train {epoch:>4d}",
        unit="img",
        mininterval=10,
        disable=not cfg.other.tqdm,
        bar_format=(
            "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt}, "
            "{elapsed}<{remaining}, {rate_fmt}{postfix}"
        ),
    )

    for batch_idx, images in islice(ds_train, bpe):
        # images: [A B C H W]
        # f_backbone: [AB H'W' C]
        # f_global, p_global: [AB C]
        # f_slots, p_slots: [AB K C]
        B = images.shape[1]
        images = rearrange(images, "A B C H W -> (A B) C H W")
        images = torch.from_numpy(np.copy(images)).to(cfg.other.device)
        output: ModelOutput = model(images)

        loss_dict = {}
        loss = torch.zeros([], device=cfg.other.device)

        l_global = loss_fn_global(output)
        if l_global is not None:
            loss += cfg.losses.l_global.weight * l_global
            loss_dict["l_global/train"] = l_global.item()

        l_objects = loss_fn_objects(output)
        if l_objects is not None:
            loss += cfg.losses.l_objects.weight * l_objects
            loss_dict["l_objects/train"] = l_objects.item()

        wandb.log(
            {**loss_dict, "lr": optimizer.param_groups[0]["lr"]},
            commit=False,
            step=int(step_counter),
        )

        now = time.time()
        if (now - last) > 60:
            epoch_bar.set_postfix({**loss_dict, "steps": int(step_counter)})
            last = now

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step_counter.step()
        epoch_bar.update(B)

    epoch_bar.close()


def batches_per_epoch_train(cfg: DictConfig) -> int:
    """Compute number of batches in one training epoch."""
    return batches_per_epoch(
        cfg.data.train.max_samples, cfg.training.batch_size, drop_last=True
    )


def batches_per_epoch_val(cfg: DictConfig) -> int:
    """Compute number of batches in one validation epoch."""
    return batches_per_epoch(
        cfg.data.val.max_samples, cfg.training.batch_size, drop_last=False
    )


def batches_per_epoch_test(cfg: DictConfig) -> int:
    """Compute number of batches in one test epoch."""
    return batches_per_epoch(
        cfg.data.test.max_samples, cfg.training.batch_size, drop_last=False
    )


@torch.no_grad()
def run_val_epoch(
    cfg: DictConfig,
    ds_val: Iterable,
    epoch: int,
    model: Model,
    step_counter: StepCounter,
    loss_fn_global: ModelLoss,
    loss_fn_objects: ModelLoss,
):
    """Run one validation epoch.

    Args:
        cfg:
        ds_val:
        epoch:
        model:
        step_counter:
        loss_fn_global:
        loss_fn_objects:
    """
    model.eval()

    loss_dict = defaultdict(AverageMetric)

    last = -1
    bpe = batches_per_epoch_val(cfg)
    epoch_bar = tqdm.tqdm(
        total=cfg.data.val.max_samples,
        desc=f"Val   {epoch:>4d}",
        unit="img",
        mininterval=10,
        disable=not cfg.other.tqdm,
        bar_format=(
            "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt}, "
            "{elapsed}<{remaining}, {rate_fmt}{postfix}"
        ),
    )

    for batch_idx, images in islice(ds_val, bpe):
        # images: [A B C H W]
        # f_backbone: [AB H'W' C]
        # f_global, p_global: [AB C]
        # f_slots, p_slots: [AB K C]
        B = images.shape[1]
        images = rearrange(images, "A B C H W -> (A B) C H W")
        images = torch.from_numpy(np.copy(images)).to(cfg.other.device)
        output: ModelOutput = model(images)

        l_global = loss_fn_global(output, reduction="none")
        if l_global is not None:
            loss_dict["l_global/val"].update(l_global)

        l_objects = loss_fn_objects(output, reduction="none")
        if l_objects is not None:
            loss_dict["l_objects/val"].update(l_objects)

        now = time.time()
        if (now - last) > 60:
            epoch_bar.set_postfix(
                {name: f"{am.compute():.4f}" for name, am in loss_dict.items()}
            )
            last = now
        epoch_bar.update(B)

    epoch_bar.close()

    wandb.log(
        {name: am.compute() for name, am in loss_dict.items()},
        commit=False,
        step=int(step_counter),
    )


@torch.no_grad()
def run_viz(
    cfg: DictConfig,
    viz_batch: np.ndarray,
    epoch: int,
    model: Model,
    step_counter: StepCounter,
    loss_fn_global: ModelLoss,
    loss_fn_objects: ModelLoss,
):
    """Run inference on a single batch of images and visualize everything!

    Args:
        cfg:
        viz_batch:
        epoch:
        model:
        step_counter:
        loss_fn_global:
        loss_fn_objects:
    """

    # region Prepare input images, save if it's the first epoch
    images = torch.from_numpy(np.copy(viz_batch))
    images_np = rearrange(
        unnormalize_pt(images, cfg.data.normalize.mean, cfg.data.normalize.std),
        "A B C H W -> B A H W C",
    ).numpy()
    images = images.to(cfg.other.device)
    if epoch == 0:
        array_to_pil(rearrange(images_np, "B A H W C -> (A H) (B W) C")).save(
            "viz_batch.png"
        )
    # endregion

    # region Prepare dirs for epoch and image dict for wandb:
    # - epoch0/*.png
    # - epoch0/viz_batch/**/*.png
    epoch_dir = Path(f"epoch{epoch}/viz_batch")
    epoch_dir.mkdir(exist_ok=True, parents=True)
    wandb_imgs = {}
    # endregion

    # region Positional embedding (downscaled)
    num_patches = np.array(cfg.data.crop_size) // np.array(
        cfg.model.backbone.patch_size
    )
    if isinstance(model.backbone.pos_embed, PositionalEmbedding):
        fig = viz_positional_embedding(
            model.backbone.pos_embed.embed,
            num_patches=num_patches,
            target_patches=(8, 8),
        )
        fig.savefig(f"epoch{epoch}/pos-embed-backbone.png", dpi=100)
        plt.close(fig)
        wandb_imgs["pos-embed/backbone"] = fig
    if model.obj_fn is not None and isinstance(
        model.obj_fn.pos_embed, PositionalEmbedding
    ):
        fig = viz_positional_embedding(
            model.obj_fn.pos_embed.embed,
            num_patches=num_patches,
            target_patches=(8, 8),
        )
        fig.savefig(f"epoch{epoch}/pos-embed-obj-fn.png", dpi=100)
        plt.close(fig)
        wandb_imgs["pos-embed/obj_fn"] = fig
    # endregion

    # region Forward pass and save attn matrices
    model.eval()
    with forward_with_attns(model) as attns:
        output = model(rearrange(images, "A B C H W -> (A B) C H W"))
    torch.save(
        {
            "f_backbone": output.f_backbone,
            "f_global": output.f_global,
            "f_slots": output.f_slots,
            "p_global": output.p_global,
            "p_slots": output.p_slots,
            "attns": attns,
        },
        epoch_dir / "viz.pth",
    )
    # endregion

    # region Global loss
    l_global = loss_fn_global(output)
    if l_global is not None:
        fig = viz_contrastive_loss_global_probs(
            output.p_global, temp=cfg.losses.l_global.temp, loss=l_global.item()
        )
        fig.savefig(epoch_dir / "contrastive-global.png", dpi=200)
        wandb_imgs["losses/l_global"] = fig
        plt.close(fig)
    # endregion

    # region Object loss
    l_objects = loss_fn_objects(output)
    if l_objects is not None:
        fig = viz_contrastive_loss_objects_probs(
            output.p_slots, temp=cfg.losses.l_objects.temp, loss=l_objects.item()
        )
        fig.savefig(epoch_dir / "contrastive-objects.png", dpi=200)
        wandb_imgs["losses/l_objects"] = fig
        plt.close(fig)
    # endregion

    # region Compute rollouts
    A = 2
    B = cfg.data.viz.max_samples

    vit_attns = {k: attns[k] for k in attns.keys() if k.startswith("backbone.")}
    vit_rollout = self_attn_rollout(vit_attns, global_avg_pool=False)
    if cfg.model.backbone.global_pool == "cls":
        # split and re-normalize vit_rollout: [AB 1+Q 1+K] -> [AB 1 K], [AB Q K]
        global_rollout = normalize_sum_to_one(vit_rollout[:, 0, 1:])
        vit_rollout = normalize_sum_to_one(vit_rollout[:, 1:, 1:])
    elif cfg.model.backbone.global_pool == "avg":
        global_rollout = reduce(vit_rollout, "AB Q K -> AB K", reduction="mean")
    else:
        raise ValueError(cfg.model.backbone.global_pool)

    obj_attns = None
    obj_rollout = None
    obj_rollout_full = None
    if cfg.model.obj_fn is not None:
        if cfg.model.obj_fn.name == "slot-attention":
            obj_attns = {
                k: attns[k] for k in attns.keys() if k.startswith("obj_fn.slot_attn.")
            }
            obj_rollout = slot_attn_rollout(obj_attns)
        elif cfg.model.obj_fn.name == "cross-attention":
            obj_attns = {
                k: attns[k] for k in attns.keys() if k.startswith("obj_fn.attn_blocks.")
            }
            obj_rollout = cross_attn_rollout(obj_attns)
        else:
            raise ValueError(cfg.model.obj_fn.name)
        obj_rollout_full = torch.bmm(obj_rollout, vit_rollout)

    if cfg.model.architecture == "backbone-global_fn-global_proj":
        global_rollout = global_rollout
    elif cfg.model.architecture == "backbone(-global_fn-global_proj)-obj_fn-obj_proj":
        global_rollout = global_rollout
    elif cfg.model.architecture == "backbone-obj_fn(-global_fn-global_proj)-obj_proj":
        global_rollout = reduce(obj_rollout_full, "AB S K -> AB K", reduction="mean")
    else:
        raise ValueError(cfg.model.architecture)

    vit_rollout = reduce(vit_rollout, "AB Q K -> AB K", reduction="mean")
    # endregion

    # region Reshape rollouts as images
    K_h = K_w = int(np.sqrt(vit_rollout.shape[-1]))
    shapes = {"A": A, "B": B, "K_h": K_h, "K_w": K_w}
    vit_rollout = rearrange(
        vit_rollout.cpu(), "(A B) (K_h K_w) -> B A K_h K_w", **shapes
    )
    global_rollout = rearrange(
        global_rollout.cpu(), "(A B) (K_h K_w) -> B A K_h K_w", **shapes
    )
    if obj_rollout is not None:
        obj_rollout = rearrange(
            obj_rollout.cpu(), "(A B) S (K_h K_w) -> B A S K_h K_w", **shapes
        )
    if obj_rollout_full is not None:
        obj_rollout_full = rearrange(
            obj_rollout_full.cpu(), "(A B) S (K_h K_w) -> B A S K_h K_w", **shapes
        )
    # endregion

    # region Compute K-means backbone
    if cfg.data.name == "CLEVR10":
        n_clusters = 11
        cmap = "tab20"
    else:
        raise ValueError(cfg.data.name)
    vit_kmeans = kmeans_clusters(output.f_backbone, n_clusters)
    vit_kmeans = rearrange(vit_kmeans, "(A B) (K_h K_w) -> B A K_h K_w", **shapes)
    vit_kmeans = plt.get_cmap(cmap)(vit_kmeans)[..., :3]  # RGB not RGBA
    # endregion

    # region Save individual images
    for b, a in np.ndindex(B, A):
        d = epoch_dir / f"img{b}" / f"aug{a}"
        d.mkdir(exist_ok=True, parents=True)

        # Input image
        img_input = array_to_pil(images_np[b, a])
        img_input.save(d / "input.png")
        WH = img_input.size

        # K-Means of backbone features
        img_kmeans = array_to_pil(vit_kmeans[b, a])
        img_kmeans.save(d / "vit.kmeans.png")
        img_kmeans = img_kmeans.resize(WH, resample=PIL.Image.NEAREST)

        # ViT backbone, one img per head + avg head
        imgs_vit: List[List[PIL.Image.Image]] = [[img_input]]
        for lyr, k in enumerate(vit_attns.keys()):
            imgs_vit.append([])
            attns = vit_attns[k]
            if cfg.model.backbone.global_pool == "cls":
                # split and re-normalize attns: [AB head 1+Q 1+K] -> [AB head Q K]
                attns = normalize_sum_to_one(attns[:, :, 1:, 1:])
            elif cfg.model.backbone.global_pool == "avg":
                pass
            else:
                raise ValueError(cfg.model.backbone.global_pool)

            # Single heads
            for h in range(cfg.model.backbone.num_heads):
                img_vit_head = array_to_pil(
                    reduce(
                        attns[a * B + b, h],
                        "Q (K_h K_w) -> K_h K_w",
                        K_h=K_h,
                        K_w=K_w,
                        reduction="mean",
                    )
                )
                img_vit_head.save(d / f"vit.block{lyr}.head{h}.png")
                img_vit_head = img_vit_head.resize(WH, resample=PIL.Image.NEAREST)
                imgs_vit[-1].append(img_vit_head)

            # Average heads
            img_vit_head = array_to_pil(
                reduce(
                    attns[a * B + b],
                    "heads Q (K_h K_w) -> K_h K_w",
                    K_h=K_h,
                    K_w=K_w,
                    reduction="mean",
                ),
                cmap="inferno",
            )
            img_vit_head.save(d / f"vit.block{lyr}.mean.png")
            img_vit_head = img_vit_head.resize(WH, resample=PIL.Image.NEAREST)
            imgs_vit[-1].insert(0, img_vit_head)

        # ViT rollout
        img_vit_rollout = array_to_pil(vit_rollout[b, a], cmap="inferno")
        img_vit_rollout.save(d / "vit.rollout.png")
        img_vit_rollout = img_vit_rollout.resize(WH, resample=PIL.Image.NEAREST)
        imgs_vit.append([img_vit_rollout])

        # Summary of ViT images for wandb
        imgs_vit: PIL.Image.Image = make_grid_pil(imgs_vit)
        imgs_vit.save(d / "vit.summary.png")
        wandb_imgs[f"vit/img{b}/aug{a}"] = imgs_vit

        # Rollout of global representation
        img_global_rollout = array_to_pil(global_rollout[b, a], cmap="inferno")
        img_global_rollout.save(d / "global.rollout.png")
        img_global_rollout = img_global_rollout.resize(WH, resample=PIL.Image.NEAREST)

        # Object attention, one img per iteration/block per object
        if cfg.model.obj_fn is not None and obj_attns is not None:
            imgs_slot: List[List[PIL.Image.Image]] = [[img_input, img_vit_rollout]]

            if cfg.model.obj_fn.name == "slot-attention":
                for i, k in enumerate(obj_attns.keys()):
                    imgs_slot.append([])
                    for s in range(cfg.model.obj_queries.num_objects):
                        img_slot = array_to_pil(
                            rearrange(
                                obj_attns[k][a * B + b, s],
                                "(K_h K_w) -> K_h K_w",
                                K_h=K_h,
                                K_w=K_w,
                            )
                        )
                        img_slot.save(d / f"obj.slot{s}.iter{i}.png")
                        img_slot = img_slot.resize(WH, resample=PIL.Image.NEAREST)
                        imgs_slot[-1].append(img_slot)
            elif cfg.model.obj_fn.name == "cross-attention":
                for i, k in enumerate(
                    [k for k in obj_attns.keys() if k.endswith(".cross_attn")]
                ):
                    imgs_slot.append([])
                    for s in range(cfg.model.obj_queries.num_objects):
                        img_slot = array_to_pil(
                            reduce(
                                obj_attns[k][a * B + b, :, s],
                                "heads (K_h K_w) -> K_h K_w",
                                K_h=K_h,
                                K_w=K_w,
                                reduction="mean",
                            )
                        )
                        img_slot.save(d / f"obj.cross{s}.block{i}.png")
                        img_slot = img_slot.resize(WH, resample=PIL.Image.NEAREST)
                        imgs_slot[-1].append(img_slot)
            else:
                raise ValueError(cfg.model.obj_fn.name)
            imgs_slot.append([])  # for each slot, rollout of slot attn only
            imgs_slot.append([])  # for each slot, full rollout to the input

            # Rollout of slot attention
            imgs_summary: List[List[PIL.Image.Image]] = [
                [img_input, img_kmeans, img_vit_rollout, img_global_rollout],
                [],  # for each slot, rollout of slot attn only
                [],  # for each slot, full rollout to the input
            ]
            for s in range(cfg.model.obj_queries.num_objects):
                img_slot_rollout = array_to_pil(obj_rollout[b, a, s], cmap="inferno")
                img_slot_rollout.save(d / f"obj.slot{s}.rollout.png")
                img_slot_rollout = img_slot_rollout.resize(
                    WH, resample=PIL.Image.NEAREST
                )
                imgs_slot[-2].append(img_slot_rollout)
                imgs_summary[-2].append(img_slot_rollout)

                img_slot_r_full = array_to_pil(
                    obj_rollout_full[b, a, s], cmap="inferno"
                )
                img_slot_r_full.save(d / f"obj.slot{s}.rollout_full.png")
                img_slot_r_full = img_slot_r_full.resize(WH, resample=PIL.Image.NEAREST)
                imgs_slot[-1].append(img_slot_r_full)
                imgs_summary[-1].append(img_slot_r_full)

            imgs_slot: PIL.Image.Image = make_grid_pil(imgs_slot)
            imgs_slot.save(d / "obj.slots.png")
            wandb_imgs[f"obj/img{b}/aug{a}"] = imgs_slot

            imgs_summary: PIL.Image.Image = make_grid_pil(imgs_summary)
            imgs_summary.save(d / "summary.png")
            wandb_imgs[f"summary/img{b}/aug{a}"] = imgs_summary
    # endregion

    wandb.log(
        {k: wandb.Image(v) for k, v in wandb_imgs.items()},
        step=int(step_counter),
        commit=False,
    )


@torch.no_grad()
def run_test_segmentation(
    cfg: DictConfig,
    ds_test: tf.data.Dataset,
    epoch: int,
    model: Model,
    step_counter: StepCounter,
):
    # Skip segmentation if the models doesn't contain object attention
    if cfg.model.obj_fn is None:
        return

    model.eval()
    num_patches = np.array(cfg.data.crop_size) // np.array(
        cfg.model.backbone.patch_size
    )

    ious = []
    dices = []

    epoch_bar = tqdm.tqdm(
        total=cfg.data.test.max_samples,
        desc=f"Test  {epoch:>4d}",
        unit="img",
        mininterval=10,
        disable=not cfg.other.tqdm,
        bar_format=(
            "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt}, "
            "{elapsed}<{remaining}, {rate_fmt}{postfix}"
        ),
    )
    for batch_idx, examples in ds_test.as_numpy_iterator():
        B, T = examples["visibility"].shape
        images = torch.from_numpy(np.copy(examples["image"]))
        masks = torch.from_numpy(np.copy(examples["mask"]))
        visibility = examples["visibility"]

        with forward_with_attns(model) as attns:
            _ = model(images.to(cfg.other.device))

        if cfg.model.obj_fn.name == "slot-attention":
            obj_rollout = attns["obj_fn.slot_attn.0"]
        elif cfg.model.obj_fn.name == "cross-attention":
            # TODO: check if this works well for segmentation
            obj_rollout = attns["obj_fn.attn_blocks.0.cross_attn"]
            obj_rollout = reduce(obj_rollout, "B heads S K -> B S K", reduction="mean")
        else:
            raise ValueError(cfg.model.obj_fn.name)
        obj_rollout = torch.nn.functional.interpolate(
            rearrange(
                obj_rollout,
                "B S (K_h K_w) -> B S K_h K_w",
                K_h=num_patches[0],
                K_w=num_patches[1],
            ),
            tuple(cfg.data.crop_size),
            mode="nearest",
        )
        obj_rollout_thres = batched_otsu_pt(obj_rollout)
        iou_idx, iou_val, _, dice_val = match_segmentation_masks(
            masks=masks.to(cfg.other.device),
            preds=obj_rollout,
            preds_thres=obj_rollout_thres,
        )
        iou_val = iou_val.cpu().numpy()
        dice_val = dice_val.cpu().numpy()

        ious.extend(iou_val[b, visibility[b]] for b in range(B))
        dices.extend(dice_val[b, visibility[b]] for b in range(B))

        # TODO: handle case where number of GT masks T != number of slots S

        # First batch only, for the first 8 images save a figure with:
        # - input image
        # - attn maps per slot (ordered to match ground-truth)
        # - thresholded attn maps per slot (ordered to match ground-truth)
        # - overlay of img and thresholded attn maps per slot (ordered to match GT)
        # - ground-truth masks and per-slot IoU
        if batch_idx == 0:
            wandb_imgs = {}
            d = Path(f"epoch{epoch}/test")
            d.mkdir(exist_ok=True, parents=True)

            iou_idx = iou_idx.cpu().numpy()
            images = unnormalize_pt(
                images, cfg.data.normalize.mean, cfg.data.normalize.std
            )
            masks = masks.cpu()
            obj_rollout = obj_rollout.cpu()
            obj_rollout_thres = obj_rollout_thres.cpu()

            for b in range(min(8, B)):
                fig, axs = subplots_grid(5, T, ax_height_inch=1.5, dpi=72)
                val_total = np.mean(iou_val[b, 1:], where=visibility[b, 1:])

                axs[0, 0].imshow(to_pil_image(images[b]))
                axs[0, 0].set_title(f"Image {b}")

                # t: index of ground-truth mask, 0, 1, 2, ...
                # s: index of attn mask that best matches with t
                for t in range(T):
                    s = iou_idx[b, t]
                    axs[1, t].imshow(obj_rollout[b, s], interpolation="none")
                    axs[2, t].imshow(obj_rollout_thres[b, s], interpolation="none")
                    axs[3, t].imshow(
                        to_pil_image(
                            torch.where(
                                obj_rollout_thres[b, s], images[b], 0.3 * images[b]
                            )
                        ),
                    )
                    axs[4, t].imshow(masks[b, t], interpolation="none")
                    axs[4, t].set_xlabel(f"Mask {t} slot {s}\nIoU {iou_val[b, t]:.2f}")

                axs[1, 0].set_ylabel("Attn")
                axs[2, 0].set_ylabel("Thresholded")
                axs[3, 0].set_ylabel("Overlay")
                axs[4, 0].set_ylabel(f"Ground-truth\nIoU total {val_total:.2f}")

                for ax in axs.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                for ax in axs[0, 1:]:
                    ax.set_axis_off()

                fig.set_facecolor("white")
                fig.tight_layout(pad=0, h_pad=0, w_pad=0)
                fig.savefig(d / f"img{b}.iou.png")
                wandb_imgs[f"test/iou/img{b}"] = fig
                plt.close(fig)
            wandb.log(
                {k: wandb.Image(v) for k, v in wandb_imgs.items()},
                step=int(step_counter),
                commit=False,
            )

        epoch_bar.update(B)

    epoch_bar.close()
    wandb.log(
        {
            "test/iou/by_img_no_bg": np.mean(
                [img_ious[1:].mean() for img_ious in ious]
            ),
            "test/dice/by_img_no_bg": np.mean(
                [img_dices[1:].mean() for img_dices in dices]
            ),
        },
        commit=False,
        step=int(step_counter),
    )


def run_test_linear_probes(
    cfg: DictConfig,
    ds_vqa: tf.data.Dataset,
    epoch: int,
    model: Model,
    step_counter: StepCounter,
):
    wandb_imgs = {}

    features, targets = extract_vqa_features(cfg, ds_vqa, epoch, model)
    split = int(cfg.data.test.vqa.split * len(features))
    features_train = features[:split].to(cfg.other.device)
    features_test = features[split:].to(cfg.other.device)
    targets_train = targets[:split].float().to(cfg.other.device)
    targets_test = targets[split:].float().to(cfg.other.device)

    # For each True label there are 14 False labels
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=(
            None
            if cfg.data.test.vqa.pos_weight is None
            else torch.tensor(cfg.data.test.vqa.pos_weight)
        )
    )
    torch.manual_seed(cfg.data.test.vqa.seed)
    probe = build_linear_probe(cfg, targets.shape[-1]).to(cfg.other.device)
    optimizer = torch.optim.AdamW(probe.parameters())

    # region Train
    hist = []
    for i in tqdm.trange(
        cfg.data.test.vqa.num_epochs,
        desc=f"VQA p {epoch:>4d}",
        ncols=0,
        mininterval=10,
        disable=not cfg.other.tqdm,
    ):
        with torch.no_grad():
            preds = probe(features_test)
            loss_test = loss_fn(preds.flatten(), targets_test.flatten())

        optimizer.zero_grad()
        preds = probe(features_train)
        loss_train = loss_fn(preds.flatten(), targets_train.flatten())
        loss_train.backward()
        optimizer.step()

        hist.append((i, loss_train.item(), loss_test.item()))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    df = pd.DataFrame(hist, columns=["step", "train", "val"]).set_index("step")
    df.plot(ax=axs[0])
    df[cfg.data.test.vqa.num_epochs // 2 :].plot(ax=axs[1])
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Linear probe optimization")
    fig.set_facecolor("white")
    wandb_imgs["vqa/linear"] = wandb.Image(fig)
    # endregion

    with torch.no_grad():
        preds = probe(features_test)
        loss_test = loss_fn(preds, targets_test).item()
        targets_test = targets_test.cpu().numpy().astype(bool)
        sample_weight = np.where(targets_test.flatten(), 14.0, 1.0)
        preds = preds.sigmoid_().cpu().numpy()

    for plot_name, plot_fn in [
        ("vqa/pr", PrecisionRecallDisplay.from_predictions),
        ("vqa/roc", RocCurveDisplay.from_predictions),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_fn(
            y_true=targets_test.flatten(),
            y_pred=preds.flatten(),
            ax=ax,
            name="Not adjusted",
        )
        plot_fn(
            y_true=targets_test.flatten(),
            y_pred=preds.flatten(),
            sample_weight=sample_weight,
            ax=ax,
            name="Adjusted",
        )
        fig.set_facecolor("white")
        wandb_imgs[plot_name] = wandb.Image(fig)

    ap_balanced = average_precision_score(
        y_true=targets_test.flatten(),
        y_score=preds.flatten(),
        sample_weight=sample_weight,
    )
    ap_imbalanced = average_precision_score(
        y_true=targets_test.flatten(),
        y_score=preds.flatten(),
        sample_weight=None,
    )
    auroc_balanced = roc_auc_score(
        y_true=targets_test.flatten(),
        y_score=preds.flatten(),
        sample_weight=sample_weight,
    )
    auroc_imbalanced = roc_auc_score(
        y_true=targets_test.flatten(),
        y_score=preds.flatten(),
        sample_weight=None,
    )

    log.info(
        "VQA: loss %.3f, average precision %.2f, auroc %.2f",
        loss_test,
        100 * ap_balanced,
        100 * auroc_balanced,
    )
    wandb.log(
        {
            "vqa/loss": loss_test,
            "vqa/ap": ap_balanced,
            "vqa/ap/imbalanced": ap_imbalanced,
            "vqa/auroc": auroc_balanced,
            "vqa/auroc/imbalanced": auroc_imbalanced,
            **wandb_imgs,
        },
        commit=False,
        step=int(step_counter),
    )


def build_linear_probe(cfg, output_classes: int) -> torch.nn.Module:
    D = cfg.model.backbone.embed_dim

    if cfg.model.architecture == "backbone-global_fn-global_proj":
        return torch.nn.Linear(D, output_classes)

    if cfg.model.architecture in {
        "backbone(-global_fn-global_proj)-obj_fn-obj_proj",
        "backbone-obj_fn(-global_fn-global_proj)-obj_proj",
    }:
        return torch.nn.Sequential(
            torch.nn.Linear(D, output_classes),
            Reduce("B S C -> B C", reduction="max"),
        )

    raise ValueError(cfg.model.architecture)


@torch.no_grad()
def extract_vqa_features(
    cfg: DictConfig, ds_vqa: tf.data.Dataset, epoch: int, model: Model
) -> Tuple[Tensor, Tensor]:
    features = []
    targets = []
    model.eval()
    if cfg.model.architecture == "backbone-global_fn-global_proj":
        get_feats: Callable[[ModelOutput], Tensor] = attrgetter("f_global")
    elif cfg.model.architecture in {
        "backbone(-global_fn-global_proj)-obj_fn-obj_proj",
        "backbone-obj_fn(-global_fn-global_proj)-obj_proj",
    }:
        get_feats: Callable[[ModelOutput], Tensor] = attrgetter("f_slots")
    else:
        raise ValueError(cfg.model.architecture)

    epoch_bar = tqdm.tqdm(
        total=cfg.data.test.max_samples,
        desc=f"VQA f {epoch:>4d}",
        unit="img",
        mininterval=10,
        disable=not cfg.other.tqdm,
        bar_format=(
            "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt}, "
            "{elapsed}<{remaining}, {rate_fmt}{postfix}"
        ),
    )
    for batch_idx, examples in ds_vqa.as_numpy_iterator():
        images = torch.from_numpy(np.copy(examples["image"]))
        output = model(images.to(cfg.other.device))
        features.append(get_feats(output).cpu())
        targets.append(examples["vqa_target"])
        epoch_bar.update(images.shape[0])
    epoch_bar.close()

    features = torch.cat(features)
    targets = torch.from_numpy(np.concatenate(targets))
    return features, targets
