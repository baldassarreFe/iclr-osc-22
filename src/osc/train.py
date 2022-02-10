import logging
import random
import time
from functools import partial
from itertools import islice
from pathlib import Path

import einops
import hydra
import namesgenerator
import numpy as np
import omegaconf
import PIL.Image as PilImage
import tabulate
import tensorflow as tf
import torch
import tqdm
import wandb
import wandb.util
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

import osc.data.clevr_with_masks
from osc.data.tfrecords import deserialize_image
from osc.data.utils import augment_train, augment_twice, unnormalize_pt, wrap_with_seed
from osc.loss_global import contrastive_loss, cosine_sim_loss
from osc.loss_objects import (
    matching_contrastive_loss,
    matching_contrastive_loss_per_img,
)
from osc.lr_scheduler import LinearWarmupCosineAnneal
from osc.models import (
    MLP,
    PositionalEmbedding,
    SlotAttention,
    ViTBackbone,
    VitGlobalSlotModel,
    VitSlotGlobalModel,
    global_avg_pool,
    global_max_pool,
)
from osc.models.models import vit_slot_forward_with_attns
from osc.rollout import self_attn_rollout, slot_attn_rollout
from osc.utils import SigIntCatcher, batches_per_epoch, seed_everything
from osc.viz import (
    array_to_pil,
    kmeans_clusters,
    make_grid_pil,
    viz_contrastive_loss_global_probs,
    viz_contrastive_loss_objects_probs,
    viz_positional_embedding,
)
from osc.wandb_utils import setup_wandb

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    log.info("Config original:\n%s", OmegaConf.to_yaml(cfg))
    cfg = update_cfg(cfg)
    with open("train.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(cfg.other.seed)
    setup_wandb(cfg)

    ds_train = build_dataset_train(cfg)
    ds_val = build_dataset_val(cfg)
    viz_batch = get_viz_batch(cfg)

    model = build_model(cfg).to(cfg.other.device)
    log.info(
        "Model parameters:\n%s",
        tabulate.tabulate(
            [
                [name, sum(p.numel() for p in child.parameters() if p.requires_grad)]
                for name, child in model.named_children()
            ],
            headers=["Module", "Parameters"],
        ),
    )

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn_global, loss_fn_objects = build_losses(cfg)
    run_train_val_viz_epochs(
        cfg,
        model,
        optimizer,
        scheduler,
        ds_train,
        ds_val,
        viz_batch,
        loss_fn_global,
        loss_fn_objects,
    )

    log.info("Run dir: %s", Path.cwd().relative_to(hydra.utils.get_original_cwd()))


def update_cfg(cfg: DictConfig, readonly=True):
    up = DictConfig({})

    if cfg.other.debug:
        OmegaConf.update(up, "training.num_epochs", 3)
        OmegaConf.update(up, "data.train.max_samples", 10 * cfg.training.batch_size)
        OmegaConf.update(up, "data.val.max_samples", 10 * cfg.training.batch_size)
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

        if cfg.other.seed is None:
            OmegaConf.update(up, "other.seed", random.randint(0, 2 ** 32))

        if cfg.data.train.seed is None:
            OmegaConf.update(up, "data.train.seed", random.randint(0, 2 ** 32))

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


def build_model(cfg):
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

    if cfg.model.obj_fn.pos_embed == "backbone":
        obj_fn_pos_embed = backbone_pos_embed
    elif cfg.model.obj_fn.pos_embed == "learned":
        obj_fn_pos_embed = PositionalEmbedding(
            np.prod(num_patches),
            cfg.model.backbone.embed_dim,
            dropout=cfg.model.backbone.pos_embed_dropout,
        )
    elif cfg.model.obj_fn.pos_embed is None:
        obj_fn_pos_embed = None
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
    )

    global_fn = torch.nn.Sequential(
        {
            "avg": global_avg_pool,
            "max": global_max_pool,
        }[cfg.model.global_fn.pooling],
        MLP(
            in_features=cfg.model.backbone.embed_dim,
            hidden_features=int(
                np.round(cfg.model.global_fn.hidden_mult * cfg.model.backbone.embed_dim)
            ),
            out_features=cfg.model.backbone.embed_dim,
            activation=activations[cfg.model.global_fn.activation],
            dropout=0.0,
        ),
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

    if cfg.model.obj_fn.name == "slot-attention":
        if cfg.model.obj_fn.queries == "sample":
            obj_fn = SlotAttention(
                num_slots=cfg.model.obj_fn.num_objects,
                dim=cfg.model.backbone.embed_dim,
                pos_embed=obj_fn_pos_embed,
                iters=cfg.model.obj_fn.num_iters,
                hidden_dim=int(
                    np.round(
                        cfg.model.obj_fn.hidden_mult * cfg.model.backbone.embed_dim
                    )
                ),
            )
        else:
            raise ValueError(cfg.model.obj_fn.queries)
    else:
        raise ValueError(cfg.model.obj_fn.name)

    obj_proj = MLP(
        in_features=cfg.model.backbone.embed_dim,
        hidden_features=4 * cfg.model.backbone.embed_dim,
        out_features=cfg.model.backbone.embed_dim,
        activation=activations[cfg.model.obj_proj.activation],
        out_bias=False,
        dropout=0.0,
    )

    if cfg.model.architecture == "backbone(-global_fn-global_proj)-obj_fn-obj_proj":
        return VitGlobalSlotModel(
            backbone=backbone,
            global_fn=global_fn,
            obj_fn=obj_fn,
            global_proj=global_proj,
            obj_proj=obj_proj,
        )

    elif cfg.model.architecture == "backbone-obj_fn(-global_fn-global_proj)-obj_proj":
        return VitSlotGlobalModel(
            backbone=backbone,
            global_fn=global_fn,
            obj_fn=obj_fn,
            global_proj=global_proj,
            obj_proj=obj_proj,
        )

    else:
        raise ValueError(cfg.model.architecture)


def build_optimizer(cfg, model):
    if cfg.optimizer.name == "adam":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.start_lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
    raise ValueError(cfg.optimizer.name)


def build_scheduler(cfg, optimizer):
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


def build_losses(cfg):
    if cfg.losses.l_global.name == "sim":

        def loss_fn_global(f_global, f_slots, p_global, p_slots, reduction="mean"):
            del f_slots, p_slots
            return cosine_sim_loss(f_global, p_global, reduction=reduction)

    elif cfg.losses.l_global.name == "ctr":

        def loss_fn_global(f_global, f_slots, p_global, p_slots, reduction="mean"):
            del f_global, f_slots, p_slots
            return contrastive_loss(
                p_global, temperature=cfg.losses.l_global.temp, reduction=reduction
            )

    else:
        raise ValueError(cfg.losses.l_global.name)

    if cfg.losses.l_objects.name == "ctr_img":

        def loss_fn_objects(f_global, f_slots, p_global, p_slots, reduction="mean"):
            del f_global, f_slots, p_global
            return matching_contrastive_loss_per_img(
                p_slots, temperature=cfg.losses.l_objects.temp, reduction=reduction
            )

    elif cfg.losses.l_objects.name == "ctr_all":

        def loss_fn_objects(f_global, f_slots, p_global, p_slots, reduction="mean"):
            del f_global, f_slots, p_global
            return matching_contrastive_loss(
                p_slots, temperature=cfg.losses.l_objects.temp, reduction=reduction
            )

    else:
        raise ValueError(cfg.losses.l_global.name)

    return loss_fn_global, loss_fn_objects


def build_dataset_train(cfg):
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


def build_dataset_val(cfg):
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


def get_viz_batch(cfg):
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
    cfg,
    model,
    optimizer,
    scheduler,
    ds_train,
    ds_val,
    viz_batch,
    loss_fn_global,
    loss_fn_objects,
):
    log.info("Start training for %d epochs", cfg.training.num_epochs)
    step_counter = StepCounter()
    with SigIntCatcher() as should_stop:
        for epoch in range(cfg.training.num_epochs):
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
            run_viz(
                cfg,
                viz_batch,
                epoch,
                model,
                step_counter,
                loss_fn_global,
                loss_fn_objects,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "steps": int(step_counter),
                },
                f"./checkpoint.{epoch}.pth",
            )
            if should_stop:
                break
    log.info("Done training for %d/%d epochs", epoch + 1, cfg.training.num_epochs)


def run_train_epoch(
    cfg,
    ds_train,
    epoch,
    model,
    optimizer,
    scheduler,
    step_counter,
    loss_fn_global,
    loss_fn_objects,
):
    model.train()

    wandb.log({"epoch": epoch}, commit=False, step=int(step_counter))

    last = -1
    bpe = batches_per_epoch_train(cfg)
    epoch_bar = tqdm.tqdm(
        total=cfg.training.batch_size * bpe,
        desc=f"T{epoch:>4d}",
        unit="img",
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
        images = torch.from_numpy(np.copy(images)).to(cfg.other.device)
        f_backbone, f_global, f_slots, p_global, p_slots = model(
            einops.rearrange(images, "A B C H W -> (A B) C H W")
        )

        l_global = loss_fn_global(f_global, f_slots, p_global, p_slots)
        l_objects = loss_fn_objects(f_global, f_slots, p_global, p_slots)
        loss = (
            cfg.losses.l_global.weight * l_global
            + cfg.losses.l_objects.weight * l_objects
        )

        wandb.log(
            {
                "l_global/train": l_global.item(),
                "l_objects/train": l_objects.item(),
                "lr": optimizer.param_groups[0]["lr"],
            },
            commit=False,
            step=int(step_counter),
        )

        now = time.time()
        if (now - last) > 60:
            epoch_bar.set_postfix(
                {
                    "l_global": f"{l_global.item():.4f}",
                    "l_objects": f"{l_objects.item():.4f}",
                    "steps": int(step_counter),
                }
            )
            last = now

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step_counter.step()

        epoch_bar.update(images.shape[1])

    epoch_bar.close()


def batches_per_epoch_train(cfg):
    return batches_per_epoch(
        cfg.data.train.max_samples, cfg.training.batch_size, drop_last=True
    )


def batches_per_epoch_val(cfg):
    return batches_per_epoch(
        cfg.data.val.max_samples, cfg.training.batch_size, drop_last=False
    )


@torch.no_grad()
def run_val_epoch(
    cfg, ds_val, epoch, model, step_counter, loss_fn_global, loss_fn_objects
):
    model.eval()

    l_global = 0
    l_global_div = 0
    l_objects = 0
    l_objects_div = 0

    last = -1
    bpe = batches_per_epoch_val(cfg)
    epoch_bar = tqdm.tqdm(
        total=cfg.data.val.max_samples,
        desc=f"V{epoch:>4d}",
        unit="img",
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
        images = torch.from_numpy(np.copy(images)).to(cfg.other.device)
        f_backbone, f_global, f_slots, p_global, p_slots = model(
            einops.rearrange(images, "A B C H W -> (A B) C H W")
        )

        l_global += loss_fn_global(
            f_global, f_slots, p_global, p_slots, reduction="sum"
        ).item()
        l_global_div += p_global.shape[0]

        l_objects += loss_fn_objects(
            f_global, f_slots, p_global, p_slots, reduction="sum"
        ).item()
        l_objects_div += p_slots.shape[0] * p_slots.shape[1]

        now = time.time()
        if (now - last) > 60:
            epoch_bar.set_postfix(
                {
                    "l_global": f"{l_global / l_global_div:.4f}",
                    "l_objects": f"{l_objects / l_objects_div:.4f}",
                }
            )
            last = now
        epoch_bar.update(images.shape[1])

    epoch_bar.close()

    wandb.log(
        {
            "l_global/val": l_global / l_global_div,
            "l_objects/val": l_objects / l_objects_div,
        },
        commit=False,
        step=int(step_counter),
    )


@torch.no_grad()
def run_viz(
    cfg, viz_batch, epoch, model, step_counter, loss_fn_global, loss_fn_objects
):
    # region Prepare input images, save if it's the first epoch
    images = torch.from_numpy(np.copy(viz_batch)).to(cfg.other.device)
    images_np = (
        einops.rearrange(
            unnormalize_pt(images, cfg.data.normalize.mean, cfg.data.normalize.std),
            "A B C H W -> B A H W C",
        )
        .cpu()
        .numpy()
    )
    if epoch == 0:
        array_to_pil(einops.rearrange(images_np, "B A H W C -> (A H) (B W) C")).save(
            "viz_batch.png"
        )
    # endregion

    # region Prepare dirs for epoch and image dict for wandb:
    # - epoch0/*.png
    # - epoch0/viz_batch/**/*.png
    d = Path(f"epoch{epoch}/viz_batch")
    d.mkdir(exist_ok=True, parents=True)
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
        wandb_imgs["pos-embed/backbone"] = fig  # f"epoch{epoch}/pos-embed-backbone.png"
    if isinstance(model.obj_fn.pos_embed, PositionalEmbedding):
        fig = viz_positional_embedding(
            model.obj_fn.pos_embed.embed,
            num_patches=num_patches,
            target_patches=(8, 8),
        )
        fig.savefig(f"epoch{epoch}/pos-embed-obj-fn.png", dpi=100)
        plt.close(fig)
        wandb_imgs["pos_embed/obj_fn"] = fig  # f"epoch{epoch}/pos-embed-obj-fn.png"
    # endregion

    # region Forward pass and save attn matrices
    model.eval()
    with vit_slot_forward_with_attns(model) as f:
        (
            (f_backbone, f_global, f_slots, p_global, p_slots),
            vit_attns,
            slot_attns,
        ) = f(einops.rearrange(images, "A B C H W -> (A B) C H W"))
    # endregion

    # region Global loss
    l_global = loss_fn_global(f_global, f_slots, p_global, p_slots).item()
    fig = viz_contrastive_loss_global_probs(
        p_global, temp=cfg.losses.l_global.temp, loss=l_global
    )
    fig.savefig(d / "contrastive-global.png", dpi=200)
    wandb_imgs["viz_batch/l_global"] = fig
    plt.close(fig)
    # endregion

    # region Object loss
    l_objects = loss_fn_objects(f_global, f_slots, p_global, p_slots).item()
    fig = viz_contrastive_loss_objects_probs(
        p_slots, temp=cfg.losses.l_objects.temp, loss=l_objects
    )
    fig.savefig(d / "contrastive-objects.png", dpi=200)
    wandb_imgs["viz_batch/l_objects"] = fig
    plt.close(fig)
    # endregion

    # region Compute rollouts
    A = 2
    B = cfg.data.viz.max_samples
    S = cfg.model.obj_fn.num_objects

    vit_rollout = self_attn_rollout(vit_attns, global_avg_pool=False)

    slot_rollout = slot_attn_rollout(slot_attns)
    slot_rollout_full = torch.bmm(slot_rollout, vit_rollout)

    if cfg.model.architecture == "backbone(-global_fn-global_proj)-obj_fn-obj_proj":
        global_rollout = einops.reduce(vit_rollout, "AB Q K -> AB K", reduction="mean")
    elif cfg.model.architecture == "backbone-obj_fn(-global_fn-global_proj)-obj_proj":
        global_rollout = einops.reduce(
            slot_rollout_full, "AB S K -> AB K", reduction="mean"
        )
    else:
        raise ValueError(cfg.model.architecture)

    vit_rollout = einops.reduce(vit_rollout, "AB Q K -> AB K", reduction="mean")
    # endregion

    # region Reshape rollouts as images
    K_h = K_w = int(np.sqrt(vit_rollout.shape[-1]))
    shapes = {"A": A, "B": B, "K_h": K_h, "K_w": K_w}
    vit_rollout = einops.rearrange(
        vit_rollout.cpu().numpy(), "(A B) (K_h K_w) -> B A K_h K_w", **shapes
    )
    global_rollout = einops.rearrange(
        global_rollout.cpu().numpy(), "(A B) (K_h K_w) -> B A K_h K_w", **shapes
    )
    slot_rollout = einops.rearrange(
        slot_rollout.cpu().numpy(), "(A B) S (K_h K_w) -> B A S K_h K_w", **shapes
    )
    slot_rollout_full = einops.rearrange(
        slot_rollout_full.cpu().numpy(), "(A B) S (K_h K_w) -> B A S K_h K_w", **shapes
    )
    # endregion

    # region Compute K-means backbone
    if cfg.data.name == "CLEVR10":
        n_clusters = 11
        cmap = "tab20"
    else:
        raise ValueError(cfg.data.name)
    vit_kmeans = kmeans_clusters(f_backbone, n_clusters)
    vit_kmeans = einops.rearrange(
        vit_kmeans, "(A B) (K_h K_w) -> B A K_h K_w", **shapes
    )
    vit_kmeans = plt.get_cmap(cmap)(vit_kmeans)[..., :3]  # RGB not RGBA
    # endregion

    # region Save individual images
    for b, a in np.ndindex(B, A):
        Path(d / f"img{b}/aug{a}").mkdir(exist_ok=True, parents=True)

        # Input image
        img = array_to_pil(images_np[b, a])
        img.save(d / f"img{b}/aug{a}/input.png")
        WH = img.size

        # K-Means of backbone fetures
        kmeans = array_to_pil(vit_kmeans[b, a])
        kmeans.save(d / f"img{b}/aug{a}/vit.kmeans.png")
        kmeans = kmeans.resize(WH, resample=PilImage.NEAREST)

        # ViT backbone, one img per head + avg head
        for lyr, k in enumerate(sorted(vit_attns.keys())):
            for h in range(vit_attns[k].shape[1]):
                array_to_pil(
                    vit_attns[k][a * B + b, h].mean(axis=0).reshape(K_h, K_w)
                ).save(d / f"img{b}/aug{a}/vit.block{lyr}.head{h}.png")
            array_to_pil(
                vit_attns[k][a * B + b].mean(axis=(0, 1)).reshape(K_h, K_w),
                cmap="inferno",
            ).save(d / f"img{b}/aug{a}/vit.block{lyr}.mean.png")

        # ViT rollout
        vit_r = array_to_pil(vit_rollout[b, a], cmap="inferno")
        vit_r.save(d / f"img{b}/aug{a}/vit.rollout.png")
        vit_r = vit_r.resize(WH, resample=PilImage.NEAREST)

        # Slot attn backbone, one img per slot per iteration,
        # one rollout img per slot, one full rollout img per slot
        slot_r = []
        slot_r_f = []
        for s in range(S):
            for i, k in enumerate(sorted(slot_attns.keys())):
                array_to_pil(slot_attns[k][a * B + b, s].reshape(K_h, K_w)).save(
                    d / f"img{b}/aug{a}/obj.slot{s}.iter{i}.png"
                )

            x = array_to_pil(slot_rollout[b, a, s], cmap="inferno")
            x.save(d / f"img{b}/aug{a}/obj.slot{s}.rollout.png")
            slot_r.append(x.resize(WH, resample=PilImage.NEAREST))

            x = array_to_pil(slot_rollout_full[b, a, s], cmap="inferno")
            x.save(d / f"img{b}/aug{a}/obj.slot{s}.rollout_full.png")
            slot_r_f.append(x.resize(WH, resample=PilImage.NEAREST))

        # Global rollout
        global_r = array_to_pil(global_rollout[b, a], cmap="inferno")
        global_r.save(d / f"img{b}/aug{a}/global.rollout.png")
        global_r = global_r.resize(WH, resample=PilImage.NEAREST)

        # Summary image for wandb
        grid = make_grid_pil(
            [
                [img, kmeans, vit_r, global_r],
                slot_r,
                slot_r_f,
            ]
        )
        grid.save(d / f"img{b}/aug{a}/summary.png")
        wandb_imgs[f"img{b}/aug{a}"] = grid
    # endregion

    wandb.log(
        {k: wandb.Image(v) for k, v in wandb_imgs.items()},
        step=int(step_counter),
        commit=False,
    )


class StepCounter(object):
    def __init__(self):
        self.steps = 0

    def __int__(self):
        return self.steps

    def step(self):
        self.steps += 1

    def __str__(self):
        return f"{self.__class__.__name__}({int(self)})"
