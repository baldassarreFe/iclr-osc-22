"""
Main training methods.
"""
import logging
import math
import os
import random
import warnings
from collections import defaultdict
from contextlib import suppress
from functools import partial
from itertools import chain, islice
from pathlib import Path
from typing import Iterable, List, Optional

import hydra
import namesgenerator
import numpy as np
import omegaconf
import PIL.Image
import submitit
import tabulate
import tensorflow as tf
import timm
import torch
import torch.nn.functional
import tqdm
import wandb
import wandb.util
from einops import rearrange, reduce
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from timm.optim import create_optimizer_v2
from torch import Tensor

import osc.data.clevr_with_masks
from osc.data.augmentations import augment_train, zip_with_seeds
from osc.data.random_boxes import generate_random_boxes
from osc.data.tfrecords import deserialize_image
from osc.data.utils import unnormalize_pt
from osc.losses.loss_global import compute_losses_global
from osc.losses.loss_objects import compute_losses_objects
from osc.lr_scheduler import MySequentialLR, build_scheduler
from osc.models.builders import build_core_model
from osc.models.core_model import CoreModel, ModelOutput, RecordAttentions
from osc.segmentation import build_dataset_segmentation, run_test_segmentation
from osc.utils import (
    AverageMetric,
    SigIntCatcher,
    StepCounter,
    TimerCollection,
    batches_per_epoch,
    check_num_samples,
    seed_everything,
    tf_no_gpus,
)
from osc.viz.backbone import kmeans_clusters
from osc.viz.embeds import viz_positional_embedding
from osc.viz.matches import viz_matches_global, viz_matches_object
from osc.viz.utils import array_to_pil, make_grid_pil
from osc.vqa import build_dataset_vqa, run_test_vqa
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
    tf_no_gpus()

    ds_train = build_dataset_train(cfg)
    ds_val = build_dataset_val(cfg)
    ds_segm = build_dataset_segmentation(cfg)
    ds_vqa = build_dataset_vqa(cfg)
    viz_batch = get_viz_batch(cfg, ds_val)

    model = build_core_model(cfg).to(cfg.other.device)
    log_model_parameters(model)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    run_train_val_test(
        cfg,
        model,
        optimizer,
        scheduler,
        ds_train,
        ds_val,
        ds_segm,
        ds_vqa,
        viz_batch,
    )

    log.info("Run dir: %s", Path.cwd().relative_to(hydra.utils.get_original_cwd()))


def update_cfg(cfg: DictConfig, readonly=True):
    up = DictConfig({})

    if cfg.data.train.max_samples is None:
        if cfg.data.name == "CLEVR10":
            OmegaConf.update(
                up,
                "data.train.max_samples",
                osc.data.clevr_with_masks.NUM_SAMPLES_TRAIN,
            )
        else:
            raise ValueError(cfg.data.name)

    if cfg.data.train.seed is None:
        OmegaConf.update(up, "data.train.seed", random.randint(0, np.power(2, 32)))

    if cfg.data.val.max_samples is None:
        if cfg.data.name == "CLEVR10":
            OmegaConf.update(
                up,
                "data.val.max_samples",
                osc.data.clevr_with_masks.NUM_SAMPLES_VAL,
            )
        else:
            raise ValueError(cfg.data.name)

    if cfg.data.test.segmentation.max_samples is None:
        if cfg.data.name == "CLEVR10":
            OmegaConf.update(
                up,
                "data.test.segmentation.max_samples",
                osc.data.clevr_with_masks.NUM_SAMPLES_TEST,
            )
        else:
            raise ValueError(cfg.data.name)

    if cfg.data.test.vqa.max_samples is None:
        if cfg.data.name == "CLEVR10":
            OmegaConf.update(
                up,
                "data.test.vqa.max_samples",
                osc.data.clevr_with_masks.NUM_SAMPLES_TEST,
            )
        else:
            raise ValueError(cfg.data.name)

    if cfg.data.normalize.mean is None and cfg.data.normalize.std is None:
        if cfg.model.backbone.pretrained is None:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            if cfg.model.backbone.name == "vit":
                pretrained_cfgs = timm.models.vision_transformer.default_cfgs
            elif cfg.model.backbone.name == "swin":
                pretrained_cfgs = timm.models.swin_transformer.default_cfgs
            else:
                raise NotImplementedError(cfg.model.backbone.name)
            mean = pretrained_cfgs[cfg.model.backbone.pretrained]["mean"]
            std = pretrained_cfgs[cfg.model.backbone.pretrained]["std"]
        OmegaConf.update(up, "data.normalize.mean", mean)
        OmegaConf.update(up, "data.normalize.std", std)

    if cfg.logging.id is None:
        OmegaConf.update(up, "logging.id", wandb.util.generate_id())

    if cfg.logging.name is None:
        name = namesgenerator.get_random_name("-") + f"-{random.randint(0, 99):<02d}"
        OmegaConf.update(up, "logging.name", name)

    if cfg.other.seed is None:
        OmegaConf.update(up, "other.seed", random.randint(0, np.power(2, 32)))

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


def log_model_parameters(model: torch.nn.Module):
    """Log model parameters as a table. First level only."""
    headers = ["Module", "Class", "Parameters", "Trainable"]
    table = [
        [
            name,
            child.__class__.__name__,
            sum(p.numel() for p in child.parameters()),
            sum(p.numel() for p in child.parameters() if p.requires_grad),
        ]
        for name, child in model.named_children()
    ]
    table.append(["TOTAL", "", sum(r[-2] for r in table), sum(r[-1] for r in table)])
    table = tabulate.tabulate(table, headers=headers)
    log.info("Model parameters:\n%s", table)


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build optimizer for training.

    Args:
        cfg: configuration
        model: model

    Returns:
        An optimizer.
    """
    return create_optimizer_v2(
        model,
        opt=cfg.optimizer.name,
        lr=cfg.optimizer.start_lr,
        weight_decay=cfg.optimizer.weight_decay,
        filter_bias_and_bn=True,
    )


MAP_KW = {"num_parallel_calls": tf.data.AUTOTUNE, "deterministic": True}


def build_dataset_train(cfg) -> Iterable[np.ndarray]:
    """Build training dataset.

    Args:
        cfg: configuration

    Returns:
        Training dataset.
    """
    if cfg.data.name == "CLEVR10":
        tfr_path = Path(cfg.data.root) / "clevr_with_masks" / "imgs_train.tfrecords"
        tfr_path = tfr_path.expanduser().resolve().as_posix()
        img_size = osc.data.clevr_with_masks.IMAGE_SIZE
        check_num_samples(
            cfg.data.train.max_samples, osc.data.clevr_with_masks.NUM_SAMPLES_TRAIN
        )
    else:
        raise NotImplementedError(cfg.data.name)

    ds = tf.data.TFRecordDataset(tfr_path, compression_type="GZIP")
    # Limit number of samples
    ds = ds.take(cfg.data.train.max_samples)
    # Decode image
    ds = ds.map(partial(deserialize_image, img_size=img_size), **MAP_KW)
    # Shuffle using a buffer
    ds = ds.shuffle(
        cfg.data.train.shuffle_batches * cfg.training.batch_size,
        seed=cfg.data.train.seed,
    )
    # Repeat forever and add seeds for augmentations
    ds = zip_with_seeds(ds.repeat(), cfg.data.train.seed + 1)
    # Augment large and small crops [H W C] -> ([num_large C H W], [num_small C H W])
    augment_fn = partial(
        augment_train,
        large_boxes=tf.convert_to_tensor(
            generate_random_boxes(
                area_min=cfg.data.crops.large.area[0],
                area_max=cfg.data.crops.large.area[1],
                ratio_min=cfg.data.crops.large.ratio[0],
                ratio_max=cfg.data.crops.large.ratio[1],
                num_boxes=1_000_000,
                seed=cfg.data.train.seed + 1,
            ),
            dtype=tf.float32,
        ),
        large_num_crops=cfg.data.crops.large.num,
        large_strength=cfg.data.crops.large.strength,
        large_crop_hw=tuple(cfg.data.crops.large.size),
        small_boxes=tf.convert_to_tensor(
            generate_random_boxes(
                area_min=cfg.data.crops.small.area[0],
                area_max=cfg.data.crops.small.area[1],
                ratio_min=cfg.data.crops.small.ratio[0],
                ratio_max=cfg.data.crops.small.ratio[1],
                num_boxes=1_000_000,
                seed=cfg.data.train.seed + 2,
            ),
            dtype=tf.float32,
        ),
        small_num_crops=cfg.data.crops.small.num,
        small_strength=cfg.data.crops.small.strength,
        small_crop_hw=tuple(cfg.data.crops.small.size),
        normalize_mean=tf.constant(cfg.data.normalize.mean),
        normalize_std=tf.constant(cfg.data.normalize.std),
    )
    ds = ds.map(augment_fn, **MAP_KW)
    # Batch: ([B num_large C H W], [B num_small C H W])
    ds = ds.batch(cfg.training.batch_size, drop_remainder=True, **MAP_KW)
    # Turn into a (numpy) iterator so that new samples are returned with every iteration
    ds = ds.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    log.info(
        "Dataset train, %d samples, %d batch size",
        cfg.data.train.max_samples,
        cfg.training.batch_size,
    )
    return ds


def build_dataset_val(cfg: DictConfig) -> tf.data.Dataset:
    """Build validation dataset.

    Args:
        cfg: configuration

    Returns:
        Validation dataset.
    """
    if cfg.data.name == "CLEVR10":
        tfr_path = Path(cfg.data.root) / "clevr_with_masks" / "imgs_val.tfrecords"
        tfr_path = tfr_path.expanduser().resolve().as_posix()
        img_size = osc.data.clevr_with_masks.IMAGE_SIZE
        check_num_samples(
            cfg.data.val.max_samples, osc.data.clevr_with_masks.NUM_SAMPLES_VAL
        )
    else:
        raise NotImplementedError(cfg.data.name)

    ds = tf.data.TFRecordDataset(tfr_path, compression_type="GZIP")
    # Limit number of samples
    ds = ds.take(cfg.data.val.max_samples)
    # Decode image
    ds = ds.map(partial(deserialize_image, img_size=img_size), **MAP_KW)
    # Add seeds for augmentations (seeds repeat identically at every iteration)
    ds = zip_with_seeds(ds, cfg.data.val.seed)
    # Augment large and small crops [H W C] -> ([num_large C H W], [num_small C H W])
    augment_fn = partial(
        augment_train,
        large_boxes=tf.convert_to_tensor(
            generate_random_boxes(
                area_min=cfg.data.crops.large.area[0],
                area_max=cfg.data.crops.large.area[1],
                ratio_min=cfg.data.crops.large.ratio[0],
                ratio_max=cfg.data.crops.large.ratio[1],
                num_boxes=1_000_000,
                seed=cfg.data.val.seed + 1,
            ),
            dtype=tf.float32,
        ),
        large_num_crops=cfg.data.crops.large.num,
        large_strength=cfg.data.crops.large.strength,
        large_crop_hw=tuple(cfg.data.crops.large.size),
        small_boxes=tf.convert_to_tensor(
            generate_random_boxes(
                area_min=cfg.data.crops.small.area[0],
                area_max=cfg.data.crops.small.area[1],
                ratio_min=cfg.data.crops.small.ratio[0],
                ratio_max=cfg.data.crops.small.ratio[1],
                num_boxes=1_000_000,
                seed=cfg.data.val.seed + 2,
            ),
            dtype=tf.float32,
        ),
        small_num_crops=cfg.data.crops.small.num,
        small_strength=cfg.data.crops.small.strength,
        small_crop_hw=tuple(cfg.data.crops.small.size),
        normalize_mean=tf.constant(cfg.data.normalize.mean),
        normalize_std=tf.constant(cfg.data.normalize.std),
    )
    ds = ds.map(augment_fn, **MAP_KW)
    # Batch: ([B num_large C H W], [B num_small C H W])
    ds = ds.batch(cfg.training.batch_size, drop_remainder=False, **MAP_KW)
    # Just prefetch, do not call .as_numpy_iterator()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    log.info(
        "Dataset val, %d samples, %d batch size, %d batches",
        cfg.data.val.max_samples,
        cfg.training.batch_size,
        batches_per_epoch(
            cfg.data.val.max_samples, cfg.training.batch_size, drop_last=False
        ),
    )
    return ds


def get_viz_batch(cfg: DictConfig, ds_val: tf.data.Dataset) -> Tensor:
    """Prepare a batch of images for visualization.

    Args:
        cfg: configuration
        ds_val: validation dataset from which to take the samples

    Returns:
        A batch of images, shape ``[B A C H W]``.
    """
    ds_val = iter(ds_val)
    viz = []
    for large, _ in ds_val:
        viz.append(large)
        if sum([imgs.shape[0] for imgs in viz]) >= cfg.data.val.viz_samples:
            break
    else:
        raise RuntimeError(
            f"Too few validation samples ({sum([imgs.shape[0] for imgs in viz],0)}) "
            f"to fill the requested viz batch ({cfg.data.val.viz_samples})"
        )
    viz = tf.concat(viz, axis=0)[: cfg.data.val.viz_samples]
    viz = torch.from_numpy(np.copy(viz.numpy()))
    log.info("Viz batch, %d samples, shape %s", cfg.data.val.viz_samples, viz.shape)
    return viz


def run_train_val_test(
    cfg: DictConfig,
    model: CoreModel,
    optimizer: torch.optim.Optimizer,
    scheduler: MySequentialLR,
    ds_train: Iterable,
    ds_val: tf.data.Dataset,
    ds_segm: tf.data.Dataset,
    ds_vqa: tf.data.Dataset,
    viz_batch: Tensor,
):
    """Run train, val, viz for a certain number of steps. Handle CTRL+C gracefully.

    Args:
        cfg:
        model:
        optimizer:
        scheduler:
        ds_train:
        ds_val:
        ds_segm:
        ds_vqa:
        viz_batch:
    """
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="The given NumPy array is not writable"
    )

    num_steps = cfg.lr_scheduler.multiplier * (
        cfg.lr_scheduler.warmup.steps
        + cfg.lr_scheduler.decay.steps
        + cfg.lr_scheduler.fixed.steps
    )
    step_interval = math.gcd(
        cfg.training.checkpoint_interval,
        cfg.training.val_interval,
        cfg.training.test_interval,
        cfg.training.viz_interval,
    )
    log.info(
        "Start training for %d steps with intervals every %d steps",
        num_steps,
        step_interval,
    )

    timers = TimerCollection()
    step_counter = StepCounter()
    with SigIntCatcher() as catcher:
        while int(step_counter) < num_steps and catcher.count < 1:
            timers.resume("train")
            run_train_loop(
                cfg,
                ds_train,
                min(step_interval, num_steps - int(step_counter)),
                model,
                optimizer,
                scheduler,
                step_counter,
                catcher,
            )
            timers.pause("train")

            is_last = int(step_counter) >= num_steps or catcher.count > 0
            is_val_time = int(step_counter) % cfg.training.val_interval == 0
            is_test_time = int(step_counter) % cfg.training.test_interval == 0
            is_viz_time = int(step_counter) % cfg.training.viz_interval == 0
            is_ckpt_time = int(step_counter) % cfg.training.checkpoint_interval == 0

            if is_val_time or is_last:
                log.info("Val at %d steps", int(step_counter))
                timers.resume("val")
                run_val_loop(
                    cfg,
                    ds_val,
                    model,
                    int(step_counter),
                    catcher,
                )
                timers.pause("val")

            if is_test_time or is_last:
                log.info("Testing at %d steps", int(step_counter))
                timers.resume("segm")
                run_test_segmentation(cfg, ds_segm, model, int(step_counter), catcher)
                timers.pause("segm")
                timers.resume("vqa")
                run_test_vqa(cfg, ds_vqa, model, int(step_counter))
                timers.pause("vqa")

            if is_viz_time or is_last:
                log.info("Viz at %d steps", int(step_counter))
                timers.resume("viz")
                run_viz(
                    cfg,
                    viz_batch,
                    model,
                    int(step_counter),
                )
                timers.pause("viz")

            if is_ckpt_time or is_last:
                log.info("Checkpoint at %d steps", int(step_counter))
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "steps": int(step_counter),
                    },
                    f"./checkpoint.{int(step_counter):d}.pth",
                )

            log.info(
                "Done %d/%d steps, minutes total: %s",
                int(step_counter),
                num_steps,
                ", ".join(f"{n} {t.seconds() / 60:.1f}" for n, t in timers),
            )


def run_train_loop(
    cfg: DictConfig,
    ds_train: Iterable,
    num_steps: int,
    model: CoreModel,
    optimizer: torch.optim.Optimizer,
    scheduler: MySequentialLR,
    step_counter: StepCounter,
    catcher: SigIntCatcher,
):
    """Run train loop for num_steps.

    Args:
        cfg:
        ds_train:
        num_steps:
        model:
        optimizer:
        scheduler:
        step_counter:
        catcher:
    """
    model.train()
    device = cfg.other.device

    bar = tqdm.tqdm(
        total=num_steps * cfg.training.batch_size,
        desc=f"{int(step_counter):>07d} Train",
        unit="img",
        mininterval=10,  # seconds
        disable=not cfg.other.tqdm,
        ncols=0,
    )
    for large, small in islice(ds_train, num_steps):
        # large: [B 2 C H W]
        # f_backbone: [B 2 H'W' C]
        # f_global, p_global: [B 2 C]
        # f_slots, p_slots: [B 2 S C]
        B, Al = large.shape[:2]
        large = torch.from_numpy(rearrange(large, "B Al C H W -> (B Al) C H W"))
        large_out: ModelOutput = model(large.to(device))
        large_out = large_out.reshape_batch(B, Al)

        small_out: Optional[ModelOutput] = None
        if small.size > 0:
            B, As = small.shape[:2]
            small = torch.from_numpy(rearrange(small, "B As C H W -> (B As) C H W"))
            small_out = model(small.to(device))
            small_out = small_out.reshape_batch(B, As)

        l_glb, logs_glb = compute_losses_global(cfg, large_out)
        l_obj, logs_obj = compute_losses_objects(cfg, large_out, small_out)
        loss = l_glb + l_obj

        logs = logs_glb | logs_obj
        logs["loss/total"] = loss
        logs = {f"{k}/train": v.item() for k, v in logs.items()}
        logs["lr"] = optimizer.param_groups[0]["lr"]
        wandb.log(logs, commit=False, step=int(step_counter))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step_counter.step()
        bar.set_postfix(
            {"loss": logs["loss/total/train"], "steps": int(step_counter)},
            refresh=False,
        )
        bar.update(B)
        if catcher.count >= 1:
            log.info("Training interrupted after %d steps", int(step_counter))
            break

    bar.close()


@torch.no_grad()
def run_val_loop(
    cfg: DictConfig,
    ds_val: tf.data.Dataset,
    model: CoreModel,
    step_counter: int,
    catcher: SigIntCatcher,
):
    """Run validation loop.

    Args:
        cfg:
        ds_val:
        model:
        step_counter:
    """
    model.eval()
    device = cfg.other.device
    logs = defaultdict(AverageMetric)

    bar = tqdm.tqdm(
        total=cfg.data.val.max_samples,
        desc=f"{step_counter:>07d} Val",
        unit="img",
        mininterval=10,  # seconds
        disable=not cfg.other.tqdm,
        ncols=0,
    )
    for large, small in ds_val.as_numpy_iterator():
        # large: [B A C H W]
        # f_backbone: [BA H'W' C]
        # f_global, p_global: [BA C]
        # f_slots, p_slots: [BA K C]
        B, Al = large.shape[:2]
        large = torch.from_numpy(rearrange(large, "B Al C H W -> (B Al) C H W"))
        large_out: ModelOutput = model(large.to(device))
        large_out = large_out.reshape_batch(B, Al)

        small_out: Optional[ModelOutput] = None
        if small.size > 0:
            B, As = small.shape[:2]
            small = torch.from_numpy(rearrange(small, "B As C H W -> (B As) C H W"))
            small_out = model(small.to(device))
            small_out = small_out.reshape_batch(B, As)

        _, logs_glb = compute_losses_global(cfg, large_out)
        _, logs_obj = compute_losses_objects(cfg, large_out, small_out)
        for k, v in chain(logs_glb.items(), logs_obj.items()):
            logs[k].update_scalar(v, B)
        bar.update(B)

        if catcher.count >= 2:
            log.info("Val interrupted, no metrics logged")
            bar.close()
            return
    bar.close()

    logs = {k: v.compute() for k, v in logs.items()}
    logs["loss/total"] = logs["loss/global/total"] + logs["loss/objects/total"]
    wandb.log({f"{k}/val": v for k, v in logs.items()}, commit=False, step=step_counter)


@torch.no_grad()
def run_viz(
    cfg: DictConfig,
    viz_batch: Tensor,
    model: CoreModel,
    step_counter: int,
):
    """Run inference on a single batch of images and visualize everything!

    Args:
        cfg:
        viz_batch: image tensor, float32, ``[B A C H W]``
        model:
        step_counter:
    """

    # region Save input images
    B = viz_batch.shape[0]
    imgs_np = unnormalize_pt(viz_batch, **cfg.data.normalize).numpy()
    array_to_pil(rearrange(imgs_np, "B A C H W -> (B H) (A W) C")).save("viz_batch.png")
    imgs_np = rearrange(imgs_np, "B A C H W -> (B A) H W C")
    # endregion

    # region Prepare dirs and image dict for wandb:
    # - step_123/*.png
    # - step_123/viz_batch/**/*.png
    step_dir = Path(f"step_{step_counter:d}/viz")
    step_dir.mkdir(exist_ok=True, parents=True)
    wandb_imgs = {}
    # endregion

    # region Positional embedding (downscaled)
    if model.backbone.pos_embed is not None:
        fig = viz_positional_embedding(
            model.backbone.get_pos_embed(),
            res_HW=(8, 8),
        )
        fig.savefig(f"step_{step_counter:d}/pos-embed-backbone.png", dpi=100)
        wandb_imgs["pos-embed/backbone"] = wandb.Image(fig)
        plt.close(fig)
    if model.obj_fn is not None:
        if model.obj_fn is not None and model.obj_fn.pos_embed is not None:
            fig = viz_positional_embedding(
                model.obj_fn.pos_embed.reshape(
                    *model.backbone.patch_embed.grid_size, -1
                ),
                res_HW=(8, 8),
            )
            fig.savefig(f"step_{step_counter:d}/pos-embed-obj-fn.png", dpi=100)
            wandb_imgs["pos-embed/obj_fn"] = wandb.Image(fig)
            plt.close(fig)
    # endregion

    # region Forward pass, save output and attns (except backbone because it's too big)
    model.eval()
    with RecordAttentions(model) as all_attns:
        viz_batch = rearrange(viz_batch, "B A C H W -> (B A) C H W")
        out: ModelOutput = model(viz_batch.to(cfg.other.device))
        del viz_batch
    torch.save(
        {
            "output": out._asdict() | {"f_backbone": None},
            "attns": {
                name: [
                    attn_info.as_dict()
                    | {"module": str(attn_info.module)}
                    | ({"attn": None} if name == "backbone" else {})
                    for attn_info in attns
                ]
                for name, attns in all_attns.items()
            },
        },
        step_dir / "viz.pth",
    )
    # endregion

    # region Global matches
    fig = viz_matches_global(
        rearrange(imgs_np, "(B A) H W C -> B A H W C", A=2),
        out.p_global.reshape(B, 2, *out.p_global.shape[1:]),
    )
    fig.savefig(step_dir / "matches-global.png", dpi=200)
    wandb_imgs["matches/global"] = wandb.Image(fig)
    plt.close(fig)
    # endregion

    # region Object matches
    if out.p_slots is not None:
        fig = viz_matches_object(
            rearrange(imgs_np, "(B A) H W C -> B A H W C", A=2),
            out.p_slots.reshape(B, 2, *out.p_slots.shape[1:]),
        )
        fig.savefig(step_dir / "matches-objects.png", dpi=200)
        wandb_imgs["matches/objects"] = wandb.Image(fig)
        plt.close(fig)
    # endregion

    # region Compute rollouts
    # backbone [BA Q K]
    roll_bb = model.backbone.rollout(all_attns["backbone"])
    if "obj_fn" in all_attns:
        # obj_fn [BA S K]
        roll_obj = model.obj_fn.rollout(all_attns["obj_fn"])
        roll_obj_full = torch.bmm(roll_obj, roll_bb)
        # global [BA 1 K] -> [BA K]
        roll_glb = torch.bmm(
            model.global_fn.rollout(all_attns["global_fn"]),
            roll_obj_full,
        ).squeeze(-2)
    else:
        roll_obj = None
        roll_obj_full = None
        # global [BA 1 K] -> [BA K]
        roll_glb = torch.bmm(
            model.global_fn.rollout(all_attns["global_fn"]),
            roll_bb,
        ).squeeze(-2)
    # endregion

    # region Reshape attns and rollouts as images
    K_h, K_w = model.backbone.patch_embed.grid_size

    roll_bb = reduce(roll_bb, "BA Q (K_h K_w) -> BA K_h K_w", "mean", K_h=K_h)
    for attn_info in all_attns["backbone"]:
        attn_info.img = model.backbone.attn_to_img(attn_info)

    if "obj_fn" in all_attns:
        roll_obj = rearrange(roll_obj, "BA S (K_h K_w) -> BA S K_h K_w", K_h=K_h)
        roll_obj_full = rearrange(
            roll_obj_full, "BA S (K_h K_w) -> BA S K_h K_w", K_h=K_h
        )
        for attn_info in all_attns["obj_fn"]:
            attn_info.img = model.obj_fn.attn_to_img(attn_info, (K_h, K_w))
    else:
        for attn_info in all_attns["global_fn"]:
            attn_info.img = model.global_fn.attn_to_img(attn_info, (K_h, K_w))

    roll_glb = rearrange(roll_glb, "BA (K_h K_w) -> BA K_h K_w", K_h=K_h)
    # endregion

    # region Compute K-means backbone
    if cfg.data.name == "CLEVR10":
        n_clusters = 11
        cmap = "tab20"
    else:
        raise NotImplementedError(cfg.data.name)
    bb_kmeans = rearrange(out.f_backbone, "BA H W C -> BA (H W) C")
    bb_kmeans = kmeans_clusters(bb_kmeans, n_clusters)
    bb_kmeans = rearrange(bb_kmeans, "BA (H W) -> BA H W", W=out.f_backbone.shape[-2])
    bb_kmeans = plt.get_cmap(cmap)(bb_kmeans)[..., :3]  # RGB not RGBA
    # endregion

    # region Save individual images
    for ba in range(B * 2):
        ba_str = "img{0}/aug{1}".format(*np.unravel_index(ba, (B, 2)))
        Path.mkdir(step_dir / ba_str, exist_ok=True, parents=True)

        # Input image
        img_input = array_to_pil(imgs_np[ba])
        WH = img_input.size

        # Backbone images
        imgs_bb: List[List[PIL.Image.Image]] = [
            # Input image | kmeans backbone features | rollout backbone
            [
                array_to_pil(imgs_np[ba]),
                array_to_pil(bb_kmeans[ba]),
                array_to_pil(roll_bb[ba], cmap="inferno"),
            ],
            # For each layer: avg head | head 0 | head 1 | ...
            *[
                [array_to_pil(attn_info.img[ba].mean(dim=0), cmap="inferno")]
                + [array_to_pil(head) for head in attn_info.img[ba]]
                for attn_info in all_attns["backbone"]
            ],
        ]
        imgs_bb = [
            [img.resize(WH, resample=PIL.Image.NEAREST) for img in row]
            for row in imgs_bb
        ]
        imgs_bb: PIL.Image.Image = make_grid_pil(imgs_bb)
        imgs_bb.save(step_dir / ba_str / "backbone.png")
        wandb_imgs[f"backbone/{ba_str}"] = wandb.Image(imgs_bb)

        # Object function (if present), one img per iteration (or layer) and per slot
        if "obj_fn" in all_attns:
            imgs_slot: List[List[PIL.Image.Image]] = [
                # Input image | kmeans backbone | backbone rollout | global rollout
                [
                    array_to_pil(imgs_np[ba]),
                    array_to_pil(bb_kmeans[ba]),
                    array_to_pil(roll_bb[ba], cmap="inferno"),
                    array_to_pil(roll_glb[ba], cmap="inferno"),
                ],
                # For each iteration, attention map of each slot (avg heads if present)
                *[
                    [array_to_pil(slot) for slot in attn_info.img[ba]]
                    for attn_info in all_attns["obj_fn"]
                    if attn_info.img is not None
                ],
                # Rollout of slots -> backbone output
                [array_to_pil(slot, cmap="inferno") for slot in roll_obj[ba]],
                # Rollout of slots -> input image
                [array_to_pil(slot, cmap="inferno") for slot in roll_obj_full[ba]],
            ]
            imgs_slot = [
                [img.resize(WH, resample=PIL.Image.NEAREST) for img in row]
                for row in imgs_slot
            ]
            imgs_slot: PIL.Image.Image = make_grid_pil(imgs_slot)
            imgs_slot.save(step_dir / ba_str / "obj.png")
            wandb_imgs[f"obj/{ba_str}"] = wandb.Image(imgs_slot)

        # Summary images
        imgs_summary: List[List[PIL.Image.Image]] = [
            # Input image | kmeans backbone features | backbone rollout | global rollout
            [
                array_to_pil(imgs_np[ba]),
                array_to_pil(bb_kmeans[ba]),
                array_to_pil(roll_bb[ba], cmap="inferno"),
                array_to_pil(roll_glb[ba], cmap="inferno"),
            ],
        ]
        if "obj_fn" in all_attns:
            imgs_summary.extend(
                [
                    # Rollout: slots -> backbone output
                    [array_to_pil(slot, cmap="inferno") for slot in roll_obj[ba]],
                    # Rollout: slots -> backbone output -> input image
                    [array_to_pil(slot, cmap="inferno") for slot in roll_obj_full[ba]],
                ]
            )
        imgs_summary = [
            [img.resize(WH, resample=PIL.Image.NEAREST) for img in row]
            for row in imgs_summary
        ]
        imgs_summary: PIL.Image.Image = make_grid_pil(imgs_summary)
        imgs_summary.save(step_dir / ba_str / "summary.png")
        wandb_imgs[f"summary/{ba_str}"] = wandb.Image(imgs_summary)
    # endregion

    wandb.log(wandb_imgs, step=step_counter, commit=False)
