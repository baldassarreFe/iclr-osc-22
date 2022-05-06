import logging
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.optimize
import tensorflow as tf
import torch
import torch.nn.functional
import torch.nn.functional as F
import tqdm
import wandb
from einops import rearrange
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import Tensor

from osc.data.utils import unnormalize_pt
from osc.models.core_model import CoreModel, RecordAttentions
from osc.utils import SigIntCatcher, batches_per_epoch, check_num_samples
from osc.viz.utils import subplots_grid, threshold_otsu_pt

log = logging.getLogger(__name__)


def build_dataset_segmentation(cfg: DictConfig) -> tf.data.Dataset:
    """Build segmentation dataset.

    Args:
        cfg: configuration

    Returns:
        Segmentation dataset.
    """
    MAP_KW = {"num_parallel_calls": tf.data.AUTOTUNE, "deterministic": True}

    if cfg.data.name == "CLEVR10":
        import osc.data.clevr_with_masks

        tfr_path = Path(cfg.data.root) / "clevr_with_masks" / "imgs_test.tfrecords"
        tfr_path = tfr_path.expanduser().resolve().as_posix()
        check_num_samples(
            cfg.data.test.segmentation.max_samples,
            osc.data.clevr_with_masks.NUM_SAMPLES_TEST,
        )

        ds = (
            tf.data.TFRecordDataset(tfr_path, compression_type="GZIP")
            .take(cfg.data.test.segmentation.max_samples)
            .map(osc.data.clevr_with_masks.decode, **MAP_KW)
            .map(osc.data.clevr_with_masks.fix_tf_dtypes, **MAP_KW)
            .map(
                partial(
                    osc.data.clevr_with_masks.prepare_test_segmentation,
                    img_size=tuple(osc.data.clevr_with_masks.IMAGE_SIZE),
                    crop_size=tuple(cfg.data.crops.large.size),
                    mean=tf.constant(cfg.data.normalize.mean),
                    std=tf.constant(cfg.data.normalize.std),
                ),
                **MAP_KW,
            )
        )
    else:
        raise ValueError(cfg.data.name)

    # Batch, prefetch, do not call .as_numpy_iterator()
    ds = ds.batch(cfg.training.batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    log.info(
        "Dataset test segmentation, %d samples, %d batch size, %d batches",
        cfg.data.test.segmentation.max_samples,
        cfg.training.batch_size,
        batches_per_epoch(
            cfg.data.test.segmentation.max_samples,
            cfg.training.batch_size,
            drop_last=False,
        ),
    )
    return ds


@torch.no_grad()
def run_test_segmentation(
    cfg: DictConfig,
    ds_test: tf.data.Dataset,
    model: CoreModel,
    step_counter: int,
    catcher: SigIntCatcher,
):
    # Skip segmentation if the models doesn't contain object attention
    if "obj_fn" not in cfg.model:
        return

    model.eval()
    device = cfg.other.device
    K_h, K_w = model.backbone.patch_embed.grid_size
    HW = model.backbone.patch_embed.img_size

    metrics_dict = {
        "iou/by_img_no_bg": [],
        "iou/by_img_with_bg": [],
        "dice/by_img_no_bg": [],
        "dice/by_img_with_bg": [],
    }
    bar = tqdm.tqdm(
        total=cfg.data.test.segmentation.max_samples,
        desc=f"{step_counter:>07d} Segment",
        unit="img",
        mininterval=10,  # seconds
        disable=not cfg.other.tqdm,
        ncols=0,
    )
    with RecordAttentions(model) as all_attns:
        for batch_idx, examples in ds_test.enumerate().as_numpy_iterator():
            B, T = examples["visibility"].shape
            images = torch.from_numpy(examples["image"])
            masks = torch.from_numpy(examples["mask"]).to(device)
            visibility = examples["visibility"]

            for attn_list in all_attns.values():
                attn_list.clear()
            _ = model(images.to(device))

            # Compute rollout: objects -> image
            # roll_obj: [B (Q_h Q_w) (K_h K_w)]
            # roll_obj: [B S (K_h K_w)] -> [B S K_h K_w]
            roll_bb = model.backbone.rollout(all_attns["backbone"])
            roll_obj = torch.bmm(model.obj_fn.rollout(all_attns["obj_fn"]), roll_bb)
            roll_obj = rearrange(roll_obj, "B S (K_h K_w) -> B S K_h K_w", K_h=K_h)

            # Up-sample rollout, normalize the heatmap of each slot in the [0, 1] range
            # as if it was a segmentation map for the corresponding object.
            # roll_obj: [B S (K_h K_w)] -> [B S K_h K_w] -> [B S H W]
            roll_obj = F.interpolate(roll_obj, HW, mode="bilinear", align_corners=False)
            roll_obj = roll_to_segm(roll_obj)

            # DICE score is computed on the raw segmentation maps
            dice_idx, dice = match_segmentation_masks_dice(masks, roll_obj)

            # IoU score requires thresholding the segmentation maps
            # NOTE: this is not the same as image segmentation, where each pixel gets
            # assigned to only one class. Here, if two segmentation maps overlap, those
            # pixels will belong to both object instances.
            # BUG: torch.histogram fails on CPU, see
            #      https://github.com/pytorch/pytorch/issues/69519
            roll_obj_thres = roll_obj.gt(
                threshold_otsu_pt(roll_obj.cpu()).to(roll_obj.device)[..., None, None]
            )
            iou_idx, iou = match_segmentation_masks_iou(masks, roll_obj_thres)

            metrics_dict["iou/by_img_with_bg"].append(
                np.mean(iou, axis=-1, where=visibility)
            )
            metrics_dict["iou/by_img_no_bg"].append(
                np.mean(iou[:, 1:], axis=-1, where=visibility[:, 1:])
            )
            metrics_dict["dice/by_img_with_bg"].append(
                np.mean(dice, axis=-1, where=visibility)
            )
            metrics_dict["dice/by_img_no_bg"].append(
                np.mean(dice[:, 1:], axis=-1, where=visibility[:, 1:])
            )

            # First batch only, for the first 8 images save a figure with:
            # - input image
            # - attn maps per slot (at most T slots, ordered to match ground-truth)
            # - thresholded attn maps per slot (ordered to match ground-truth)
            # - overlay of img and thresholded attn maps per slot (ordered to match GT)
            # - ground-truth masks and per-slot IoU
            if batch_idx == 0:
                imgs_np = unnormalize_pt(images[:8], **cfg.data.normalize).numpy()
                imgs_np = rearrange(imgs_np, "B C H W -> B H W C")
                viz_segmentation_iou(
                    imgs_np,
                    visibility[:8],
                    masks[:8].cpu().numpy(),
                    roll_obj[:8].cpu().numpy(),
                    roll_obj_thres[:8].cpu().numpy(),
                    iou_idx[:8],
                    iou[:8],
                    step_counter,
                )
                viz_segmentation_dice(
                    imgs_np,
                    visibility[:8],
                    masks[:8].cpu().numpy(),
                    roll_obj[:8].cpu().numpy(),
                    dice_idx[:8],
                    dice[:8],
                    step_counter,
                )

            bar.update(B)
            if catcher.count >= 2:
                log.info("Segm interrupted")
                bar.close()
                return
    bar.close()

    metrics_dict = {
        f"segm/{k}": np.mean(np.concatenate(v)) for k, v in metrics_dict.items()
    }
    wandb.log(metrics_dict, commit=False, step=step_counter)


@torch.jit.script
def roll_to_segm(roll_obj: Tensor) -> Tensor:
    """Translate the ``[H, W]`` attention map of each slot to the range ``[0, 1]``"""
    B, S, H, W = roll_obj.shape
    roll_obj = roll_obj.reshape(B, S, H * W)
    vmin, vmax = roll_obj.aminmax(dim=-1, keepdim=True)
    roll_obj = (roll_obj - vmin) / (vmax - vmin).clamp_min(1.0e-8)
    return roll_obj.reshape(B, S, H, W)


def match_segmentation_masks_iou(
    masks: Tensor, preds: Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Match predicted and true segmentation masks based on IoU.

    Args:
        masks: torch.bool ground-truth masks ``[B T H W]``
        preds: torch.bool prediction masks ``[B S H W]`` for IoU matching

    Returns:
        Tuple of ``(idx, iou)``. The best match for ``masks[b, t]`` is
        ``preds[b, idx[b, t]]`` with an IoU of ``iou[b, t]``.
    """
    B, T = masks.shape[:2]
    masks = rearrange(masks, "B T H W -> B T 1 (H W)")
    preds = rearrange(preds, "B S H W -> B 1 S (H W)")

    # iou_all: [B T S]
    intersection = torch.logical_and(masks, preds).float().sum(-1)
    union = torch.logical_or(masks, preds).float().sum(-1)
    iou_all = intersection.div_(union.clamp_min_(1e-8)).cpu().numpy()

    # idx, iou: [B T]
    iou = np.empty((B, T), dtype=float)
    idx = np.empty((B, T), dtype=int)
    for b in range(B):
        rows, cols = scipy.optimize.linear_sum_assignment(iou_all[b], maximize=True)
        iou[b, :] = iou_all[b, rows, cols]
        idx[b, :] = cols

    return idx, iou


def match_segmentation_masks_dice(
    masks: Tensor, preds: Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Match predicted and true segmentation masks based on DICE.

    Args:
        masks: torch.bool ground-truth masks ``[B T H W]``
        preds: torch.float32 prediction masks ``[B S H W]`` for DICE matching

    Returns:
        Tuple of ``(idx, dice)``. The best match for ``masks[b, t]`` is
        ``preds[b, idx[b, t]]`` with a DICE score of ``dice[b, t]``.
    """
    assert masks.dtype == torch.bool
    B, T = masks.shape[:2]
    masks = rearrange(masks, "B T H W -> B T 1 (H W)").float()
    preds = rearrange(preds, "B S H W -> B 1 S (H W)")

    # dice_all: [B T S]
    intersection = torch.sum(masks * preds, dim=-1)
    union = torch.sum(masks, dim=-1) + torch.sum(preds, dim=-1)
    dice_all = intersection.mul_(2).div_(union.clamp_min_(1e-8)).cpu().numpy()

    # idx, dice: [B T]
    dice = np.empty((B, T))
    idx = np.empty((B, T), dtype=int)
    for b in range(B):
        rows, cols = scipy.optimize.linear_sum_assignment(dice_all[b], maximize=True)
        dice[b, :] = dice_all[b, rows, cols]
        idx[b, :] = cols

    return idx, dice


def viz_segmentation_iou(
    images: np.ndarray,
    visibility: np.ndarray,
    masks: np.ndarray,
    roll_obj: np.ndarray,
    roll_obj_thres: np.ndarray,
    iou_idx: np.ndarray,
    iou_val: np.ndarray,
    step_counter: int,
):
    wandb_imgs = {}
    d = Path(f"step_{step_counter:d}/segm/iou")
    d.mkdir(exist_ok=True, parents=True)
    B, T = visibility.shape
    for b in range(B):
        fig, axs = subplots_grid(5, T, ax_height_inch=1.5, dpi=72)
        val_total = np.mean(iou_val[b, 1:], where=visibility[b, 1:])

        axs[0, 0].imshow(images[b])
        axs[0, 0].set_title(f"Image {b}")
        axs[1, 0].set_ylabel("Attn")
        axs[2, 0].set_ylabel("Thresholded")
        axs[3, 0].set_ylabel("Overlay")
        axs[4, 0].set_ylabel(f"Ground-truth\nIoU no bg {val_total:.2f}")

        # t: index of ground-truth mask, 0, 1, 2, ...
        # s: index of attn mask that best matches with t
        for t in range(T):
            s = iou_idx[b, t]
            axs[1, t].imshow(roll_obj[b, s], interpolation="none")
            axs[2, t].imshow(roll_obj_thres[b, s], interpolation="none")
            axs[3, t].imshow(
                np.where(roll_obj_thres[b, s, :, :, None], images[b], 0.3 * images[b]),
                interpolation="none",
            )
            axs[4, t].imshow(
                masks[b, t], cmap="gray", vmin=0, vmax=1, interpolation="none"
            )
            axs[4, t].set_xlabel(f"Mask {t} slot {s}\nIoU {iou_val[b, t]:.2f}")

        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axs[0, 1:]:
            ax.set_axis_off()

        fig.set_facecolor("white")
        fig.tight_layout(pad=0, h_pad=0, w_pad=0)
        fig.savefig(d / f"img{b}.png")
        wandb_imgs[f"segm/iou/img{b}"] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(wandb_imgs, step=step_counter, commit=False)


def viz_segmentation_dice(
    images: np.ndarray,
    visibility: np.ndarray,
    masks: np.ndarray,
    roll_obj: np.ndarray,
    dice_idx: np.ndarray,
    dice_val: np.ndarray,
    step_counter: int,
):
    wandb_imgs = {}
    d = Path(f"step_{step_counter:d}/segm/dice")
    d.mkdir(exist_ok=True, parents=True)
    B, T = visibility.shape
    for b in range(B):
        fig, axs = subplots_grid(4, T, ax_height_inch=1.5, dpi=72)
        val_total = np.mean(dice_val[b, 1:], where=visibility[b, 1:])

        axs[0, 0].imshow(images[b])
        axs[0, 0].set_title(f"Image {b}")
        axs[1, 0].set_ylabel("Attn")
        axs[2, 0].set_ylabel("Overlay")
        axs[3, 0].set_ylabel(f"Ground-truth\nDICE no bg {val_total:.2f}")

        # t: index of ground-truth mask, 0, 1, 2, ...
        # s: index of attn mask that best matches with t
        for t in range(T):
            s = dice_idx[b, t]
            axs[1, t].imshow(roll_obj[b, s], interpolation="none")
            axs[2, t].imshow(
                roll_obj[b, s, :, :, None] * images[b],
                interpolation="none",
            )
            axs[3, t].imshow(
                masks[b, t], cmap="gray", vmin=0, vmax=1, interpolation="none"
            )
            axs[3, t].set_xlabel(f"Mask {t} slot {s}\nDICE {dice_val[b, t]:.2f}")

        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axs[0, 1:]:
            ax.set_axis_off()

        fig.set_facecolor("white")
        fig.tight_layout(pad=0, h_pad=0, w_pad=0)
        fig.savefig(d / f"img{b}.png")
        wandb_imgs[f"segm/dice/img{b}"] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(wandb_imgs, step=step_counter, commit=False)
