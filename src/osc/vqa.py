import logging
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn
import tqdm
import wandb
from einops.layers.torch import Reduce
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)
from torch import Tensor

from osc.models.core_model import CoreModel
from osc.utils import batches_per_epoch, check_num_samples

log = logging.getLogger(__name__)


def run_test_vqa(
    cfg: DictConfig,
    ds_vqa: tf.data.Dataset,
    model: CoreModel,
    step_counter: int,
):
    """Train and evaluate linear probes for a VQA using both global and slot features.

    Args:
        cfg:
        ds_vqa:
        model:
        step_counter:

    Returns:

    """
    f_slots, f_global, targets = extract_vqa_features(cfg, ds_vqa, model)
    for feat_name, features in [("obj", f_slots), ("global", f_global)]:
        if features is None:
            continue
        wandb_metrics, wandb_imgs = train_vqa_linear_probe(cfg, features, targets)
        log.info(
            "VQA %s: %s",
            feat_name,
            ", ".join(f"{k} {v:.2f}" for k, v in wandb_metrics.items()),
        )
        wandb.log(
            {
                f"vqa/{feat_name}/{k}": v
                for k, v in (wandb_metrics | wandb_imgs).items()
            },
            commit=False,
            step=step_counter,
        )


@torch.no_grad()
def extract_vqa_features(
    cfg: DictConfig,
    ds_vqa: tf.data.Dataset,
    model: CoreModel,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Extract VQA features, both per-slot and global, and one-hot targets.

    Args:
        cfg:
        ds_vqa:
        model:

    Returns:
        Three tensors, ``(f_slots, f_global, targets)``.
    """
    f_slots = []
    f_global = []
    targets = []
    device = cfg.other.device

    model.eval()
    bar = tqdm.tqdm(
        total=cfg.data.test.vqa.max_samples,
        desc="VQA feats",
        unit="img",
        mininterval=10,  # seconds
        disable=not cfg.other.tqdm,
        ncols=0,
    )
    for examples in ds_vqa.as_numpy_iterator():
        images = torch.from_numpy(examples["image"])
        output = model(images.to(device))
        if output.f_slots is not None:
            f_slots.append(output.f_slots.cpu())
        f_global.append(output.f_global.cpu())
        targets.append(examples["vqa_target"])
        bar.update(images.shape[0])
    bar.close()

    f_slots = torch.cat(f_slots) if len(f_slots) > 0 else None
    f_global = torch.cat(f_global)
    targets = torch.from_numpy(np.concatenate(targets))
    return f_slots, f_global, targets


def train_vqa_linear_probe(
    cfg: DictConfig, features: Tensor, targets: Tensor
) -> Tuple[Dict[str, float], Dict[str, wandb.Image]]:
    wandb_imgs = {}
    wandb_metrics = {}

    device = cfg.other.device
    split = int(cfg.data.test.vqa.split * len(features))
    features_train = features[:split].to(device)
    features_test = features[split:].to(device)
    targets_train = targets[:split].float().to(device)
    targets_test = targets[split:].float().to(device)

    # For each True label there are 14 False labels
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=(
            None
            if cfg.data.test.vqa.pos_weight is None
            else torch.tensor(cfg.data.test.vqa.pos_weight)
        )
    )
    torch.manual_seed(cfg.data.test.vqa.seed)
    probe = build_linear_probe(features.shape[1:], targets.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(probe.parameters())

    # region Train
    history = []
    num_steps = cfg.data.test.vqa.num_steps
    for i in tqdm.trange(
        num_steps,
        desc="VQA probe",
        unit="step",
        ncols=0,
        mininterval=10,  # seconds
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
        history.append((i, loss_train.item(), loss_test.item()))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    df = pd.DataFrame(history, columns=["step", "train", "val"]).set_index("step")
    df.plot(ax=axs[0])
    df[num_steps // 2 :].plot(ax=axs[1])
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Linear probe optimization")
    axs[0].fill_between(
        np.arange(num_steps),
        0,
        1,
        where=np.arange(num_steps) > (num_steps // 2),
        color="gray",
        alpha=0.2,
        transform=axs[0].get_xaxis_transform(),
    )
    fig.set_facecolor("white")
    wandb_imgs["linear_probe_opt"] = wandb.Image(fig)
    plt.close(fig)
    # endregion

    # region Predictions
    with torch.no_grad():
        preds = probe(features_test)

    wandb_metrics["loss"] = loss_fn(preds, targets_test).item()
    targets_test = targets_test.cpu().numpy().astype(bool)
    sample_weight = np.where(targets_test.flatten(), 14.0, 1.0)
    preds = preds.sigmoid_().cpu().numpy()

    for plot_name, plot_fn in [
        ("pr", PrecisionRecallDisplay.from_predictions),
        ("auroc", RocCurveDisplay.from_predictions),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_fn(
            y_true=targets_test.flatten(),
            y_pred=preds.flatten(),
            ax=ax,
            name="Imbalanced",
        )
        plot_fn(
            y_true=targets_test.flatten(),
            y_pred=preds.flatten(),
            sample_weight=sample_weight,
            ax=ax,
            name="Balanced",
        )
        fig.set_facecolor("white")
        wandb_imgs[plot_name] = wandb.Image(fig)
        plt.close(fig)

    for metric_name, metric_fn in [
        ("ap", average_precision_score),
        ("auroc", roc_auc_score),
    ]:
        for balanced in [False, True]:
            metric_name_full = metric_name + ["/imbalanced", "/balanced"][balanced]
            wandb_metrics[metric_name_full] = metric_fn(
                y_true=targets_test.flatten(),
                y_score=preds.flatten(),
                sample_weight=sample_weight if balanced else None,
            )
    # endregion

    return wandb_metrics, wandb_imgs


def build_linear_probe(
    feat_shape: Tuple[int, ...], output_classes: int
) -> torch.nn.Module:
    if len(feat_shape) == 1:
        # Global features [C]
        return torch.nn.Linear(feat_shape[0], output_classes)

    if len(feat_shape) == 2:
        # Slot features [S, C]
        return torch.nn.Sequential(
            torch.nn.Linear(feat_shape[1], output_classes),
            Reduce("B S C -> B C", reduction="max"),
        )

    raise ValueError(f"Invalid feature shape {feat_shape}")


def build_dataset_vqa(cfg: DictConfig) -> tf.data.Dataset:
    """Build VQA dataset for linear probing.

    Args:
        cfg: configuration

    Returns:
        VQA dataset.
    """
    MAP_KW = {"num_parallel_calls": tf.data.AUTOTUNE, "deterministic": True}

    if cfg.data.name == "CLEVR10":
        import osc.data.clevr_with_masks

        tfr_path = Path(cfg.data.root) / "clevr_with_masks" / "imgs_test.tfrecords"
        check_num_samples(
            cfg.data.test.vqa.max_samples, osc.data.clevr_with_masks.NUM_SAMPLES_TEST
        )

        ds = (
            tf.data.TFRecordDataset(
                tfr_path.expanduser().resolve().as_posix(), compression_type="GZIP"
            )
            .take(cfg.data.test.vqa.max_samples)
            .map(osc.data.clevr_with_masks.decode, **MAP_KW)
            .map(osc.data.clevr_with_masks.fix_tf_dtypes, **MAP_KW)
            .map(
                partial(
                    osc.data.clevr_with_masks.prepare_test_vqa,
                    crop_size=tuple(cfg.data.crops.large.size),
                    mean=tuple(cfg.data.normalize.mean),
                    std=tuple(cfg.data.normalize.std),
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
        "Dataset test VQA, %d samples, %d batch size, %d batches",
        cfg.data.test.vqa.max_samples,
        cfg.training.batch_size,
        batches_per_epoch(
            cfg.data.test.vqa.max_samples, cfg.training.batch_size, drop_last=False
        ),
    )
    return ds
