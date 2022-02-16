from typing import Tuple

import numpy as np
import scipy.optimize
import torch
from einops import rearrange
from torch import Tensor


def match_segmentation_masks(
    masks: Tensor, preds: Tensor, preds_thres: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Match predicted and ground-truth segmentation masks.

    Args:
        masks: binary ground-truth masks ``[B T H W]``
        preds: float prediction masks ``[B S H W]`` for DICE matching
        preds_thres: binarized prediction masks ``[B S H W]`` for IoU matching

    Returns:
        Tuple of ``(iou_idx, iou_val, dice_idx, dice_val)``.
        IoU indexes indicate that the best match for ``masks[b, t]`` is
        ``preds_thres[b, iou_idx[b, t]]`` with an IoU of ``iou_val[b, t]`.
        DICE indexes indicate that the best match for ``masks[b, t]`` is
        ``preds[b, dice_idx[b, t]]`` with a DICE score of ``dice_val[b, t]`.
    """
    assert masks.dtype == torch.bool
    B = masks.shape[0]
    device = masks.device
    masks = rearrange(masks, "B T H W -> B T 1 (H W)")
    preds = rearrange(preds, "B S H W -> B 1 S (H W)")
    preds_thres = rearrange(preds_thres, "B S H W -> B 1 S (H W)")

    # iou [B T S]
    intersection = torch.logical_and(masks, preds_thres).float().sum(-1)
    union = torch.logical_or(masks, preds_thres).float().sum(-1)
    iou = intersection.div_(union.clamp_min_(1e-8))
    iou = iou.cpu().numpy()

    # iou_idx, iou_val: [B T]
    iou_idx = []
    iou_val = []
    for b in range(B):
        rows, cols = scipy.optimize.linear_sum_assignment(iou[b], maximize=True)
        iou_val.append(iou[b, rows, cols])
        iou_idx.append(cols[np.argsort(rows)])
    iou_idx = torch.from_numpy(np.stack(iou_idx, axis=0)).to(device)
    iou_val = torch.from_numpy(np.stack(iou_val, axis=0)).to(device)

    # dice: [B T S]
    masks = masks.float()
    intersection = torch.sum(masks * preds, dim=-1)
    union = torch.sum(masks, dim=-1) + torch.sum(preds, dim=-1)
    dice = intersection.mul_(2).div_(union.clamp_min_(1e-8))
    dice = dice.cpu().numpy()

    # dice_idx, dice_val: [B T]
    dice_idx = []
    dice_val = []
    for b in range(masks.shape[0]):
        rows, cols = scipy.optimize.linear_sum_assignment(dice[b], maximize=True)
        dice_val.append(dice[b, rows, cols])
        dice_idx.append(cols[np.argsort(rows)])
    dice_idx = torch.from_numpy(np.stack(dice_idx, axis=0)).to(device)
    dice_val = torch.from_numpy(np.stack(dice_val, axis=0)).to(device)

    return iou_idx, iou_val, dice_idx, dice_val
