"""
Collection of utilities.
"""

import inspect
import logging
import random
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tabulate
import tensorflow as tf
import torch

log = logging.getLogger(__name__)

ImgSizeHW = Tuple[int, int]
ImgMean = Tuple[float, float, float]
ImgStd = Tuple[float, float, float]


def print_arrays(
    x: Union[
        Union[str, np.ndarray, torch.Tensor],
        Sequence[Union[str, np.ndarray, torch.Tensor]],
        Mapping[str, Union[np.ndarray, torch.Tensor]],
    ]
):
    """Print array/tensor info (name, shape, dtype).

    The argument can be one of the following:

    - a single array/tensor (no name),
    - a list of variable names from the calling context,
    - or a list of arrays/tensors (no names),
    - or a mapping from names to arrays/tensors.

    Args:
        x: what to print
    """
    orig_type = type(x)

    if isinstance(x, (str, np.ndarray, torch.Tensor)):
        x = [x]

    if isinstance(x, Sequence):
        if len(x) > 0:
            if isinstance(x[0], (np.ndarray, torch.Tensor)):
                x = dict(enumerate(x))
            elif isinstance(x[0], str):
                locs = inspect.currentframe().f_back.f_locals
                x = {k: locs[k] for k in x}
            else:
                raise ValueError(f"Invalid type: {orig_type}")
        else:
            x = dict()

    if isinstance(x, Mapping):
        print(
            tabulate.tabulate(
                [[k, v.dtype, list(v.shape)] for k, v in x.items()],
                headers=["name", "dtype", "shape"],
            )
        )
    else:
        raise ValueError(f"Invalid type: {orig_type}")


@torch.jit.script
def l2_normalize_(a: torch.Tensor) -> torch.Tensor:
    """L2 normalization in-place along the last dimension.

    Args:
        a: [N, C] tensor to normalize.

    Returns:
        The input tensor with normalized rows.
    """
    norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    return a.div_(norm.clamp_min_(1e-10))


@torch.jit.script
def l2_normalize(a: torch.Tensor) -> torch.Tensor:
    """L2 normalization along the last dimension.

    Args:
        a: [..., C] tensor to normalize.

    Returns:
        A new tensor containing normalized rows.
    """
    norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    return a / norm.clamp_min(1e-10)


@torch.jit.script
def cos_pairwise(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pairwise cosine between the rows of two matrices.

    Args:
        a: [N, C] tensor.
        b: [M, C] tensor, defaults to ``a`` if missing.

    Returns:
        [N, M] tensor of cosine values.
    """
    a = l2_normalize(a)
    if b is not None:
        b = l2_normalize(b)
    else:
        b = a
    return torch.einsum("ic,jc->ij", a, b)


def batches_per_epoch(num_samples: int, batch_size: int, drop_last: bool) -> int:
    """Compute the number of batches in one epoch according to batch size and drop behavior.

    Args:
        num_samples:
        batch_size:
        drop_last:

    Returns:
        The number of batches.
    """
    if drop_last:
        return int(np.floor(num_samples / batch_size))
    else:
        return int(np.ceil(num_samples / batch_size))


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed // 2 ** 32)
    log.info("All random seeds: %d", seed)


def latest_checkpoint(run_dir: Union[Path, str]) -> Path:
    """Find the checkpoint with the highest numerical epoch.

    Args:
        run_dir: the search path containing ``checkpoint.*.pth`` files.

    Returns:
        Path to the checkpoint.
    """
    glob = Path(run_dir).glob("checkpoint.*.pth")
    glob = sorted(glob, key=lambda p: int(p.name.split(".")[1]))
    return glob[-1]


@torch.jit.script
def normalize_sum_to_one(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return tensor / tensor.sum(dim=dim, keepdim=True).clamp_min(1e-8)
