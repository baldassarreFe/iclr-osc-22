"""
Collection of utilities.
"""

import inspect
import logging
import random
import signal
from contextlib import AbstractContextManager
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
    """Cosine between all pairs of entries in two tensors.

    Args:
        a: [*N, C] tensor, where ``*N`` can be any number of leading dimensions.
        b: [*M, C] tensor, where ``*M`` can be any number of leading dimensions.
            Defaults to ``a`` if missing.

    Returns:
        [*N, *M] tensor of cosine values.
    """
    a = l2_normalize(a)
    b = a if b is None else l2_normalize(b)
    N = a.shape[:-1]
    M = b.shape[:-1]
    a = a.flatten(end_dim=-2)
    b = b.flatten(end_dim=-2)
    cos = torch.einsum("nc,mc->nm", a, b)
    return cos.reshape(N + M)


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
    np.random.seed(np.cast[np.uint32](seed))
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


class SigIntCatcher(AbstractContextManager):
    """Context manager to gracefully handle SIGINT or KeyboardInterrupt.

    Example:
        Gracefully terminate a loop:

        >>> with SigIntCatcher() as should_stop:
        >>> for i in range(1000):
        >>>     print(f"Step {i}...")
        >>>     time.sleep(5)
        >>>     print("Done")
        >>>     if should_stop:
        >>>         break
    """

    def __init__(self):
        self._interrupted = None
        self._old_handler = None

    def __bool__(self):
        return self._interrupted

    def __enter__(self):
        self._old_handler = signal.getsignal(signal.SIGINT)
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._old_handler)
        self._interrupted = None
        self._old_handler = None

    # noinspection PyUnusedLocal
    def _handler(self, signum, frame):
        if not self._interrupted:
            log.warning(
                "Interrupted, wait for graceful termination, repeat to raise exception"
            )
            self._interrupted = True
        else:
            log.warning("Interrupted again, raising exception")
            raise KeyboardInterrupt()
