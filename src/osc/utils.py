"""
Collection of utilities.
"""

import inspect
import logging
import random
import signal
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tabulate
import tensorflow as tf
import torch
from fvcore.common.timer import Timer
from torch import Tensor

log = logging.getLogger(__name__)

ImgSizeHW = Tuple[int, int]
ImgMean = Tuple[float, float, float]
ImgStd = Tuple[float, float, float]


def print_arrays(
    x: Union[
        Union[str, np.ndarray, Tensor],
        Sequence[Union[str, np.ndarray, Tensor]],
        Mapping[str, Union[np.ndarray, Tensor]],
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

    if isinstance(x, (str, np.ndarray, Tensor)):
        x = [x]

    if isinstance(x, Sequence):
        if len(x) > 0:
            if isinstance(x[0], (np.ndarray, Tensor)):
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
def l2_normalize_(a: Tensor) -> Tensor:
    """L2 normalization in-place along the last dimension.

    Args:
        a: [N, C] tensor to normalize.

    Returns:
        The input tensor with normalized rows.
    """
    norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    return a.div_(norm.clamp_min_(1e-10))


@torch.jit.script
def l2_normalize(a: Tensor) -> Tensor:
    """L2 normalization along the last dimension.

    Args:
        a: [..., C] tensor to normalize.

    Returns:
        A new tensor containing normalized rows.
    """
    norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    return a / norm.clamp_min(1e-10)


@torch.jit.script
def cos_pairwise(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
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
    """Seed python, torch, tensorflow, and numpy"""
    random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(np.cast[np.uint32](seed))
    log.info("All random seeds: %d", seed)


def latest_checkpoint(run_dir: Union[Path, str]) -> Path:
    """Find the checkpoint with the highest numerical epoch/step identifier.

    Args:
        run_dir: the search path containing ``checkpoint.*.pth`` files.

    Returns:
        Path to the checkpoint.
    """
    glob = Path(run_dir).glob("checkpoint.*.pth")
    glob = sorted(glob, key=lambda p: int(p.name.split(".")[1]))
    return glob[-1]


@torch.jit.script
def normalize_sum_to_one(tensor: Tensor, dim: int = -1) -> Tensor:
    """Normalize tensor so that it sums to one over one dimension (e.g. the last)."""
    return tensor / tensor.sum(dim=dim, keepdim=True).clamp_min(1e-8)


@torch.jit.script
def normalize_zero_one(tensor: Tensor, dim: int = -1) -> Tensor:
    """Normalize tensor so that its values are in the ``[0, 1]`` range."""
    min_max = torch.aminmax(tensor, dim=dim, keepdim=True)
    return (tensor - min_max[0]) / (min_max[1] - min_max[0]).clamp_min(1e-8)


def fill_diagonal_(x: Tensor, value: float = -torch.inf) -> Tensor:
    """Fill diagonal of torch tensor, batched and in-place.

    Args:
        x: ``[..., N, M]`` tensor with leading batch dimensions.
        value: value to fill the diagonal with.

    Returns:
        The same ``[..., N, M]`` input tensor with diagonal entries
        ``x[..., i, i]`` replaced.
    """
    N, M = x.shape[-2:]
    mask = torch.eye(N, M, dtype=torch.bool, device=x.device)
    return x.masked_fill_(mask, value)


def fill_diagonal(x: Tensor, value: float = -torch.inf) -> Tensor:
    """Fill diagonal of torch tensor, batched and out-of-place.

    Args:
        x: ``[..., N, M]`` tensor with leading batch dimensions.
        value: value to fill the diagonal with.

    Returns:
        A new ``[..., N, M]`` tensor with diagonal entries
        ``x[..., i, i]`` replaced.
    """
    N, M = x.shape[-2:]
    mask = torch.eye(N, M, dtype=torch.bool, device=x.device)
    return x.masked_fill(mask, value)


@tf.function
def ravel_multi_index_tf(
    multi_index: Tuple[tf.Tensor, ...], dims: Tuple[int, ...]
) -> tf.Tensor:
    """Same as :func:``np.ravel_multi_index`` with no checks.

    Args:
        multi_index: a tuple of index tensors, one per dimension
        dims: the shape of the tensor. The number of dimensions should be the same
            as the number of index tensors in ``multi_index``.
            Also, ``all(multi_index[i] < dims[i])`` is expected to hold for all ``i``.

    Example:
        How to use:

        >>> ravel_multi_index_tf(
        >>>   (
        >>>     tf.constant([4, 3, 3, 2]),
        >>>     tf.constant([0, 1, 5, 6]),
        >>>     tf.constant([0, 1, 2, 0]),
        >>>   ),
        >>>   dims=(5, 7, 3)
        >>> )
        [84, 67, 80, 60]

    Returns:
        A 1D tensor of flat indexes.
    """
    strides = tf.math.cumprod(tf.constant([*dims, 1]), reverse=True)[1:]
    multi_index = tf.stack(multi_index, axis=0)
    result = multi_index * strides[:, None]
    return tf.reduce_sum(result, axis=0)


def to_dotlist(d, prf: Tuple[str, ...] = tuple()) -> Iterator[Tuple[str, Any]]:
    """Iterate over a flattened dict using dot-separated keys.

    Args:
        d: a hierarchical dictionary with strings as keys
        prf: tuple of prefixes used for recursion

    Notes:
        The type of ``d`` can be defined as ``StringDict``, but it clashes with Sphynx:
        ``StringDict = Mapping[str, Union[Any, StringDict]]``.

    Examples:
        Flatten a dict:

        >>> print(*to_dotlist({'a': {'b': {}, 'c': 1, 'd': [2]}, 'x': []}), sep="\\n")
        ('a.b', {})
        ('a.c', 1)
        ('a.d', [2])
        ('x', [])

    Yields:
        Tuples of dot-separated keys and values. Call :func:``dict`` on the returned
        iterator to build a flat dict.
    """
    for k, v in d.items():
        if isinstance(v, dict) and len(v) > 0:
            yield from to_dotlist(v, prf=(*prf, k))
        else:
            yield ".".join((*prf, k)), v


class SigIntCatcher(AbstractContextManager):
    """Context manager to gracefully handle SIGINT or KeyboardInterrupt.

    Example:
        Gracefully terminate a loop:

        >>> with SigIntCatcher() as catcher:
        >>>     for i in range(1000):
        >>>         print(f"Step {i}...")
        >>>         time.sleep(5)
        >>>         print("Done")
        >>>         if catcher.count > 0:
        >>>             break
    """

    def __init__(self, raise_after=5):
        self._count = None
        self._old_handler = None
        self._raise_after = raise_after

    @property
    def count(self) -> int:
        if self._count is None:
            raise RuntimeError("SigIntCatcher context manager is not active")
        return self._count

    def __enter__(self):
        self._old_handler = signal.getsignal(signal.SIGINT)
        self._count = 0
        signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._old_handler)
        self._count = None
        self._old_handler = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(count={self._count}, raise_after={self._raise_after})"
        )

    # noinspection PyUnusedLocal
    def _handler(self, signum, frame):
        if self._count is None:
            raise RuntimeError("SigIntCatcher context manager is not active")
        self._count += 1
        if self._count >= self._raise_after:
            log.warning("Interrupted again, raising exception")
            raise KeyboardInterrupt()
        else:
            log.warning(
                "Interrupted %d times, wait for graceful termination, "
                "repeat %d more times to raise exception",
                self._count,
                self._raise_after - self._count,
            )


class StepCounter(object):
    """A simple counter to keep track of update steps."""

    def __init__(self):
        self.steps = 0

    def __int__(self):
        return self.steps

    def step(self):
        """Increment by 1."""
        self.steps += 1

    def __str__(self):
        return f"{self.__class__.__name__}({int(self)})"


class AverageMetric(object):
    """Keep track of the average value of a metric across batches"""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update_tensor(self, tensor: Tensor):
        """Update with the values from a tensor.

        Args:
            tensor: per-sample metric values, it will be flattened.
        """
        self.total += tensor.detach().sum().item()
        self.count += tensor.numel()

    def update_scalar(self, scalar: Tensor, num: int):
        """Update with a single scalar that represents the average of ``num`` samples.

        Args:
            scalar: a scalar value representing the average of ``num`` samples
            num: how many samples ``scalar`` stands for
        """
        self.total += num * scalar.item()
        self.count += num

    def compute(self) -> float:
        """Compute.

        Returns:
            The current average.
        """
        return self.total / self.count


def normalized_meshgrid(height: int, width: int) -> Tensor:
    height = torch.linspace(-1, 1, steps=height)
    width = torch.linspace(-1, 1, steps=width)
    grid = torch.meshgrid(height, width, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    return grid


def check_num_samples(max_samples, num_samples: int):
    if max_samples > num_samples:
        raise ValueError(
            f"Requested {max_samples} max samples, "
            f"but the dataset only contains {num_samples} samples",
        )


class TimerCollection(object):
    """Collection of :class:`Timer` objects that are instantiated lazily."""

    def __init__(self):
        self._timers: Dict[str, Timer] = {}

    def resume(self, name: str):
        """Resume an existing timer. If it doesn't exist yet, a new timer is created"""
        # Workaround: timers start immediately upon creation and trying to resume
        # a started timer would produce an error.
        if name not in self._timers:
            self._timers[name] = Timer()
        else:
            self[name].resume()

    def pause(self, name: str):
        self[name].pause()

    def __getitem__(self, name: str) -> Timer:
        return self._timers[name]

    def __iter__(self) -> Iterator[Tuple[str, Timer]]:
        yield from self._timers.items()
