from bisect import bisect_right

import torch.optim
from omegaconf import DictConfig
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    MultiplicativeLR,
    SequentialLR,
)


class MySequentialLR(SequentialLR):
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
        super().__init__(optimizer, schedulers, milestones, last_epoch, verbose)

    """Just like ``SequentialLR`` but without warnings."""

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()

    def __str__(self):
        details = ", ".join(
            f"{m}: {s}" for m, s in zip([0] + self._milestones, self._schedulers)
        )
        return f"{self.__class__.__name__}({details})"


def exponential_lr_gamma(reduction_factor: float, num_steps: int) -> float:
    """Helper to compute ``gamma`` for ExponentialLR

    Args:
        reduction_factor: the desired reduction factor after ``num_steps``,
                          i.e. ``target_lr/start_lr``, for example 0.05
        num_steps: number of steps in which the reduction factor should be achieved

    Returns:
        The ``gamma`` factor for ``ExponentialLR``.
    """
    return reduction_factor ** (1 / num_steps)


def build_scheduler(
    cfg: DictConfig, optimizer: torch.optim.Optimizer
) -> MySequentialLR:
    """Build learning rate scheduler.

    The scheduler has three phases: linear warmup, exponential/cosine decay,
    and then a fixed rate forever.

    Args:
        cfg: configuration
        optimizer: optimizer

    Returns:
        A learning rate scheduler.
    """
    schedulers = []
    milestones = []

    # Linear warmup
    warmup_steps = cfg.lr_scheduler.multiplier * cfg.lr_scheduler.warmup.steps
    milestones.append(warmup_steps)
    schedulers.append(
        LinearLR(
            optimizer,
            start_factor=cfg.lr_scheduler.warmup.start_factor,
            end_factor=cfg.lr_scheduler.warmup.end_factor,
            total_iters=warmup_steps,
        )
    )

    # Decay
    decay_steps = cfg.lr_scheduler.multiplier * cfg.lr_scheduler.decay.steps
    milestones.append(warmup_steps + decay_steps)
    if cfg.lr_scheduler.decay.name == "cosine":
        schedulers.append(
            CosineAnnealingLR(
                optimizer,
                T_max=decay_steps,
                eta_min=cfg.lr_scheduler.decay.end_factor * cfg.optimizer.start_lr,
            )
        )
    elif cfg.lr_scheduler.decay.name == "exponential":
        schedulers.append(
            ExponentialLR(
                optimizer,
                gamma=exponential_lr_gamma(
                    cfg.lr_scheduler.decay.end_factor, decay_steps
                ),
            )
        )
    else:
        raise ValueError(cfg.lr_scheduler.decay.name)

    # Fixed value
    schedulers.append(MultiplicativeLR(optimizer, lambda _: 1.0))

    return MySequentialLR(optimizer, schedulers, milestones)
