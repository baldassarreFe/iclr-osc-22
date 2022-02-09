from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR


class LinearWarmupCosineAnneal(object):
    """Increase LR linearly, then decay with cosine annealing"""

    def __init__(self, optimizer, warmup_steps: int, decay_steps, end_lr):
        self.steps = 0
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.warmup = LinearLR(
            optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
        )
        self.decay = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=end_lr)

    def step(self):
        if self.steps < self.warmup_steps:
            self.warmup.step()
        elif self.steps - self.warmup_steps < self.decay_steps:
            self.decay.step()
        self.steps += 1


def exponential_lr_gamma(reduction_factor, num_steps):
    """Helper to compute ``gamma`` for ExponentialLR

    Args:
        reduction_factor: the desired reduction factor after ``num_steps``,
                          i.e. ``target_lr/start_lr``, recommended 0.05
        num_steps: number of steps in which the reduction factor should be achieved

    Returns:

    """
    return reduction_factor ** (1 / num_steps)
