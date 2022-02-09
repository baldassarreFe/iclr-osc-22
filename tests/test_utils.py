import torch

from osc.utils import l2_normalize


def test_l2_normalize():
    """Just test shape and backward, not numerical result"""
    x = torch.rand(10, 16).requires_grad_()
    y = x + 1

    y_norm = l2_normalize(torch.clone(y))
    assert x.shape == y_norm.shape

    z = x + y_norm
    z.sum().backward()
    assert x.grad is not None
