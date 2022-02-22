import numpy as np
import numpy.testing as npt
import tensorflow as tf
import torch

from osc.utils import l2_normalize, ravel_multi_index_tf


def test_l2_normalize():
    """Just test shape and backward, not numerical result"""
    x = torch.rand(10, 16).requires_grad_()
    y = x + 1

    y_norm = l2_normalize(torch.clone(y))
    assert x.shape == y_norm.shape

    z = x + y_norm
    z.sum().backward()
    assert x.grad is not None


def test_ravel_multi_index_tf():
    """Test against numpy version"""
    multi_index = (
        [4, 3, 3, 2],
        [0, 1, 5, 6],
        [0, 1, 2, 0],
    )
    dims = (5, 7, 3)

    result_np = np.ravel_multi_index(multi_index, dims=dims)
    result_tf = ravel_multi_index_tf(tuple(map(tf.constant, multi_index)), dims)
    npt.assert_equal(result_tf.numpy(), result_np)
