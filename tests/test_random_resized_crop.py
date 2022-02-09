import numpy as np
import tensorflow as tf

from src.osc.data.random_resized_crop import random_resized_crop


def test_random_resized_crop():
    """Just test output shape, not actual cropping"""
    rng = tf.random.Generator.from_seed(0)
    seed = rng.make_seeds()[:, 0]

    size = (64, 64)

    img = np.random.rand(100, 120, 3)
    img2 = random_resized_crop(img, size=size, seed=seed)
    assert img2.shape == (*size, 3)

    img = np.random.rand(1000, 10, 3)
    img2 = random_resized_crop(img, size=size, seed=seed)
    assert img2.shape == (*size, 3)

    img = np.random.rand(10, 1000, 3)
    img2 = random_resized_crop(img, size=size, seed=seed)
    assert img2.shape == (*size, 3)
