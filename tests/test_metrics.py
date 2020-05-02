import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import tensorflow as tf
from .utils import transfer
from nn.metrics import crossEntropy


def test_crossentropy():
    y = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    probs = np.array([[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]])

    ce = crossEntropy(probs, y.argmax(axis=1))
    cce = tf.keras.losses.CategoricalCrossentropy()(y, probs)

    assert_almost_equal(ce, cce.numpy())
