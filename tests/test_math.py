import numpy as np
from numpy.testing import assert_array_almost_equal
import tensorflow as tf
from .utils import transfer
from nn.math import sigmoid, softmax


def test_sigmoid():
    x = np.array([-100., -10., -0.2, -0.1,  0.,  0.1, 0.2, 10., 100.])

    y, _ = transfer(x, tf.math.sigmoid)

    assert_array_almost_equal(sigmoid(x), y)


def test_softmax():
    x = np.array([[-100., -10., -1.  ],
                  [-0.1 ,  0. ,  0.1 ],
                  [ 1.  ,  10.,  100.]])

    y, _ = transfer(x, tf.keras.activations.softmax)

    assert_array_almost_equal(softmax(x), y)
