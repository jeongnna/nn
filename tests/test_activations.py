import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import tensorflow as tf
from .utils import transfer
from nn.activations import Identity, Sigmoid, Relu


def test_identity():
    identity = Identity()

    x = np.array([[-0.4, -0.3, -0.2],
                  [-0.1,  0. ,  0.1],
                  [ 0.2,  0.3,  0.4]])

    y = x.copy()

    assert_array_equal(identity(x), y)
    assert_array_equal(identity.backward(x), y)


def test_sigmoid():
    sigmoid = Sigmoid()

    x = np.array([[-0.4, -0.3, -0.2],
                  [-0.1,  0. ,  0.1],
                  [ 0.2,  0.3,  0.4]])

    y, dydx = transfer(x, tf.keras.activations.sigmoid)

    assert_array_almost_equal(sigmoid(x), y)

    dout = np.array([[ 0. ,  0.1, -0.2],
                     [ 0.3, -0.4,  0.5],
                     [-0.6,  0.7, -0.8]])

    assert_array_almost_equal(sigmoid.backward(dout), dout * dydx)


def test_relu():
    relu = Relu()

    x = np.array([[-0.4, -0.3, -0.2],
                  [-0.1,  0. ,  0.1],
                  [ 0.2,  0.3,  0.4]])

    y, dydx = transfer(x, tf.keras.activations.relu)

    assert_array_equal(relu(x), y)

    dout = np.array([[ 0. ,  0.1, -0.2],
                     [ 0.3, -0.4,  0.5],
                     [-0.6,  0.7, -0.8]])

    assert_array_equal(relu.backward(dout), dout * dydx)
