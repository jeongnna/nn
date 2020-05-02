import numpy as np
from numpy.testing import assert_array_almost_equal
import tensorflow as tf
from .utils import transfer, eval_numerical_gradient_array
from nn.layers import Dense


def test_dense():
    weight = np.array([[-0.3, -0.2],
                       [-0.1,  0. ],
                       [ 0.1,  0.2]])

    bias = np.array([1., 2.])

    dense = Dense(*weight.shape)
    dense.weight = weight
    dense.bias = bias

    tf_dense = tf.keras.layers.Dense(weight.shape[1],
                                     kernel_initializer=tf.constant_initializer(weight),
                                     bias_initializer=tf.constant_initializer(bias))
    tf_dense.build(input_shape=weight.shape[0])

    x = np.array([[0. , 0.1, 0.2],
                  [0.3, 0.4, 0.5],
                  [0.6, 0.7, 0.8],
                  [0.9, 1. , 1.1],
                  [1.2, 1.3, 1.4]])

    y = tf_dense(tf.constant(x)).numpy()

    assert_array_almost_equal(dense(x), y)

    dout = np.array([[0. , 0.1],
                     [0.2, 0.3],
                     [0.4, 0.5],
                     [0.6, 0.7],
                     [0.8, 0.9]])

    dx = eval_numerical_gradient_array(lambda x: dense(x), x, dout)

    assert_array_almost_equal(dense.backward(dout), dx)
