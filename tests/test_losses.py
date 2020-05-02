import numpy as np
from numpy.testing import assert_array_almost_equal
import tensorflow as tf
from .utils import transfer
from nn.losses import SoftmaxCrossEntropy, NegativeLogLikelihood


def test_softmax_crossentropy():
    sceloss = SoftmaxCrossEntropy()

    x = np.array([[-0.4, -0.3, -0.2],
                  [-0.1,  0. ,  0.1],
                  [ 0.2,  0.3,  0.4]])

    y_2d = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    y_1d = y_2d.argmax(axis=1) # [0, 1, 2]

    def tf_sce(x_tensor):
        probs_tensor = tf.keras.activations.softmax(x_tensor)
        return tf.keras.losses.CategoricalCrossentropy()(tf.constant(y_2d), probs_tensor)

    losses, dldx = transfer(x, tf_sce)

    assert_array_almost_equal(sceloss(x, y_1d), losses)
    assert_array_almost_equal(sceloss.backward(x, y_1d), dldx)


def test_negative_loglikelihood():
    pass
