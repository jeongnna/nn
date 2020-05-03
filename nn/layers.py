from abc import ABC, abstractmethod
import numpy as np
from .safeutils import _check_layer_dim, _check_activation


class Layer(ABC):

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class Flatten(Layer):

    def forward(self, x, *args, **kwargs):
        self.input_shape = x.shape

        return x.reshape(x.shape[0], -1)

    def backward(self, dout, *args, **kwargs):
        return dout.reshape(self.input_shape)


class Affine(Layer):

    def __init__(self, input_dim, output_dim):
        input_dim = _check_layer_dim(input_dim)
        output_dim = _check_layer_dim(output_dim)

        self.weights = np.zeros((*input_dim, *output_dim))
        self.bias = np.zeros((1, *output_dim))

    def forward(self, x, *args, **kwargs):
        self._x = x

        return x.dot(self.weights) + self.bias

    def backward(self, dout, *args, **kwargs):
        needUpdate = kwargs.get('needUpdate', True)

        assert dout.shape[-1]
        dx = dout.dot(self.weights.T)

        if needUpdate:
            dw = self._x.T.dot(dout)
            db = np.sum(dout, axis=0).reshape(self.bias.shape)
            self._update(dw, db)

        return dx

    def _update(self, dw, db):
        assert dw.shape == self.weights.shape and db.shape == self.bias.shape
        self.weights -= dw
        self.bias -= db


class Dense(Layer):

    def __init__(self, input_dim, output_dim, activation=None):
        self.affine = Affine(input_dim, output_dim)
        self.activation = _check_activation(activation)

    def forward(self, x, *args, **kwargs):
        affined = self.affine(x)

        return self.activation(affined)

    def backward(self, dout, *args, **kwargs):
        dact = self.activation.backward(dout, *args, **kwargs)
        dx = self.affine.backward(dact, *args, **kwargs)

        return dx

    @property
    def weight(self):
        return self.affine.weight

    @weight.setter
    def weight(self, value):
        self.affine.weight = value

    @property
    def bias(self):
        return self.affine.bias

    @bias.setter
    def bias(self, value):
        self.affine.bias = value
