from abc import ABC, abstractmethod
import numpy as np
from .safeutils import _checkLayerDimension, _checkActivation


class Layer(ABC):

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class Affine(Layer):

    def __init__(self, inputDim, outputDim):
        inputDim = _checkLayerDimension(inputDim)
        outputDim = _checkLayerDimension(outputDim)

        self.weight = np.zeros((*inputDim, *outputDim))
        self.bias = np.zeros((1, *outputDim))

    def forward(self, x, *args, **kwargs):
        self._x = x

        return x.dot(self.weight) + self.bias

    def backward(self, dout, *args, **kwargs):
        needUpdate = kwargs.get('needUpdate', True)

        assert dout.shape[-1]
        dx = dout.dot(self.weight.T)

        if needUpdate:
            dw = self._x.T.dot(dout)
            db = np.sum(dout, axis=0).reshape(self.bias.shape)
            self._update(dw, db)

        return dx

    def _update(self, dw, db):
        assert dw.shape == self.weight.shape and db.shape == self.bias.shape
        self.weight -= dw
        self.bias -= db


class Dense(Layer):

    def __init__(self, inputDim, outputDim, activation=None):
        self.affine = Affine(inputDim, outputDim)
        self.activation = _checkActivation(activation)

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
