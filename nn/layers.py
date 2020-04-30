from abc import ABC, abstractmethod
import numpy as np
from .safeutils import safelyInputDimension
from .activations import activations, Activation


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
        inputDim = safelyInputDimension(inputDim)
        outputDim = safelyInputDimension(outputDim)

        self.w = np.zeros((*inputDim, *outputDim))
        self.b = np.zeros((1, *outputDim))

    def forward(self, x, *args, **kwargs):
        self.x = x

        return x.dot(self.w) + self.b

    def backward(self, dout, *args, **kwargs):
        needUpdate = kwargs.get('needUpdate', True)

        print('dout: {}'.format(dout.shape))
        print('w: {}'.format(self.w.shape))

        assert dout.shape[-1]
        dx = dout.dot(self.w.T)

        if needUpdate:
            dw = self.x.T.dot(dout)
            db = np.sum(dout, axis=0).reshape(self.b.shape)
            self.update(dw, db)

        return dx

    def update(self, dw, db):
        assert dw.shape == self.w.shape
        assert db.shape == self.b.shape
        self.w -= dw
        self.b -= db


class Dense(Layer):

    def __init__(self, inputDim, outputDim, activation=None, *args, **kwargs):
        if activation is None:
            self.activation = activations['identity']()

        elif isinstance(activation, str):
            self.activation = activations[activation]()

        elif isinstance(activation, Activation):
            self.activation = activation

        else:
            raise TypeError(f'Argument `activation` must be a string or an instance of {Activation}. Current value is {activation}.')

        self.affine = Affine(inputDim, outputDim)

    def forward(self, x, *args, **kwargs):
        affined = self.affine(x)

        return self.activation(affined)

    def backward(self, dout, *args, **kwargs):
        dact = self.activation.backward(dout, *args, **kwargs)
        dx = self.affine.backward(dact, *args, **kwargs)

        return dx
