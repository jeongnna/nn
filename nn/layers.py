from abc import ABC, abstractmethod
import numpy as np
from .safeutils import _check_layer_dim, _check_activation


class Layer(ABC):

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self._update_state = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def trainable_variables(self):
        # This method (actually property) should return dictionary which has information about trainable variable of the layer.
        # For example:
        # return {'weight': self.weights, 'bias': self.bias}
        pass

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abstractmethod
    def backward(self, dout, **kwargs):
        pass

    def _update(self, grads, optimizer):
        assert self.trainable_variables.keys() == grads.keys()

        for var, grad in zip(self.trainable_variables.values(), grads.values()):
            self._update_state = optimizer.apply_gradients(var, grad, **self._update_state)


class Flatten(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):
        self.input_shape = x.shape

        return x.reshape(x.shape[0], -1)

    def backward(self, dout, **kwargs):
        return dout.reshape(self.input_shape)


class Affine(Layer):

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)

        input_dim = _check_layer_dim(input_dim)
        output_dim = _check_layer_dim(output_dim)

        self.weights = np.zeros((*input_dim, *output_dim))
        self.bias = np.zeros((1, *output_dim))

    @property
    def trainable_variables(self):
        return {'weight': self.weights, 'bias': self.bias}

    def forward(self, x, **kwargs):
        self._x = x

        return x.dot(self.weights) + self.bias

    def backward(self, dout, **kwargs):
        assert dout.shape[-1]

        dw = self._x.T.dot(dout)
        db = np.sum(dout, axis=0).reshape(self.bias.shape)
        grads = {'weight': dw, 'bias': db}

        optimizer = kwargs.get('optimizer')
        if optimizer:
            self._update(grads, optimizer)

        dx = dout.dot(self.weights.T)

        return dx


class Dense(Layer):

    def __init__(self, input_dim, output_dim, activation=None, **kwargs):
        super().__init__(**kwargs)

        self.affine = Affine(input_dim, output_dim)
        self.activation = _check_activation(activation)

    @property
    def weights(self):
        return self.affine.weights

    @weights.setter
    def weights(self, value):
        self.affine.weights = value

    @property
    def bias(self):
        return self.affine.bias

    @bias.setter
    def bias(self, value):
        self.affine.bias = value

    def forward(self, x, **kwargs):
        affined = self.affine(x)

        return self.activation(affined)

    def backward(self, dout, **kwargs):
        dact = self.activation.backward(dout, **kwargs)
        dx = self.affine.backward(dact, **kwargs)

        return dx
