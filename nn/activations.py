from abc import ABC, abstractmethod
import numpy as np
from .math import sigmoid


class Activation(ABC):

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class Identity(Activation):

    def forward(self, x, *args, **kwargs):
        return x

    def backward(self, dout, *args, **kwargs):
        return dout


class Sigmoid(Activation):

    def forward(self, x, *args, **kwargs):
        self.sigmoids = sigmoid(x)

        return self.sigmoids

    def backward(self, dout, *args, **kwargs):
        dsig = self.sigmoids * (1.0 - self.sigmoids)
        dx = dout * dsig

        return dx

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


class Relu(Activation):

    def forward(self, x, *args, **kwargs):
        self.relu_mask = x > 0.0

        return x * self.relu_mask

    def backward(self, dout, *args, **kwargs):
        return dout * self.relu_mask


activations = {
    'identity': Identity,
    'sigmoid': Sigmoid,
    'relu': Relu,
}
