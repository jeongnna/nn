from abc import ABC, abstractmethod
import numpy as np


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
        self.sigmoid = self._sigmoid(x)

        return self.sigmoid

    def backward(self, dout, *args, **kwargs):
        dsig = self.sigmoid * (1 - self.sigmoid)
        dx = dout * dsig

        return dx

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


activations = {
    None: Identity,
    'identity': Identity,
    'sigmoid': Sigmoid,
}
