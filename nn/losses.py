from abc import ABC, abstractmethod
import numpy as np
from .math import sigmoid, softmax
from .metrics import cross_entropy


class Loss(ABC):

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class SoftmaxCrossEntropy(Loss):

    def forward(self, out, y):
        self.probs = softmax(out)

        return cross_entropy(self.probs, y)

    def backward(self, out, y):
        N = y.shape[0]

        dloss = self.probs
        dloss[range(N), y] -= 1
        dloss /= N

        return dloss


class NegativeLogLikelihood(Loss):

    def forward(self, out, y):
        N = y.shape[0]

        self.sigmoids = sigmoid(y.reshape(N, 1) * out)

        return np.sum(np.log(self.sigmoids)) / N

    def backward(self, out, y):
        N = y.shape[0]

        h = 1 - self.sigmoids
        dloss = - (y.reshape(N, 1) * h) / N

        return dloss
